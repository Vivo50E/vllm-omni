"""
E2E accuracy guard for the LMCache KV + hidden-state offload path.

The offload path only pays off when a later request skips prefill for a cached
prefix. That shortcut is only safe if the restored KV *and* the restored hidden
states reproduce what a fresh prefill would have produced -- the talker consumes
the thinker's hidden states for the whole sequence, so a hole or a misplaced row
in the restored prefix silently degrades audio without failing anything.

This test pins that: the same prompts are decoded greedily with a fixed seed on
a plain baseline (no LMCache, no in-GPU prefix cache) and on the offload config
(LMCache + prefix caching, i.e. the combination several deploy profiles ship),
where a second round is served from cache. The outputs must match token for
token.
"""

import pytest
import torch

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"

# Long enough to span several LMCache chunks so a cache hit actually skips
# prefill; shared by every prompt so later requests hit the cached prefix.
SHARED_PREFIX = " ".join(
    f"Reference item {i}: the quick brown fox jumps over the lazy dog number {i}." for i in range(24)
)


def _stage_overrides(*, lmcache: bool, prefix_caching: bool, gpu_memory_utilization: float) -> dict:
    """Patch the thinker stage and keep the model's default talker/code2wav
    stages, since audio is what the hidden-state restore path feeds.

    Every stage is pinned to device 0: the default deploy config spreads the
    stages across two GPUs, and a 3B pipeline fits on one comfortably.
    """
    thinker: dict = {
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 4,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,
        "enable_prefix_caching": prefix_caching,
        "async_chunk": False,
        "devices": "0",
        "default_sampling_params": {
            "temperature": 0.0,
            "max_tokens": 48,
            "seed": 42,
        },
    }
    if lmcache:
        thinker["omni_kv_config"] = {"kv_store_config": {"lmcache_config": {"config_file": ""}}}
    # The talker ships with temperature 0.9 / top_p 0.8 / top_k 40. Under
    # sampling, the float-level differences that prefix caching legitimately
    # introduces (a cached prefix skips prefill, so attention is computed over
    # different chunk boundaries) flip a code and the audio diverges for reasons
    # that have nothing to do with the restore path. Pin every stage to greedy
    # so a mismatch actually means the restored state was wrong.
    greedy = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42, "max_tokens": 48}
    return {
        "0": thinker,
        "1": {
            "devices": "0",
            "gpu_memory_utilization": 0.1,
            "enforce_eager": True,
            "default_sampling_params": greedy,
        },
        "2": {
            "devices": "0",
            "gpu_memory_utilization": 0.05,
            "enforce_eager": True,
            "default_sampling_params": greedy,
        },
    }


def _prompts(n: int = 3) -> list[dict]:
    return [
        {
            "prompt": (
                f"<|im_start|>user\n{SHARED_PREFIX}\nUsing only the list above, "
                f"state item {i} verbatim.<|im_end|>\n<|im_start|>assistant\n"
            )
        }
        for i in range(n)
    ]


def _collect(omni, prompts) -> dict[str, dict]:
    """Run one round and index text/audio by prompt.

    Request ids are per-engine, so keying on them would pair a prompt's baseline
    output with a different prompt's cached output. Text outputs carry the
    prompt, so use that to resolve each request id to the prompt that produced
    it, and key the results on the prompt itself.
    """
    by_request: dict[str, dict] = {}
    prompt_of: dict[str, str] = {}
    for out in omni.generate(prompts, omni.default_sampling_params_list):
        entry = by_request.setdefault(out.request_id, {})
        if out.final_output_type == "text":
            entry["text"] = out.outputs[0].text
            if getattr(out, "prompt", None):
                prompt_of[out.request_id] = out.prompt
        elif out.final_output_type == "audio":
            audio = out.outputs[0].multimodal_output["audio"]
            entry["audio"] = audio.detach().cpu().float()

    results: dict[str, dict] = {}
    for rid, entry in by_request.items():
        key = prompt_of.get(rid)
        assert key is not None, f"no text output (and therefore no prompt) for request {rid}"
        results[key] = entry
    return results


def _run(*, lmcache: bool, prefix_caching: bool, rounds: int, gpu_memory_utilization: float) -> dict[str, dict]:
    """Build an engine, run ``rounds`` identical rounds, return the last one."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=DEFAULT_MODEL,
        stage_overrides=_stage_overrides(
            lmcache=lmcache,
            prefix_caching=prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        trust_remote_code=True,
        stage_init_timeout=600,
        batch_timeout=5,
        init_timeout=600,
    )
    try:
        prompts = _prompts()
        last: dict[str, dict] = {}
        for _ in range(rounds):
            last = _collect(omni, prompts)
        return last
    finally:
        omni.close()


def _audio_len(entry: dict) -> int:
    audio = entry.get("audio")
    return 0 if audio is None else int(audio.numel())


@pytest.mark.parametrize("gpu_memory_utilization", [0.8])
def test_kv_offload_matches_uncached_baseline(gpu_memory_utilization):
    """A cache-served round must reproduce the uncached baseline exactly."""
    pytest.importorskip("lmcache", reason="lmcache not installed")

    baseline = _run(
        lmcache=False,
        prefix_caching=False,
        rounds=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    # Round 1 populates LMCache; round 2 is served from the cached prefix, so it
    # exercises the KV + hidden-state restore path.
    cached = _run(
        lmcache=True,
        prefix_caching=True,
        rounds=2,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"

    # Audio is the signal that actually depends on restored hidden states: the
    # talker is conditioned on the thinker's HS for the cached prefix too.
    assert any(_audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    for i, prompt in enumerate(sorted(baseline)):
        want, got = baseline[prompt], cached[prompt]
        assert got["text"] == want["text"], (
            f"prompt {i}: text diverged after restore\nbaseline={want['text']!r}\ncached={got['text']!r}"
        )
        # An empty cached waveform means the talker was handed conditioning it
        # could not decode -- the exact failure a bad HS restore produces.
        assert _audio_len(got) == _audio_len(want), (
            f"prompt {i}: audio length differs after restore "
            f"({_audio_len(want)} -> {_audio_len(got)}); an empty cached waveform means "
            "the talker produced no codes for stage-2"
        )
        if _audio_len(want):
            assert torch.allclose(got["audio"], want["audio"], atol=1e-3, rtol=1e-3), (
                f"prompt {i}: audio diverged after hidden-state restore"
            )
