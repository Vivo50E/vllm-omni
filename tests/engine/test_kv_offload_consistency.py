"""E2E accuracy guard for the LMCache KV + hidden-state offload path.

A cache hit skips prefill, so the restored KV and hidden states must reproduce
what a fresh prefill would have produced -- otherwise the talker is conditioned
on a hole and audio degrades silently.

Runs the same prompts with and without LMCache, both with the in-GPU prefix
cache on so LMCache is the only variable, and requires identical output.
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
    """Patch the thinker; keep the default talker/code2wav stages so audio runs.

    All stages are pinned to device 0 (the default config spreads them over two).
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
    # The talker defaults to temperature 0.9, which amplifies any float-level
    # difference into a different audio sequence.
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
    """Run one round, keyed by prompt -- request ids are per-engine and would
    pair one prompt's baseline output with another's cached output."""
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
def test_kv_offload_matches_prefix_cached_baseline(gpu_memory_utilization):
    """Adding LMCache offload must not change what a cache hit produces."""
    pytest.importorskip("lmcache", reason="lmcache not installed")

    # Round 1 populates the cache; round 2 is served from it.
    baseline = _run(
        lmcache=False,
        prefix_caching=True,
        rounds=2,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    cached = _run(
        lmcache=True,
        prefix_caching=True,
        rounds=2,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"

    # Audio is what actually depends on the restored hidden states.
    assert any(_audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    for i, prompt in enumerate(sorted(baseline)):
        want, got = baseline[prompt], cached[prompt]
        assert got["text"] == want["text"], (
            f"prompt {i}: text diverged after restore\nbaseline={want['text']!r}\ncached={got['text']!r}"
        )
        assert _audio_len(got) == _audio_len(want), (
            f"prompt {i}: audio length differs after restore "
            f"({_audio_len(want)} -> {_audio_len(got)}); an empty cached waveform means "
            "the talker produced no codes for stage-2"
        )
        if _audio_len(want):
            assert torch.allclose(got["audio"], want["audio"], atol=1e-3, rtol=1e-3), (
                f"prompt {i}: audio diverged after hidden-state restore"
            )
