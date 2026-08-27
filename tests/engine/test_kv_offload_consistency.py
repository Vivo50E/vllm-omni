"""E2E accuracy guard for the LMCache KV + hidden-state offload path.

A cache hit skips prefill, so the restored KV and hidden states must reproduce
what a fresh prefill would have produced -- otherwise the talker is conditioned
on a hole and audio degrades silently.

Runs the same prompts with and without LMCache, both with the in-GPU prefix
cache on so LMCache is the only variable, and requires identical text and an
audio waveform wherever the baseline produced one.

The waveform itself is reported but not asserted: the talker decodes
autoregressively, so any float-level difference flips a code and the audio
diverges. Enabling the in-GPU prefix cache alone already moves it.
"""

import os
import pathlib
import tempfile
from contextlib import contextmanager

import pytest

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"

_FACTS = [
    "Mercury is the closest planet to the Sun and completes an orbit in about eighty-eight Earth days.",
    "The Pacific is the largest ocean on Earth and covers roughly a third of the planet's surface.",
    "Gold has the chemical symbol Au, taken from aurum, its name in Latin.",
    "A regular hexagon has six equal sides and tiles the plane without leaving any gaps.",
    "The Nile flows northward through Egypt and empties into the Mediterranean Sea.",
    "Diamond is the hardest natural mineral and sits at the top of the Mohs scale.",
    "Venus rotates in the opposite direction to most planets, so the Sun there rises in the west.",
    "The Amazon carries more water than any other river and drains a basin shared by several countries.",
    "Helium is lighter than air, which is why a balloon filled with it floats upward.",
    "An octave spans eight notes and doubles the frequency between its first and last pitch.",
    "Antarctica is the driest continent and most of it receives less precipitation than a desert.",
    "Copper conducts electricity better than iron, which is why household wiring uses it.",
    "The Sahara is the largest hot desert and stretches across much of northern Africa.",
    "Water boils at one hundred degrees Celsius at sea level, and lower where the pressure drops.",
    "A leap year has three hundred and sixty-six days because February gains an extra one.",
    "Everest is the highest mountain above sea level and sits on the border of Nepal and Tibet.",
    "Silver is the most reflective metal, which makes it useful for mirrors and telescopes.",
    "The Dead Sea lies below sea level and is salty enough that swimmers float without effort.",
    "Bamboo is the fastest growing plant and some species gain most of a metre in a day.",
    "Neon glows red in a discharge tube, which is where the classic sign colour comes from.",
    "The Baltic is the least salty sea because so many rivers empty fresh water into it.",
    "Graphite and diamond are both pure carbon and differ only in how their atoms are arranged.",
    "Jupiter has the shortest day of the planets and turns once in under ten hours.",
    "Mount Fuji is the highest peak in Japan and last erupted in the early eighteenth century.",
]

# Long enough to span several LMCache chunks so a cache hit actually skips
# prefill; shared by every prompt so later requests hit the cached prefix.
# The facts are deliberately unrelated: near-identical candidates leave the
# greedy argmax nearly tied, so any float-level difference flips the answer.
SHARED_PREFIX = " ".join(f"Fact {i}: {fact}" for i, fact in enumerate(_FACTS))


def _stage_overrides(
    *, lmcache: bool, prefix_caching: bool, gpu_memory_utilization: float, hidden_states: bool = True
) -> dict:
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
        lmcache_config: dict = {"config_file": ""}
        if not hidden_states:
            lmcache_config["enable_hidden_state_cache"] = False
        thinker["omni_kv_config"] = {"kv_store_config": {"lmcache_config": lmcache_config}}
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
                f"<|im_start|>user\n{SHARED_PREFIX}\nRepeat Fact {i} above word for "
                f"word.<|im_end|>\n<|im_start|>assistant\n"
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


def _run(
    *,
    lmcache: bool,
    prefix_caching: bool,
    rounds: int,
    gpu_memory_utilization: float,
    hidden_states: bool = True,
) -> dict[str, dict]:
    """Build an engine, run ``rounds`` identical rounds, return the last one."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=DEFAULT_MODEL,
        stage_overrides=_stage_overrides(
            lmcache=lmcache,
            prefix_caching=prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            hidden_states=hidden_states,
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


@contextmanager
def _restore_marker():
    """Point the runner's restore hook at a temp file so the test can tell
    whether a hidden-state restore actually happened."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "restores.tsv"
        previous = os.environ.get("OMNI_HS_RESTORE_MARKER_PATH")
        os.environ["OMNI_HS_RESTORE_MARKER_PATH"] = str(path)
        try:
            yield path
        finally:
            if previous is None:
                os.environ.pop("OMNI_HS_RESTORE_MARKER_PATH", None)
            else:
                os.environ["OMNI_HS_RESTORE_MARKER_PATH"] = previous


@pytest.mark.parametrize("gpu_memory_utilization", [0.8])
@pytest.mark.parametrize("hidden_states", [False, True], ids=["kv_only", "kv_and_hs"])
@pytest.mark.parametrize("prefix_caching", [False, True], ids=["lmcache_only", "with_prefix_cache"])
def test_kv_offload_matches_baseline(prefix_caching, hidden_states, gpu_memory_utilization):
    """Adding LMCache offload must not change what a cache hit produces.

    The kv_only case turns the hidden-state store off, so a failure there is in
    LMCache's KV restore rather than in the hidden-state path this PR adds.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    # Round 1 populates the cache; round 2 is served from it.
    baseline = _run(
        lmcache=False,
        prefix_caching=prefix_caching,
        rounds=2,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    with _restore_marker() as marker:
        cached = _run(
            lmcache=True,
            prefix_caching=prefix_caching,
            rounds=2,
            gpu_memory_utilization=gpu_memory_utilization,
            hidden_states=hidden_states,
        )
        restores = marker.read_text().splitlines() if marker.exists() else []

    if hidden_states and not prefix_caching:
        # Without the in-GPU prefix cache every restored token comes from
        # LMCache, so num_computed matches what it holds and the prepend runs.
        # With it on, num_computed also covers in-GPU blocks and the length
        # check makes the restore bail out -- covered by the other case.
        assert restores, "no hidden-state restore happened; the path under test never ran"

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"

    # Audio is what actually depends on the restored hidden states.
    assert any(_audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    problems = []
    for i, prompt in enumerate(sorted(baseline)):
        want, got = baseline[prompt], cached[prompt]
        if got["text"] != want["text"]:
            problems.append(f"prompt {i}: text differs\n  baseline={want['text']!r}\n  cached={got['text']!r}")

        want_len, got_len = _audio_len(want), _audio_len(got)
        if want_len and not got_len:
            problems.append(f"prompt {i}: baseline produced {want_len} audio samples, offload produced none")
        elif want_len != got_len:
            problems.append(f"prompt {i}: audio length {want_len} -> {got_len}")
        elif want_len:
            # Reported, not asserted: the talker decodes autoregressively, so any
            # float-level difference flips a code and the waveform diverges. Even
            # enabling the in-GPU prefix cache alone moves it.
            delta = (got["audio"] - want["audio"]).abs().max().item()
            print(f"prompt {i}: audio max|delta| = {delta:.3e}")

    assert not problems, "offload run diverged from the no-offload baseline:\n" + "\n".join(problems)
