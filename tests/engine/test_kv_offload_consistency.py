"""E2E accuracy guard for the LMCache KV + hidden-state offload path (Qwen2.5-Omni).

A cache hit skips prefill, so the restored KV and hidden states must reproduce
what a fresh prefill would have produced -- otherwise the talker is conditioned
on a hole and audio degrades silently.

Runs the same prompts with and without LMCache and requires identical text and
an audio waveform wherever the baseline produced one.
"""

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

MODEL = "Qwen/Qwen2.5-Omni-3B"

# 3B fits on one card, so pin every stage there; the default config spreads them.
_THINKER = {"max_model_len": 1024, "max_num_batched_tokens": 1024, "gpu_memory_utilization": 0.8, "devices": "0"}
_DOWNSTREAM = {
    "1": {"devices": "0", "gpu_memory_utilization": 0.1},
    "2": {"devices": "0", "gpu_memory_utilization": 0.05},
}


def _run(*, lmcache: bool, prefix_caching: bool, rounds: int, hidden_states: bool = True) -> dict[str, dict]:
    return helpers.run(
        model=MODEL,
        overrides=helpers.stage_overrides(
            lmcache=lmcache,
            prefix_caching=prefix_caching,
            hidden_states=hidden_states,
            thinker_extra=_THINKER,
            downstream_extra=_DOWNSTREAM,
        ),
        rounds=rounds,
    )


@pytest.mark.parametrize("hidden_states", [False, True], ids=["kv_only", "kv_and_hs"])
@pytest.mark.parametrize("prefix_caching", [False, True], ids=["lmcache_only", "with_prefix_cache"])
def test_kv_offload_matches_baseline(prefix_caching, hidden_states):
    """Adding LMCache offload must not change what a cache hit produces.

    The kv_only case turns the hidden-state store off, so a failure there is in
    LMCache's KV restore rather than in the hidden-state path this PR adds.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    # Round 1 populates the cache; round 2 is served from it.
    baseline = _run(lmcache=False, prefix_caching=prefix_caching, rounds=2)
    with helpers.restore_marker() as marker:
        cached = _run(
            lmcache=True,
            prefix_caching=prefix_caching,
            rounds=2,
            hidden_states=hidden_states,
        )
        restores = marker.read_text().splitlines() if marker.exists() else []

    if hidden_states and not prefix_caching:
        # Without the in-GPU prefix cache every restored token comes from
        # LMCache, so num_computed matches what it holds and the prepend runs.
        assert restores, "no hidden-state restore happened; the path under test never ran"

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"
    assert any(helpers.audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    problems = helpers.compare(baseline, cached)
    assert not problems, "offload run diverged from the no-offload baseline:\n" + "\n".join(problems)
