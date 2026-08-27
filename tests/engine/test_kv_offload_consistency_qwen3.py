"""E2E accuracy guard for the LMCache offload path on Qwen3-Omni.

Same comparison as the Qwen2.5-Omni test, on the model that actually exercises
the multimodal capture taps: ``_lmcache_hs_mm_keys`` is only populated for
models exposing ``talker_config.accept_hidden_layer``, so on Qwen2.5-Omni the
whole mm-layer branch is dead code.

The 30B-A3B thinker needs a card to itself, so the default two-GPU stage layout
is kept rather than pinning everything to device 0.
"""

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

_THINKER = {"max_model_len": 2048, "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.85}
_DOWNSTREAM = {"1": {"gpu_memory_utilization": 0.3}, "2": {"gpu_memory_utilization": 0.2}}


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
        init_timeout=1800,
    )


@pytest.mark.parametrize("hidden_states", [False, True], ids=["kv_only", "kv_and_hs"])
def test_kv_offload_matches_baseline(hidden_states):
    """Offload must not change what a cache hit produces.

    Runs without the in-GPU prefix cache so every restored token comes from
    LMCache; kv_only additionally turns the hidden-state store off, separating
    LMCache's KV restore from the hidden-state path this PR adds.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    baseline = _run(lmcache=False, prefix_caching=False, rounds=2)
    with helpers.restore_marker() as marker:
        cached = _run(lmcache=True, prefix_caching=False, rounds=2, hidden_states=hidden_states)
        restores = marker.read_text().splitlines() if marker.exists() else []

    if hidden_states:
        assert restores, "no hidden-state restore happened; the path under test never ran"
        # Marker lines are "<req_id>\t<num_computed>\t<comma-joined tap keys>".
        # Qwen3-Omni captures layer 0 and accept_hidden_layer on top of the
        # final "hidden" tap, so a restore here must cover more than "hidden".
        taps = {tap for line in restores for tap in line.split("\t")[-1].split(",")}
        assert taps - {"hidden"}, f"only the final hidden tap was restored; got {sorted(taps)}"

    assert baseline, "baseline produced no output"
    assert cached, "offload run produced no output"
    assert set(baseline) == set(cached), "the two runs answered different prompts"
    assert any(helpers.audio_len(e) for e in baseline.values()), (
        "baseline produced no audio; the HS restore path is untested without it"
    )

    problems = helpers.compare(baseline, cached)
    assert not problems, "offload run diverged from the no-offload baseline:\n" + "\n".join(problems)
