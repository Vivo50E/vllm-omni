"""Minimal repro: does serving a prompt from LMCache change its own answer?

Every earlier comparison ran two engines and differed in more than one way. Here
a single engine answers the same prompt twice: round 1 is cold and round 2 is
served from LMCache, so the cache hit is the only variable.

Round 1's answer is the reference -- it is what this very engine produces with
no cache involved.
"""

import os

import pytest

from tests.engine import kv_offload_helpers as helpers

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

MODEL = "Qwen/Qwen2.5-Omni-3B"

_THINKER = {
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.8,
    "devices": "0",
    "enforce_eager": True,
    "async_chunk": False,
}
_DOWNSTREAM = {
    "1": {"devices": "0", "gpu_memory_utilization": 0.1, "enforce_eager": True},
    "2": {"devices": "0", "gpu_memory_utilization": 0.05, "enforce_eager": True},
}


@pytest.mark.parametrize("mode", ["off", "kv_only", "kv_and_hs"])
def test_second_round_matches_first(mode):
    """Round 2 must answer exactly what round 1 answered.

    ``off`` is the control: it shows whether repeating a prompt is stable at all
    here, so a failure elsewhere can be attributed to the cache hit.

    ``kv_only`` disables the hidden-state store. Text is produced by the thinker,
    which reads KV and not hidden states, so text still diverging there puts the
    fault in LMCache's KV restore rather than in the hidden-state path.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    rounds = helpers.run_rounds(
        model=MODEL,
        overrides=helpers.stage_overrides(
            lmcache=mode != "off",
            prefix_caching=False,
            hidden_states=mode == "kv_and_hs",
            thinker_extra=_THINKER,
            downstream_extra=_DOWNSTREAM,
        ),
        rounds=2,
        # One request by default: a batch of several flips greedy decoding on
        # its own, so the control fails and the comparison says nothing about
        # the restore. Raise it to probe multi-request behaviour.
        num_prompts=int(os.environ.get("OMNI_TEST_NUM_PROMPTS", "1")),
    )
    cold, served = rounds[0], rounds[1]

    for label, result in (("round 1", cold), ("round 2", served)):
        for prompt, entry in result.items():
            print(f"{label}: text={entry.get('text')!r} audio={helpers.audio_len(entry)}")
            del prompt

    assert cold and served, "a round produced no output"
    assert set(cold) == set(served), "the two rounds answered different prompts"

    problems = helpers.compare(cold, served)
    assert not problems, "round 2 diverged from this engine's own cold round:\n" + "\n".join(problems)
