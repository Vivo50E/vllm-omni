"""Unit tests for the two-phase KV offload scheduler logic.

Tests the offload trigger, state machine transitions, and ack processing
WITHOUT requiring a real model or GPU. All scheduler internals are mocked.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from vllm_omni.core.kv_store import OmniKvStoreBackend

# ---------------------------------------------------------------------------
# Lightweight fakes to avoid importing the full vLLM stack
# ---------------------------------------------------------------------------


class FakeRequest:
    """Minimal request mock matching vLLM v1 Request interface."""

    def __init__(
        self, request_id: str, num_computed_tokens: int = 100, num_prompt_tokens: int = 10, status: str = "RUNNING"
    ):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.status = status
        self.spec_token_ids = []
        self.num_preemptions = 0


class FakeWaitingQueue:
    """Minimal waiting queue mock."""

    def __init__(self):
        self._items: list[FakeRequest] = []

    def prepend_request(self, req: FakeRequest):
        self._items.insert(0, req)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


class FakeKvCacheManager:
    """Mock KV cache manager with controllable usage."""

    def __init__(self, usage: float = 0.0):
        self._usage = usage
        self._freed: list[str] = []
        self._block_ids: dict[str, tuple[list[int]]] = {}

    @property
    def usage(self) -> float:
        return self._usage

    def free(self, request):
        self._freed.append(request.request_id)

    def get_block_ids(self, req_id: str):
        return self._block_ids.get(req_id, ([0, 1, 2, 3],))


class InMemoryBackend(OmniKvStoreBackend):
    """Simple in-memory backend for testing."""

    def __init__(self, config=None):
        super().__init__(config or {})
        self.storage: dict[str, dict] = {}

    def store_kv(self, request_id, kv_data):
        self.storage[request_id] = kv_data
        self._update_store_stats(success=True, num_bytes=1)
        return True

    def load_kv(self, request_id, target_device="cpu"):
        if request_id not in self.storage:
            self._update_load_stats(success=False)
            return None
        self._update_load_stats(success=True)
        return self.storage[request_id]

    def evict_kv(self, request_id):
        if request_id in self.storage:
            del self.storage[request_id]
            self._update_evict_stats()
            return True
        return False

    def get_stats(self):
        return {**self._stats, "num_stored": len(self.storage)}


@dataclass
class FakeSchedulerOutput:
    """Minimal SchedulerOutput mock."""

    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class FakeModelRunnerOutput:
    """Minimal ModelRunnerOutput mock."""

    kv_extracted_req_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Helper: create a scheduler-like object with just the offload methods
# ---------------------------------------------------------------------------


class OffloadLogicHarness:
    """Extracts the offload logic from OmniARScheduler for isolated testing.

    Instead of instantiating the full scheduler (which requires vllm_config,
    GPU, etc.), this harness replicates the exact state and methods we need.
    """

    def __init__(self, usage: float = 0.0, threshold: float = 0.85, max_offloads: int = 2, num_waiting: int = 0):
        # State from OmniARScheduler.__init__
        self.kv_store_backend = InMemoryBackend()
        self.offloaded_requests: set[str] = set()
        self.offloaded_metadata: dict[str, dict] = {}
        self.offload_pending_extract: set[str] = set()
        self.offload_pending_free: set[str] = set()
        self.offload_usage_threshold = threshold
        self.max_offloads_per_step = max_offloads

        # Inter-stage transfer state (needed for exclusion)
        self.waiting_for_transfer_free: set[str] = set()
        self.active_kv_transfers: set[str] = set()

        # Mock dependencies
        self.kv_cache_manager = FakeKvCacheManager(usage)
        self.encoder_cache_manager = MagicMock()
        self.running: list[FakeRequest] = []
        self.waiting = FakeWaitingQueue()
        self.requests: dict[str, FakeRequest] = {}

        # Add waiting requests
        for i in range(num_waiting):
            req = FakeRequest(f"waiting_{i}", num_computed_tokens=0)
            self.waiting.prepend_request(req)

    def add_running(self, req_id: str, num_computed: int = 100, num_prompt: int = 10) -> FakeRequest:
        req = FakeRequest(req_id, num_computed, num_prompt)
        self.running.append(req)
        self.requests[req_id] = req
        return req

    def generate_offload_decisions(self, scheduled_ids: set[str] | None = None):
        """Call _generate_offload_decisions with a fake scheduler output."""
        if scheduled_ids is None:
            scheduled_ids = {r.request_id for r in self.running}
        output = FakeSchedulerOutput(num_scheduled_tokens={rid: 1 for rid in scheduled_ids})
        # Import and call the actual method logic (replicated here to avoid
        # needing the full class hierarchy)
        return self._generate_offload_decisions(output)

    def _generate_offload_decisions(self, scheduler_output):
        """Exact copy of OmniARScheduler._generate_offload_decisions."""
        offload_ids = []
        load_ids = []
        load_new_block_ids = {}

        scheduled = set(scheduler_output.num_scheduled_tokens.keys())

        for req_id in list(self.offloaded_requests):
            if req_id in scheduled:
                load_ids.append(req_id)
                try:
                    block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                    if block_ids_tuple:
                        load_new_block_ids[req_id] = list(block_ids_tuple[0])
                except Exception:
                    pass
                self.offloaded_requests.discard(req_id)
                self.offloaded_metadata.pop(req_id, None)

        if self.kv_cache_manager.usage >= self.offload_usage_threshold and len(self.waiting) > 0:
            excluded = (
                self.offload_pending_extract
                | self.offload_pending_free
                | self.offloaded_requests
                | self.waiting_for_transfer_free
                | self.active_kv_transfers
            )
            candidates = []
            for req in self.running:
                rid = req.request_id
                if rid in scheduled and rid not in excluded and req.num_computed_tokens > req.num_prompt_tokens:
                    candidates.append(req)
            candidates.sort(key=lambda r: r.num_computed_tokens, reverse=True)

            for req in candidates[: self.max_offloads_per_step]:
                offload_ids.append(req.request_id)
                self.offload_pending_extract.add(req.request_id)

        return {
            "offload_req_ids": offload_ids,
            "load_req_ids": load_ids,
            "load_new_block_ids": load_new_block_ids,
        }

    def execute_pending_frees(self):
        """Exact copy of OmniARScheduler._execute_pending_frees."""
        if not self.offload_pending_free:
            return

        for req_id in list(self.offload_pending_free):
            request = self.requests.get(req_id)
            if request is None:
                self.offload_pending_free.discard(req_id)
                continue

            seq_len = request.num_computed_tokens
            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)
            request.status = "PREEMPTED"
            request.num_computed_tokens = max(seq_len - 1, 0)
            request.spec_token_ids.clear()
            request.num_preemptions += 1
            self.running = [r for r in self.running if r.request_id != req_id]
            self.waiting.prepend_request(request)
            self.offloaded_requests.add(req_id)
            self.offloaded_metadata[req_id] = {"seq_len": seq_len}

        self.offload_pending_free.clear()

    def process_acks(self, kv_extracted_ids: list[str]):
        """Simulate the ack processing from update_from_output."""
        for req_id in kv_extracted_ids:
            if req_id in self.offload_pending_extract:
                self.offload_pending_extract.discard(req_id)
                self.offload_pending_free.add(req_id)


# ===========================================================================
# Tests
# ===========================================================================


class TestOffloadTrigger:
    """Test that offload triggers correctly based on usage and waiting."""

    def test_no_trigger_below_threshold(self):
        h = OffloadLogicHarness(usage=0.5, threshold=0.85, num_waiting=3)
        h.add_running("r1", num_computed=100)
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == []

    def test_no_trigger_no_waiting(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=0)
        h.add_running("r1", num_computed=100)
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == []

    def test_trigger_above_threshold_with_waiting(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        h.add_running("r1", num_computed=200)
        h.add_running("r2", num_computed=100)
        decisions = h.generate_offload_decisions()
        assert len(decisions["offload_req_ids"]) == 2
        # Largest KV first
        assert decisions["offload_req_ids"][0] == "r1"
        assert decisions["offload_req_ids"][1] == "r2"

    def test_cap_max_offloads_per_step(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, max_offloads=1, num_waiting=2)
        h.add_running("r1", num_computed=200)
        h.add_running("r2", num_computed=100)
        decisions = h.generate_offload_decisions()
        assert len(decisions["offload_req_ids"]) == 1
        assert decisions["offload_req_ids"][0] == "r1"  # Largest first

    def test_exclude_prefill_only_requests(self):
        """Requests still in prefill (num_computed <= num_prompt) are not candidates."""
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        h.add_running("prefill", num_computed=10, num_prompt=10)  # Still prefilling
        h.add_running("decode", num_computed=50, num_prompt=10)  # In decode
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == ["decode"]

    def test_exclude_already_pending(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        h.add_running("r1", num_computed=200)
        h.add_running("r2", num_computed=100)
        h.offload_pending_extract.add("r1")  # Already pending
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == ["r2"]

    def test_exclude_transfer_active(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        h.add_running("r1", num_computed=200)
        h.waiting_for_transfer_free.add("r1")  # Active inter-stage transfer
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == []

    def test_adds_to_pending_extract(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        h.add_running("r1", num_computed=100)
        h.generate_offload_decisions()
        assert "r1" in h.offload_pending_extract


class TestStateMachine:
    """Test the full RUNNING → PENDING_EXTRACT → PENDING_FREE → OFFLOADED → RUNNING cycle."""

    def test_full_cycle(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        req = h.add_running("r1", num_computed=100, num_prompt=10)

        # Step 1: Trigger → PENDING_EXTRACT
        decisions = h.generate_offload_decisions()
        assert "r1" in decisions["offload_req_ids"]
        assert "r1" in h.offload_pending_extract
        assert "r1" not in h.offload_pending_free

        # Step 2: Runner acks extraction → PENDING_FREE
        h.process_acks(["r1"])
        assert "r1" not in h.offload_pending_extract
        assert "r1" in h.offload_pending_free

        # Step 3: Execute pending frees → OFFLOADED (in waiting)
        h.execute_pending_frees()
        assert "r1" in h.offloaded_requests
        assert "r1" in h.offloaded_metadata
        assert req.num_computed_tokens == 99  # seq_len - 1
        assert req.status == "PREEMPTED"
        assert req.request_id not in [r.request_id for r in h.running]
        assert len(h.waiting) == 3  # 2 original + r1
        assert len(h.offload_pending_free) == 0
        assert "r1" in h.kv_cache_manager._freed

    def test_load_on_reschedule(self):
        h = OffloadLogicHarness(usage=0.5, threshold=0.85, num_waiting=0)
        h.offloaded_requests.add("r1")
        h.offloaded_metadata["r1"] = {"seq_len": 100}
        h.kv_cache_manager._block_ids["r1"] = ([10, 11, 12, 13],)

        # r1 is re-scheduled (appears in scheduled set)
        decisions = h.generate_offload_decisions(scheduled_ids={"r1", "r2"})
        assert decisions["load_req_ids"] == ["r1"]
        assert decisions["load_new_block_ids"]["r1"] == [10, 11, 12, 13]
        assert "r1" not in h.offloaded_requests

    def test_pending_free_clears_after_execute(self):
        h = OffloadLogicHarness(usage=0.5, threshold=0.85, num_waiting=0)
        h.add_running("r1", num_computed=50)
        h.offload_pending_free.add("r1")
        h.execute_pending_frees()
        assert len(h.offload_pending_free) == 0

    def test_pending_free_handles_missing_request(self):
        """Request finished/aborted before free phase."""
        h = OffloadLogicHarness(usage=0.5, threshold=0.85, num_waiting=0)
        h.offload_pending_free.add("gone")
        # Should not crash
        h.execute_pending_frees()
        assert len(h.offload_pending_free) == 0


class TestAckProcessing:
    """Test ack processing distinguishes offload acks from transfer acks."""

    def test_offload_ack_moves_to_pending_free(self):
        h = OffloadLogicHarness()
        h.offload_pending_extract.add("r1")
        h.process_acks(["r1"])
        assert "r1" not in h.offload_pending_extract
        assert "r1" in h.offload_pending_free

    def test_unknown_ack_ignored(self):
        """Ack for unknown request (not in pending_extract) is ignored."""
        h = OffloadLogicHarness()
        h.process_acks(["unknown"])
        assert len(h.offload_pending_free) == 0

    def test_mixed_acks(self):
        h = OffloadLogicHarness()
        h.offload_pending_extract.add("offload_1")
        h.process_acks(["offload_1", "transfer_1"])
        assert "offload_1" in h.offload_pending_free
        # transfer_1 is not in pending_extract, so it's ignored by our logic
        assert "transfer_1" not in h.offload_pending_free


class TestNumComputedTokensPreservation:
    """Test the seq_len - 1 trick for resume."""

    def test_preserves_computed_minus_one(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, num_waiting=2)
        req = h.add_running("r1", num_computed=150, num_prompt=10)

        # Full cycle: trigger → ack → free
        h.generate_offload_decisions()
        h.process_acks(["r1"])
        h.execute_pending_frees()

        assert req.num_computed_tokens == 149  # 150 - 1
        assert h.offloaded_metadata["r1"]["seq_len"] == 150

    def test_zero_computed_becomes_zero(self):
        h = OffloadLogicHarness()
        req = FakeRequest("r1", num_computed_tokens=0, num_prompt_tokens=0)
        h.running.append(req)
        h.requests["r1"] = req
        h.offload_pending_free.add("r1")
        h.execute_pending_frees()
        assert req.num_computed_tokens == 0  # max(0-1, 0) = 0


class TestCandidateOrdering:
    """Test that candidates are sorted by KV size (num_computed_tokens desc)."""

    def test_largest_kv_first(self):
        h = OffloadLogicHarness(usage=0.9, threshold=0.85, max_offloads=3, num_waiting=2)
        h.add_running("small", num_computed=50)
        h.add_running("large", num_computed=500)
        h.add_running("medium", num_computed=200)
        decisions = h.generate_offload_decisions()
        assert decisions["offload_req_ids"] == ["large", "medium", "small"]
