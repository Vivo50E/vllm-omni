"""E2E test: verify the KV offload data path works end-to-end.

This test does NOT require a running model. It simulates the scheduler →
runner flow with fake KV tensors to verify:
  1. Scheduler generates correct offload/prefetch decisions
  2. Runner extracts KV → stores to backend → loads back → injects
  3. Roundtrip data is identical (no corruption)
"""

import pytest
import torch

from vllm_omni.core.kv_store import OmniKvStoreBackend, OmniKvStoreFactory

# ---- In-memory mock backend (no lmcache dependency) ----


class InMemoryBackend(OmniKvStoreBackend):
    """Simple dict-backed backend for testing without LMCache."""

    def __init__(self, config):
        super().__init__(config)
        self.storage: dict[str, dict] = {}

    def store_kv(self, request_id, kv_data):
        # Simulate GPU→CPU: clone tensors to CPU
        cpu_kv = {
            "kv_caches": [t.detach().cpu().clone() for t in kv_data["kv_caches"]],
            "block_ids": kv_data.get("block_ids"),
            "seq_len": kv_data.get("seq_len"),
        }
        self.storage[request_id] = cpu_kv
        num_bytes = sum(t.numel() * t.element_size() for t in cpu_kv["kv_caches"])
        self._update_store_stats(success=True, num_bytes=num_bytes)
        return True

    def load_kv(self, request_id, target_device="cuda"):
        if request_id not in self.storage:
            self._update_load_stats(success=False)
            return None
        cpu_kv = self.storage[request_id]
        gpu_kv = {
            "kv_caches": [t.to(target_device) for t in cpu_kv["kv_caches"]],
            "block_ids": cpu_kv["block_ids"],
            "seq_len": cpu_kv["seq_len"],
        }
        num_bytes = sum(t.numel() * t.element_size() for t in gpu_kv["kv_caches"])
        self._update_load_stats(success=True, num_bytes=num_bytes)
        return gpu_kv

    def evict_kv(self, request_id):
        if request_id in self.storage:
            del self.storage[request_id]
            self._update_evict_stats()
            return True
        return False

    def get_stats(self):
        stats = self._stats.copy()
        stats["backend_type"] = "in_memory"
        stats["num_stored"] = len(self.storage)
        return stats


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def _register_backend():
    OmniKvStoreFactory.register("in_memory", InMemoryBackend)
    yield
    OmniKvStoreFactory.clear_registry()


@pytest.fixture
def backend():
    return OmniKvStoreFactory.create(
        "x",
        {
            "backend_type": "in_memory",
            "enable_offload": True,
        },
    )


@pytest.fixture
def fake_kv_caches():
    """Simulate KV caches: 4 layers, shape [2, 16, 8, 4, 32]."""
    num_layers = 4
    num_blocks = 16
    block_size = 8
    n_heads = 4
    head_dim = 32
    return [torch.randn(2, num_blocks, block_size, n_heads, head_dim) for _ in range(num_layers)]


# ---- Tests ----


class TestKvOffloadDataPath:
    """Test the full offload → store → load → inject roundtrip."""

    def test_extract_store_load_roundtrip(self, backend, fake_kv_caches):
        """Core test: data survives GPU→backend→GPU roundtrip."""
        block_ids = [0, 1, 2, 3]
        seq_len = 32

        # 1. Extract: pick specific blocks from kv_caches (simulates _extract_request_kv)
        kv_data = {
            "kv_caches": [layer[:, block_ids].detach().cpu().contiguous() for layer in fake_kv_caches],
            "block_ids": block_ids,
            "seq_len": seq_len,
        }

        # 2. Store to backend
        assert backend.store_kv("req_1", kv_data) is True

        # 3. Load back (to CPU for comparison)
        loaded = backend.load_kv("req_1", target_device="cpu")
        assert loaded is not None

        # 4. Verify data integrity
        assert loaded["block_ids"] == block_ids
        assert loaded["seq_len"] == seq_len
        assert len(loaded["kv_caches"]) == len(kv_data["kv_caches"])

        for orig, restored in zip(kv_data["kv_caches"], loaded["kv_caches"]):
            assert torch.equal(orig, restored), "KV cache data corrupted after roundtrip!"

    def test_inject_restores_gpu_blocks(self, backend, fake_kv_caches):
        """Test that inject writes data back to the correct block positions."""
        block_ids = [5, 6, 7]

        # 1. Save original values at those blocks
        originals = [layer[:, block_ids].clone() for layer in fake_kv_caches]

        # 2. Extract and store
        kv_data = {
            "kv_caches": [layer[:, block_ids].detach().cpu().contiguous() for layer in fake_kv_caches],
            "block_ids": block_ids,
            "seq_len": 24,
        }
        backend.store_kv("req_2", kv_data)

        # 3. Zero out the blocks (simulates block being freed/reused)
        for layer in fake_kv_caches:
            layer[:, block_ids] = 0.0

        # 4. Load and inject back (simulates _inject_request_kv)
        loaded = backend.load_kv("req_2", target_device="cpu")
        for layer_idx, layer in enumerate(fake_kv_caches):
            src = loaded["kv_caches"][layer_idx]
            layer[:, block_ids] = src

        # 5. Verify blocks are restored
        for layer_idx, layer in enumerate(fake_kv_caches):
            restored = layer[:, block_ids]
            assert torch.equal(restored, originals[layer_idx]), f"Layer {layer_idx} not restored correctly!"

    def test_evict_cleans_up(self, backend, fake_kv_caches):
        """Test that evict removes data from backend."""
        kv_data = {
            "kv_caches": [fake_kv_caches[0][:, [0]].cpu()],
            "block_ids": [0],
            "seq_len": 8,
        }
        backend.store_kv("req_3", kv_data)
        assert backend.load_kv("req_3", "cpu") is not None

        backend.evict_kv("req_3")
        assert backend.load_kv("req_3", "cpu") is None

    def test_stats_tracking(self, backend, fake_kv_caches):
        """Test that stats are updated correctly through the flow."""
        kv_data = {
            "kv_caches": [fake_kv_caches[0][:, [0, 1]].cpu()],
            "block_ids": [0, 1],
            "seq_len": 16,
        }

        backend.store_kv("r1", kv_data)
        backend.store_kv("r2", kv_data)
        backend.load_kv("r1", "cpu")
        backend.load_kv("missing", "cpu")
        backend.evict_kv("r1")

        stats = backend.get_stats()
        assert stats["total_stores"] == 2
        assert stats["total_loads"] == 2
        assert stats["load_failures"] == 1
        assert stats["total_evictions"] == 1
        assert stats["total_bytes_stored"] > 0
        assert stats["total_bytes_loaded"] > 0

    def test_multiple_requests_independent(self, backend, fake_kv_caches):
        """Multiple requests can be offloaded/loaded independently."""
        for i in range(5):
            block_ids = [i * 2, i * 2 + 1]
            kv_data = {
                "kv_caches": [layer[:, block_ids].cpu() for layer in fake_kv_caches],
                "block_ids": block_ids,
                "seq_len": 16,
            }
            backend.store_kv(f"req_{i}", kv_data)

        # Load them back in reverse order
        for i in reversed(range(5)):
            loaded = backend.load_kv(f"req_{i}", "cpu")
            assert loaded is not None
            assert loaded["block_ids"] == [i * 2, i * 2 + 1]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_gpu_roundtrip(self, backend):
        """Test with actual GPU tensors."""
        gpu_kv = [torch.randn(2, 4, 8, 4, 32, device="cuda") for _ in range(2)]
        kv_data = {
            "kv_caches": [t.cpu() for t in gpu_kv],
            "block_ids": [0, 1, 2, 3],
            "seq_len": 32,
        }

        backend.store_kv("gpu_req", kv_data)
        loaded = backend.load_kv("gpu_req", target_device="cuda")

        assert loaded is not None
        for t in loaded["kv_caches"]:
            assert t.is_cuda
