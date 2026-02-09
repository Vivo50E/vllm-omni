"""Tests for OmniKvStoreBackend and OmniKvStoreFactory."""

import pytest
import torch

from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend
from vllm_omni.core.kv_store.factory import OmniKvStoreFactory


# ---------------------------------------------------------------------------
# A concrete mock backend for testing
# ---------------------------------------------------------------------------
class MockBackend(OmniKvStoreBackend):
    def __init__(self, config):
        super().__init__(config)
        self.storage: dict = {}

    def store_kv(self, request_id, kv_data):
        self.storage[request_id] = kv_data
        num_bytes = sum(t.numel() * t.element_size() for t in kv_data.get("kv_caches", []))
        self._update_store_stats(success=True, num_bytes=num_bytes)
        return True

    def load_kv(self, request_id, target_device="cuda"):
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
        stats = self._stats.copy()
        stats["backend_type"] = "mock"
        stats["num_stored"] = len(self.storage)
        return stats


# ---------------------------------------------------------------------------
# backend_interface tests
# ---------------------------------------------------------------------------
class TestOmniKvStoreBackend:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            OmniKvStoreBackend({})

    def test_mock_initialization(self):
        backend = MockBackend({"key": "val"})
        assert backend.config == {"key": "val"}
        assert backend._stats["total_stores"] == 0

    def test_store_and_load(self):
        backend = MockBackend({})
        kv = {
            "kv_caches": [torch.randn(2, 4, 16, 8, 64)],
            "block_ids": [0, 1, 2, 3],
            "seq_len": 64,
        }
        assert backend.store_kv("r1", kv) is True
        loaded = backend.load_kv("r1")
        assert loaded is not None
        assert loaded["seq_len"] == 64

    def test_load_nonexistent(self):
        backend = MockBackend({})
        assert backend.load_kv("missing") is None
        assert backend._stats["load_failures"] == 1

    def test_evict(self):
        backend = MockBackend({})
        kv = {"kv_caches": [torch.randn(2, 2, 16, 4, 32)], "seq_len": 32}
        backend.store_kv("r1", kv)
        assert backend.evict_kv("r1") is True
        assert backend.load_kv("r1") is None
        assert backend._stats["total_evictions"] == 1

    def test_evict_idempotent(self):
        backend = MockBackend({})
        assert backend.evict_kv("missing") is False

    def test_stats(self):
        backend = MockBackend({})
        kv = {"kv_caches": [torch.randn(2, 2, 16, 4, 32)], "seq_len": 16}
        backend.store_kv("r1", kv)
        backend.store_kv("r2", kv)
        backend.load_kv("r1")
        backend.load_kv("gone")
        backend.evict_kv("r1")

        stats = backend.get_stats()
        assert stats["total_stores"] == 2
        assert stats["total_loads"] == 2
        assert stats["load_failures"] == 1
        assert stats["total_evictions"] == 1
        assert stats["backend_type"] == "mock"
        assert stats["num_stored"] == 1  # r2 still there

    def test_repr(self):
        backend = MockBackend({"x": 1})
        assert "MockBackend" in repr(backend)


# ---------------------------------------------------------------------------
# factory tests
# ---------------------------------------------------------------------------
class TestOmniKvStoreFactory:
    @pytest.fixture(autouse=True)
    def _clean(self):
        OmniKvStoreFactory.clear_registry()
        yield
        OmniKvStoreFactory.clear_registry()

    def test_register_and_create(self):
        OmniKvStoreFactory.register("mock", MockBackend)
        backend = OmniKvStoreFactory.create("mock", {"backend_type": "mock"})
        assert isinstance(backend, MockBackend)

    def test_create_none_disabled(self):
        result = OmniKvStoreFactory.create("x", {"backend_type": "none"})
        assert result is None

    def test_create_default_disabled(self):
        result = OmniKvStoreFactory.create("x", {})
        assert result is None

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            OmniKvStoreFactory.create("x", {"backend_type": "unknown"})

    def test_list_backends(self):
        OmniKvStoreFactory.register("a", MockBackend)
        OmniKvStoreFactory.register("b", MockBackend)
        assert set(OmniKvStoreFactory.list_backends()) == {"a", "b"}

    def test_is_registered(self):
        assert not OmniKvStoreFactory.is_registered("mock")
        OmniKvStoreFactory.register("mock", MockBackend)
        assert OmniKvStoreFactory.is_registered("mock")

    def test_clear_registry(self):
        OmniKvStoreFactory.register("mock", MockBackend)
        OmniKvStoreFactory.clear_registry()
        assert OmniKvStoreFactory.list_backends() == []
