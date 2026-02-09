"""Tests for LMCacheBackend format conversion helpers.

These tests verify the data format conversion logic without requiring
an actual LMCache installation (the helpers are static methods).
"""

import pytest
import torch

from vllm_omni.core.kv_store.lmcache_backend import LMCacheBackend


class TestConvertToLmcacheFormat:
    """Test _convert_to_lmcache_format (our format -> LMCache format)."""

    def test_basic(self):
        kv_data = {
            "kv_caches": [
                torch.randn(2, 4, 16, 8, 64),
                torch.randn(2, 4, 16, 8, 64),
            ],
            "block_ids": [0, 1, 2, 3],
            "seq_len": 64,
        }
        result = LMCacheBackend._convert_to_lmcache_format(kv_data)

        assert "layer_0" in result
        assert "layer_1" in result
        assert result["layer_0"].shape == (2, 4, 16, 8, 64)
        assert result["_metadata"]["block_ids"] == [0, 1, 2, 3]
        assert result["_metadata"]["seq_len"] == 64

    def test_empty_kv_caches(self):
        kv_data = {"kv_caches": []}
        result = LMCacheBackend._convert_to_lmcache_format(kv_data)
        assert "layer_0" not in result

    def test_no_metadata(self):
        kv_data = {"kv_caches": [torch.randn(2, 1, 16, 4, 32)]}
        result = LMCacheBackend._convert_to_lmcache_format(kv_data)
        assert "layer_0" in result
        assert "_metadata" not in result

    def test_many_layers(self):
        num_layers = 32
        kv_data = {
            "kv_caches": [torch.randn(2, 2, 16, 4, 32) for _ in range(num_layers)],
            "seq_len": 32,
        }
        result = LMCacheBackend._convert_to_lmcache_format(kv_data)
        assert f"layer_{num_layers - 1}" in result
        assert f"layer_{num_layers}" not in result


class TestConvertFromLmcacheFormat:
    """Test _convert_from_lmcache_format (LMCache format -> our format)."""

    def test_roundtrip(self):
        """Store -> convert -> restore should be lossless."""
        original = {
            "kv_caches": [
                torch.randn(2, 4, 16, 8, 64),
                torch.randn(2, 4, 16, 8, 64),
                torch.randn(2, 4, 16, 8, 64),
            ],
            "block_ids": [10, 20, 30],
            "seq_len": 48,
        }
        lmcache_fmt = LMCacheBackend._convert_to_lmcache_format(original)
        restored = LMCacheBackend._convert_from_lmcache_format(lmcache_fmt)

        assert len(restored["kv_caches"]) == 3
        assert restored["block_ids"] == [10, 20, 30]
        assert restored["seq_len"] == 48
        for orig_t, rest_t in zip(original["kv_caches"], restored["kv_caches"]):
            assert torch.equal(orig_t, rest_t)

    def test_no_metadata(self):
        lmcache_data = {"layer_0": torch.randn(2, 1, 16, 4, 32)}
        result = LMCacheBackend._convert_from_lmcache_format(lmcache_data)
        assert len(result["kv_caches"]) == 1
        assert result["block_ids"] is None
        assert result["seq_len"] is None

    def test_empty(self):
        result = LMCacheBackend._convert_from_lmcache_format({})
        assert result["kv_caches"] == []


class TestMoveToDevice:
    """Test _move_to_device."""

    def test_cpu_to_cpu(self):
        kv_data = {
            "kv_caches": [torch.randn(2, 2, 16, 4, 32)],
            "block_ids": [0],
            "seq_len": 32,
        }
        result = LMCacheBackend._move_to_device(kv_data, "cpu")
        assert result["kv_caches"][0].device == torch.device("cpu")
        assert result["block_ids"] == [0]
        assert result["seq_len"] == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_cpu_to_cuda(self):
        kv_data = {
            "kv_caches": [torch.randn(2, 2, 16, 4, 32)],
            "block_ids": [0],
            "seq_len": 16,
        }
        result = LMCacheBackend._move_to_device(kv_data, "cuda")
        assert result["kv_caches"][0].is_cuda
