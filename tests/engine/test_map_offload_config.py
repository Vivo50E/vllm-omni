"""Unit tests for OmniEngineArgs._map_offload_config() MultiConnector support.

Tests the config bridge that maps omni_kv_config YAML surface to vLLM's
KV transfer infrastructure (OffloadingConnector, LMCacheConnectorV1,
MultiConnector).
"""

import pytest

from vllm_omni.engine.arg_utils import (
    OmniEngineArgs,
    _build_connector_list,
    _map_offload_config,
)


class TestBuildConnectorList:
    """Test _build_connector_list() helper."""

    def test_offload_and_lmcache(self):
        kv_store = {"enable_offload": True, "max_cpu_memory_gb": 8.0}
        lmcache_config = {"config_file": "/tmp/lmcache.yaml"}

        connectors = _build_connector_list(kv_store, lmcache_config)

        assert len(connectors) == 2
        assert connectors[0]["kv_connector"] == "OffloadingConnector"
        assert connectors[0]["kv_connector_extra_config"]["cpu_bytes_to_use"] == 8.0 * (1 << 30)
        assert connectors[1]["kv_connector"] == "LMCacheConnectorV1"
        assert connectors[1]["kv_connector_extra_config"]["config_file"] == "/tmp/lmcache.yaml"

    def test_lmcache_only(self):
        kv_store = {"enable_offload": False}
        lmcache_config = {"config_file": "/tmp/lmcache.yaml"}

        connectors = _build_connector_list(kv_store, lmcache_config)

        assert len(connectors) == 1
        assert connectors[0]["kv_connector"] == "LMCacheConnectorV1"
        assert connectors[0]["kv_connector_extra_config"]["config_file"] == "/tmp/lmcache.yaml"

    def test_offload_without_max_cpu_gb(self):
        """Standalone test: verifies default cpu_bytes_to_use when max_cpu_memory_gb
        is omitted. Note: _map_offload_config guards with `if lmcache_config:`
        so this exact call (empty lmcache_config) won't happen in practice."""
        kv_store = {"enable_offload": True}
        lmcache_config = {}

        connectors = _build_connector_list(kv_store, lmcache_config)

        assert len(connectors) == 2
        assert connectors[0]["kv_connector"] == "OffloadingConnector"
        assert connectors[0]["kv_connector_extra_config"]["cpu_bytes_to_use"] == 10.0 * (1 << 30)
        assert connectors[1]["kv_connector"] == "LMCacheConnectorV1"

    def test_lmcache_config_is_copied(self):
        """Ensure lmcache_config dict is copied, not mutated."""
        lmcache_config = {"config_file": "/tmp/lmcache.yaml", "chunk_size": 256}
        kv_store = {}

        connectors = _build_connector_list(kv_store, lmcache_config)

        # Modifying the returned config should not affect the original
        connectors[0]["kv_connector_extra_config"]["extra_key"] = "value"
        assert "extra_key" not in lmcache_config

    def test_no_offload_no_lmcache_in_kv_store(self):
        """When kv_store has no offload, only lmcache connector is returned."""
        kv_store = {"max_cpu_memory_gb": 10.0}  # offload not explicitly enabled
        lmcache_config = {"config_file": "/tmp/lmcache.yaml"}

        connectors = _build_connector_list(kv_store, lmcache_config)

        assert len(connectors) == 1
        assert connectors[0]["kv_connector"] == "LMCacheConnectorV1"


class TestMapOffloadConfigMultiConnector:
    """Test the full _map_offload_config() with MultiConnector scenarios.

    These tests mock the KVTransferConfig import to avoid requiring vLLM
    to be installed in the test environment.
    """

    def _make_args_with_config(self, omni_kv_config):
        """Create a minimal OmniEngineArgs-like object for testing _map_offload_config.

        We bypass __post_init__ to avoid vLLM model resolution and directly
        test the config mapping logic.
        """
        # Use object.__new__ to skip __init__/__post_init__
        args = object.__new__(OmniEngineArgs)
        args.omni_kv_config = omni_kv_config
        args.kv_offloading_size = None
        args.kv_transfer_config = None
        args.disable_hybrid_kv_cache_manager = False
        return args

    def test_offload_only_sets_kv_offloading_size(self):
        """Offload-only mode should use the simple kv_offloading_size path."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "enable_offload": True,
                    "max_cpu_memory_gb": 5.0,
                }
            }
        )

        _map_offload_config(args)

        assert args.kv_offloading_size == 5.0
        assert args.disable_hybrid_kv_cache_manager is True
        assert args.kv_transfer_config is None

    def test_offload_only_default_memory(self):
        """Offload with no max_cpu_memory_gb should default to 10.0."""
        args = self._make_args_with_config({"kv_store_config": {"enable_offload": True}})

        _map_offload_config(args)

        assert args.kv_offloading_size == 10.0

    def test_lmcache_only_sets_kv_transfer_config(self):
        """LMCache without offload should set kv_transfer_config directly."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "lmcache_config": {
                        "config_file": "/tmp/lmcache.yaml",
                    }
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        assert args.kv_transfer_config is not None
        assert args.kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        assert args.kv_transfer_config.kv_connector_extra_config == {"config_file": "/tmp/lmcache.yaml"}
        assert args.kv_transfer_config.kv_role == "kv_both"
        assert args.kv_offloading_size is None
        assert args.disable_hybrid_kv_cache_manager is False

    def test_offload_plus_lmcache_sets_multi_connector(self):
        """Offload + LMCache should generate MultiConnector config."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "enable_offload": True,
                    "max_cpu_memory_gb": 8.0,
                    "lmcache_config": {
                        "config_file": "/tmp/lmcache.yaml",
                    },
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        assert args.kv_transfer_config is not None
        assert args.kv_transfer_config.kv_connector == "MultiConnector"
        extra = args.kv_transfer_config.kv_connector_extra_config
        assert "connectors" in extra
        connectors = extra["connectors"]
        assert len(connectors) == 2
        assert connectors[0]["kv_connector"] == "OffloadingConnector"
        assert connectors[0]["kv_connector_extra_config"]["cpu_bytes_to_use"] == 8.0 * (1 << 30)
        assert connectors[1]["kv_connector"] == "LMCacheConnectorV1"
        assert connectors[1]["kv_connector_extra_config"]["config_file"] == "/tmp/lmcache.yaml"

        # MultiConnector mode does NOT set kv_offloading_size (avoids
        # VllmConfig._post_init_kv_transfer_config overriding kv_connector)
        assert args.kv_offloading_size is None
        assert args.disable_hybrid_kv_cache_manager is True

    def test_no_config_is_noop(self):
        """No omni_kv_config should be a no-op."""
        args = self._make_args_with_config(None)

        _map_offload_config(args)

        assert args.kv_offloading_size is None
        assert args.kv_transfer_config is None

    def test_empty_kv_store_config_is_noop(self):
        """Empty kv_store_config should be a no-op."""
        args = self._make_args_with_config({"kv_store_config": {}})

        _map_offload_config(args)

        assert args.kv_offloading_size is None
        assert args.kv_transfer_config is None

    def test_skip_if_kv_offloading_size_already_set(self):
        """Should not override kv_offloading_size if already set."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "enable_offload": True,
                    "max_cpu_memory_gb": 20.0,
                }
            }
        )
        args.kv_offloading_size = 5.0  # Pre-set

        _map_offload_config(args)

        # Should keep the pre-set value, not override with 20.0
        assert args.kv_offloading_size == 5.0

    def test_lmcache_config_with_multiple_fields(self):
        """LMCache config with multiple fields should all be passed through."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "lmcache_config": {
                        "config_file": "/tmp/lmcache.yaml",
                        "chunk_size": 256,
                        "max_local_cache_size": "10GiB",
                    }
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        cfg = args.kv_transfer_config
        assert cfg.kv_connector == "LMCacheConnectorV1"
        assert cfg.kv_role == "kv_both"
        assert cfg.kv_connector_extra_config["config_file"] == "/tmp/lmcache.yaml"
        assert cfg.kv_connector_extra_config["chunk_size"] == 256
        assert cfg.kv_connector_extra_config["max_local_cache_size"] == "10GiB"

    def test_custom_kv_role(self):
        """Custom kv_role should be passed to KVTransferConfig."""
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "kv_role": "kv_producer",
                    "lmcache_config": {"config_file": "/tmp/lmcache.yaml"},
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        assert args.kv_transfer_config.kv_role == "kv_producer"
