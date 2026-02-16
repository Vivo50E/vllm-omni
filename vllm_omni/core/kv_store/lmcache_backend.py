"""LMCache-based KV store backend implementation."""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend

logger = init_logger(__name__)

# Lazy check: avoid crash at import time if lmcache not installed
try:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    LMCACHE_AVAILABLE = True
except ImportError:
    LMCACHE_AVAILABLE = False


class LMCacheBackend(OmniKvStoreBackend):
    """KV store backend using LMCache v1 API.

    Supports:
        - cpu: Store in CPU memory (default)
        - redis: Store in Redis
        - ceph: Store in Ceph storage

    Config keys:
        backend_type: "lmcache"
        lmcache_backend: "cpu" | "redis" | "ceph"
        lmcache_memory_limit: str  (e.g. "10GB")
        lmcache_chunk_size: int    (default 256)
        lmcache_remote_url: str    (optional, for redis/ceph)
    """

    ENGINE_NAME = "omni_kv_store"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        if not LMCACHE_AVAILABLE:
            raise ImportError("lmcache is required for LMCacheBackend. Install with: pip install lmcache")

        # ---- extract config ------------------------------------------------
        lmcache_backend = config.get("lmcache_backend", "cpu")
        memory_limit = config.get("lmcache_memory_limit", "10GB")
        chunk_size = config.get("lmcache_chunk_size", 256)
        remote_url = config.get("lmcache_remote_url", None)

        # parse "10GB" -> 10.0
        if isinstance(memory_limit, str):
            memory_limit_gb = float(memory_limit.upper().replace("GB", "").strip())
        else:
            memory_limit_gb = float(memory_limit)

        # ---- build lmcache config dict -------------------------------------
        lmcache_config: dict[str, Any] = {
            "chunk_size": chunk_size,
            "local_cpu": lmcache_backend == "cpu",
            "max_local_cpu_size": memory_limit_gb,
        }
        if lmcache_backend in ("redis", "ceph") and remote_url:
            lmcache_config["remote_url"] = remote_url
            lmcache_config["remote_serde"] = "cachegen"

        # ---- create engine --------------------------------------------------
        try:
            self.engine = LMCacheEngineBuilder.get_or_create(
                engine_name=self.ENGINE_NAME,
                config_dict=lmcache_config,
            )
            logger.info(
                "Created LMCacheBackend: backend=%s, memory_limit=%.1fGB, chunk_size=%d",
                lmcache_backend,
                memory_limit_gb,
                chunk_size,
            )
        except Exception as e:
            logger.error("Failed to create LMCache engine: %s", e)
            raise RuntimeError(f"LMCache engine creation failed: {e}") from e

        # keep for get_stats()
        self.lmcache_backend = lmcache_backend
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.remote_url = remote_url

    # =====================================================================
    # Core interface
    # =====================================================================

    def store_kv(self, request_id: str, kv_data: dict[str, Any]) -> bool:
        try:
            lmcache_data = self._convert_to_lmcache_format(kv_data)

            num_bytes = sum(t.numel() * t.element_size() for t in lmcache_data.values() if isinstance(t, torch.Tensor))

            self.engine.put(request_id, lmcache_data)
            self._update_store_stats(success=True, num_bytes=num_bytes)

            logger.debug(
                "Stored KV for '%s': %d layers, %.2f MB",
                request_id,
                len(kv_data.get("kv_caches", [])),
                num_bytes / 1024 / 1024,
            )
            return True

        except Exception as e:
            logger.error("Failed to store KV for '%s': %s", request_id, e)
            self._update_store_stats(success=False)
            return False

    def load_kv(self, request_id: str, target_device: str = "cuda") -> dict[str, Any] | None:
        try:
            lmcache_data = self.engine.get(request_id)
            if lmcache_data is None:
                logger.debug("No KV cache found for '%s'", request_id)
                self._update_load_stats(success=False)
                return None

            kv_data = self._convert_from_lmcache_format(lmcache_data)
            kv_data = self._move_to_device(kv_data, target_device)

            num_bytes = sum(t.numel() * t.element_size() for t in kv_data["kv_caches"] if isinstance(t, torch.Tensor))
            self._update_load_stats(success=True, num_bytes=num_bytes)

            logger.debug(
                "Loaded KV for '%s': %d layers -> %s",
                request_id,
                len(kv_data["kv_caches"]),
                target_device,
            )
            return kv_data

        except Exception as e:
            logger.error("Failed to load KV for '%s': %s", request_id, e)
            self._update_load_stats(success=False)
            return None

    def evict_kv(self, request_id: str) -> bool:
        try:
            self.engine.put(request_id, None)
            self._update_evict_stats()
            logger.debug("Evicted KV for '%s'", request_id)
            return True
        except Exception as e:
            logger.warning("Failed to evict KV for '%s': %s", request_id, e)
            return False

    def get_stats(self) -> dict[str, Any]:
        stats = self._stats.copy()
        stats.update(
            {
                "backend_type": "lmcache",
                "lmcache_backend": self.lmcache_backend,
                "chunk_size": self.chunk_size,
                "memory_limit_gb": self.memory_limit_gb,
                "remote_url": self.remote_url,
            }
        )
        return stats

    # =====================================================================
    # Format conversion helpers
    # =====================================================================

    @staticmethod
    def _convert_to_lmcache_format(kv_data: dict[str, Any]) -> dict[str, Any]:
        """Our format -> LMCache format.

        Input:
            {"kv_caches": [tensor_l0, tensor_l1, ...],
             "block_ids": [...], "seq_len": N}
        Output:
            {"layer_0": tensor_l0, "layer_1": tensor_l1, ...,
             "_metadata": {"block_ids": [...], "seq_len": N}}
        """
        kv_caches = kv_data.get("kv_caches", [])
        lmcache_data: dict[str, Any] = {f"layer_{i}": kv_caches[i] for i in range(len(kv_caches))}

        metadata: dict[str, Any] = {}
        if "block_ids" in kv_data:
            metadata["block_ids"] = kv_data["block_ids"]
        if "seq_len" in kv_data:
            metadata["seq_len"] = kv_data["seq_len"]
        if "metadata" in kv_data:
            metadata.update(kv_data["metadata"])
        if metadata:
            lmcache_data["_metadata"] = metadata

        return lmcache_data

    @staticmethod
    def _convert_from_lmcache_format(
        lmcache_data: dict[str, Any],
    ) -> dict[str, Any]:
        """LMCache format -> our format."""
        kv_caches: list[torch.Tensor] = []
        idx = 0
        while f"layer_{idx}" in lmcache_data:
            kv_caches.append(lmcache_data[f"layer_{idx}"])
            idx += 1

        metadata = lmcache_data.get("_metadata", {})
        return {
            "kv_caches": kv_caches,
            "block_ids": metadata.get("block_ids"),
            "seq_len": metadata.get("seq_len"),
        }

    @staticmethod
    def _move_to_device(kv_data: dict[str, Any], device: str) -> dict[str, Any]:
        """Move all KV tensors to *device*."""
        return {
            "kv_caches": [t.to(device) if isinstance(t, torch.Tensor) else t for t in kv_data["kv_caches"]],
            "block_ids": kv_data.get("block_ids"),
            "seq_len": kv_data.get("seq_len"),
        }

    def close(self) -> None:
        """Explicitly destroy the LMCache engine used by this backend.

        Call this when the backend is no longer needed. Not invoked
        automatically to avoid tearing down a shared LMCache engine
        that may be used by other components.
        """
        if not LMCACHE_AVAILABLE:
            return
        try:
            LMCacheEngineBuilder.destroy(self.ENGINE_NAME)
            logger.info("Destroyed LMCache engine '%s'", self.ENGINE_NAME)
        except Exception:
            pass
