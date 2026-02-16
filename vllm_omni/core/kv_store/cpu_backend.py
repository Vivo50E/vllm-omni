"""Simple CPU memory backend for KV cache offloading.

Stores KV cache tensors directly in CPU memory using a Python dict.
No external dependencies required. Suitable for single-node offloading.

For production with Redis/Ceph support, use LMCacheBackend instead.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend

logger = init_logger(__name__)


class CpuMemoryBackend(OmniKvStoreBackend):
    """KV store backend using CPU pinned memory.

    Config keys:
        backend_type: "cpu"
        max_cpu_memory_gb: float (default 10.0, soft limit for logging)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.storage: dict[str, dict[str, Any]] = {}
        self.max_cpu_memory_gb = config.get("max_cpu_memory_gb", 10.0)
        self.current_bytes = 0
        logger.info(
            "Created CpuMemoryBackend (max_cpu_memory=%.1fGB)",
            self.max_cpu_memory_gb,
        )

    def store_kv(self, request_id: str, kv_data: dict[str, Any]) -> bool:
        try:
            cpu_kv = {
                "kv_caches": [
                    t.detach().cpu().contiguous() if isinstance(t, torch.Tensor) else t
                    for t in kv_data.get("kv_caches", [])
                ],
                "block_ids": kv_data.get("block_ids"),
                "seq_len": kv_data.get("seq_len"),
            }

            num_bytes = sum(t.numel() * t.element_size() for t in cpu_kv["kv_caches"] if isinstance(t, torch.Tensor))

            self.storage[request_id] = cpu_kv
            self.current_bytes += num_bytes
            self._update_store_stats(success=True, num_bytes=num_bytes)
            return True

        except Exception as e:
            logger.error("Failed to store KV for '%s': %s", request_id, e)
            self._update_store_stats(success=False)
            return False

    def load_kv(self, request_id: str, target_device: str = "cuda") -> dict[str, Any] | None:
        try:
            if request_id not in self.storage:
                self._update_load_stats(success=False)
                return None

            cpu_kv = self.storage[request_id]
            gpu_kv = {
                "kv_caches": [
                    t.to(target_device).contiguous() if isinstance(t, torch.Tensor) else t for t in cpu_kv["kv_caches"]
                ],
                "block_ids": cpu_kv.get("block_ids"),
                "seq_len": cpu_kv.get("seq_len"),
            }

            num_bytes = sum(t.numel() * t.element_size() for t in gpu_kv["kv_caches"] if isinstance(t, torch.Tensor))
            self._update_load_stats(success=True, num_bytes=num_bytes)
            return gpu_kv

        except Exception as e:
            logger.error("Failed to load KV for '%s': %s", request_id, e)
            self._update_load_stats(success=False)
            return None

    def evict_kv(self, request_id: str) -> bool:
        if request_id not in self.storage:
            return False

        cpu_kv = self.storage.pop(request_id)
        freed = sum(t.numel() * t.element_size() for t in cpu_kv.get("kv_caches", []) if isinstance(t, torch.Tensor))
        self.current_bytes -= freed
        self._update_evict_stats()
        return True

    def get_stats(self) -> dict[str, Any]:
        stats = self._stats.copy()
        stats.update(
            {
                "backend_type": "cpu",
                "num_stored": len(self.storage),
                "current_bytes": self.current_bytes,
                "current_mb": self.current_bytes / 1024 / 1024,
                "max_cpu_memory_gb": self.max_cpu_memory_gb,
            }
        )
        return stats
