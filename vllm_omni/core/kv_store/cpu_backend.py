"""CPU memory backend for KV cache offloading.

Phase 2: Uses pre-allocated pinned CPU tensors and block-level allocation
for efficient GPU↔CPU transfers via swap_blocks.

Falls back to Phase 1 dict-based storage if GPU KV tensors are not provided.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend

logger = init_logger(__name__)


class CpuMemoryBackend(OmniKvStoreBackend):
    """KV store backend using pre-allocated pinned CPU memory.

    Phase 2 mode (block-level): activated when init_tensors() is called with
    GPU KV cache references. Uses pinned memory + swap_blocks for fast DMA.

    Phase 1 fallback (dict-based): used when init_tensors() is never called.
    Stores per-request tensor dicts, compatible with LmcacheBackend interface.

    Config keys:
        backend_type: "cpu"
        max_cpu_memory_gb: float (default 10.0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.max_cpu_memory_gb = config.get("max_cpu_memory_gb", 10.0)

        # Phase 2: block-level pinned tensor pool (initialized lazily)
        self.cpu_kv_caches: list[torch.Tensor] = []  # pinned, one per layer
        self.num_cpu_blocks: int = 0
        self.free_blocks: list[int] = []  # free block indices
        self.request_blocks: dict[str, list[int]] = {}  # req_id → cpu block IDs
        self.request_metadata: dict[str, dict] = {}  # req_id → {seq_len, ...}
        self._initialized: bool = False

        # Phase 1 fallback: dict-based storage
        self.storage: dict[str, dict[str, Any]] = {}
        self.current_bytes = 0

        logger.info("Created CpuMemoryBackend (max_cpu_memory=%.1fGB)", self.max_cpu_memory_gb)

    # ----------------------------------------------------------------
    # Phase 2: Block-level pinned tensor pool
    # ----------------------------------------------------------------

    def init_tensors(self, gpu_kv_caches: list[torch.Tensor]) -> None:
        """Initialize pinned CPU tensors matching GPU KV cache layout.

        Called lazily from runner on first offload extract. After this,
        the backend operates in block-level mode (Phase 2).

        Args:
            gpu_kv_caches: list of GPU KV tensors, one per layer.
                Each has shape [2, num_gpu_blocks, block_size, heads, dim].
        """
        if self._initialized or not gpu_kv_caches:
            return

        # Calculate bytes per block to determine num_cpu_blocks
        sample = gpu_kv_caches[0]  # [2, num_gpu_blocks, block_size, heads, dim]
        bytes_per_block_per_layer = sample.element_size() * sample[0].stride(0)  # one K or V block
        bytes_per_block = bytes_per_block_per_layer * 2 * len(gpu_kv_caches)  # K+V, all layers

        max_bytes = int(self.max_cpu_memory_gb * 1024**3)
        self.num_cpu_blocks = max(1, max_bytes // bytes_per_block)

        # Allocate pinned CPU tensors matching GPU layout
        for layer_kv in gpu_kv_caches:
            # layer_kv: [2, num_gpu_blocks, block_size, heads, dim]
            cpu_shape = list(layer_kv.shape)
            cpu_shape[1] = self.num_cpu_blocks  # replace num_gpu_blocks with num_cpu_blocks
            cpu_tensor = torch.zeros(cpu_shape, dtype=layer_kv.dtype, device="cpu", pin_memory=True)
            self.cpu_kv_caches.append(cpu_tensor)

        self.free_blocks = list(range(self.num_cpu_blocks))
        self._initialized = True

        logger.info(
            "Initialized pinned CPU tensors: %d layers, %d blocks (%.1f GB), block=%d bytes",
            len(self.cpu_kv_caches),
            self.num_cpu_blocks,
            self.num_cpu_blocks * bytes_per_block / 1024**3,
            bytes_per_block,
        )

    def allocate_blocks(self, request_id: str, num_blocks: int, seq_len: int = 0) -> list[int] | None:
        """Allocate CPU blocks for a request.

        Returns list of CPU block IDs, or None if not enough free blocks.
        """
        if num_blocks > len(self.free_blocks):
            logger.warning(
                "Not enough CPU blocks: need %d, have %d free",
                num_blocks,
                len(self.free_blocks),
            )
            return None

        # Pop from end of free list (fast)
        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.request_blocks[request_id] = allocated
        self.request_metadata[request_id] = {"seq_len": seq_len}
        self._update_store_stats(success=True, num_bytes=0)
        return allocated

    def get_cpu_block_ids(self, request_id: str) -> list[int] | None:
        """Get CPU block IDs for a request."""
        return self.request_blocks.get(request_id)

    def get_cpu_kv_caches(self) -> list[torch.Tensor] | None:
        """Get the pre-allocated pinned CPU KV tensors (one per layer)."""
        return self.cpu_kv_caches if self._initialized else None

    def free_blocks_for_request(self, request_id: str) -> None:
        """Return CPU blocks for a request to the free list."""
        blocks = self.request_blocks.pop(request_id, None)
        if blocks:
            self.free_blocks.extend(blocks)
        self.request_metadata.pop(request_id, None)

    @property
    def supports_block_transfer(self) -> bool:
        """Whether this backend supports Phase 2 block-level swap_blocks transfer."""
        return self._initialized

    # ----------------------------------------------------------------
    # Phase 1 fallback: dict-based storage (backward compatible)
    # ----------------------------------------------------------------

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
        # Phase 2: free block-level allocation
        if request_id in self.request_blocks:
            self.free_blocks_for_request(request_id)
            self._update_evict_stats()
            return True

        # Phase 1 fallback
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
                "num_stored": len(self.storage) + len(self.request_blocks),
                "current_bytes": self.current_bytes,
                "max_cpu_memory_gb": self.max_cpu_memory_gb,
                "block_transfer_enabled": self._initialized,
                "num_cpu_blocks": self.num_cpu_blocks,
                "free_cpu_blocks": len(self.free_blocks),
            }
        )
        return stats
