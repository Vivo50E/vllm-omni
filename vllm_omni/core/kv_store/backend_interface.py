from abc import ABC, abstractmethod
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class OmniKvStoreBackend(ABC):
    """Abstract base class for kv cache store backends."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._stats = {
            "total_stores": 0,
            "total_loads": 0,
            "total_evictions": 0,
            "store_failures": 0,
            "load_failures": 0,
            "total_bytes_stored": 0,
            "total_bytes_loaded": 0,
        }

    @abstractmethod
    def store_kv(self, request_id: str, kv_data: dict[str, Any]) -> bool:
        """Store kv cache for a given request ID."""
        pass

    @abstractmethod
    def load_kv(self, request_id: str, target_device: str = "cuda") -> dict[str, Any] | None:
        """Load kv cache for a given request ID."""
        pass

    @abstractmethod
    def evict_kv(self, request_id: str) -> bool:
        """Evict kv cache for a given request ID."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the kv store backend."""
        pass

    def _update_store_stats(self, success: bool, num_bytes: int = 0) -> None:
        self._stats["total_stores"] += 1
        if success:
            self._stats["total_bytes_stored"] += num_bytes
        else:
            self._stats["store_failures"] += 1

    def _update_load_stats(self, success: bool, num_bytes: int = 0) -> None:
        self._stats["total_loads"] += 1
        if success:
            self._stats["total_bytes_loaded"] += num_bytes
        else:
            self._stats["load_failures"] += 1

    def _update_evict_stats(self) -> None:
        self._stats["total_evictions"] += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
