from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend

# Always-available CPU memory backend (no external deps)
from vllm_omni.core.kv_store.cpu_backend import CpuMemoryBackend
from vllm_omni.core.kv_store.factory import OmniKvStoreFactory

OmniKvStoreFactory.register("cpu", CpuMemoryBackend)

# Register LMCache backend (lazy: only fails at create-time if lmcache missing)
try:
    from vllm_omni.core.kv_store.lmcache_backend import LMCacheBackend

    OmniKvStoreFactory.register("lmcache", LMCacheBackend)
except ImportError:
    pass  # lmcache not installed, skip registration

__all__ = [
    "OmniKvStoreBackend",
    "OmniKvStoreFactory",
]
