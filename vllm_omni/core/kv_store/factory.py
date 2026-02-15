from typing import Any

from vllm.logger import init_logger

from vllm_omni.core.kv_store.backend_interface import OmniKvStoreBackend

logger = init_logger(__name__)


class OmniKvStoreFactory:
    """Factory class for creating kv store backend instances."""

    _registry: dict[str, type[OmniKvStoreBackend]] = {}

    @classmethod
    def register(cls, backend_name: str, backend_cls: type[OmniKvStoreBackend]) -> None:
        """Register a kv store backend class with a given name.

        Args:
            backend_name (str): The name of the backend.
            backend_cls (Type[OmniKvStoreBackend]): The backend class to register
        """

        if backend_name in cls._registry:
            logger.warning(f"Backend '{backend_name}' is already registered. Overwriting.")
        cls._registry[backend_name] = backend_cls
        logger.info(f"Registered kv store backend '{backend_name}'.")

    @classmethod
    def create(cls, backend_name: str, config: dict[str, Any]) -> OmniKvStoreBackend:
        """Create an instance of the specified kv store backend.

        Args:
            backend_name (str): The name of the backend to create.
            config (dict[str, Any]): Configuration dictionary for the backend.

        Returns:
            OmniKvStoreBackend: An instance of the requested kv store backend.

        Raises:
            ValueError: If the specified backend is not registered.
        """

        backend_type = config.get("backend_type", "none")

        if backend_type == "none":
            logger.info("KV store offloading disabled")
            return None

        if backend_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown backend: '{backend_type}'. Available: {available}")

        backend_cls = cls._registry[backend_type]
        try:
            return backend_cls(config)
        except Exception as e:
            logger.error(f"Failed to create backend '{backend_type}': {e}")
            raise

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered kv store backends.

        Returns:
            list[str]: A list of registered backend names.
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, backend_name: str) -> bool:
        """Check if a backend is registered.

        Args:
            backend_name (str): The name of the backend to check.
        Returns:
            bool: True if the backend is registered, False otherwise.
        """

        return backend_name in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered backends."""
        cls._registry.clear()
        logger.info("Cleared all registered kv store backends.")
