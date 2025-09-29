"""
Device utilities for managing PyTorch device selection, validation, and fallbacks.

This module provides centralized device management for neural network computations,
with support for configurable device preferences, automatic fallbacks, and tensor
compatibility validation.
"""

import logging
import warnings
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Centralized device manager for PyTorch neural network computations.

    Handles device selection, validation, fallbacks, and memory management.
    """

    def __init__(
        self,
        preference: str = "auto",
        fallback: bool = True,
        memory_fraction: Optional[float] = None,
        validate_compatibility: bool = True,
    ):
        """
        Initialize the device manager.

        Args:
            preference: Device preference ("auto", "cpu", "cuda", "cuda:X")
            fallback: Whether to fallback to CPU if preferred device unavailable
            memory_fraction: GPU memory fraction to reserve (0.0-1.0)
            validate_compatibility: Whether to validate tensor compatibility
        """
        self.preference = preference
        self.fallback = fallback
        self.memory_fraction = memory_fraction
        self.validate_compatibility = validate_compatibility
        self._device: Optional[torch.device] = None
        self._initialized = False

    def get_device(self) -> torch.device:
        """
        Get the appropriate device based on configuration and availability.

        Returns:
            torch.device: The selected device
        """
        if not self._initialized:
            self._device = self._resolve_device()
            self._initialized = True
            self._configure_device()

        assert self._device is not None, "Device should be initialized"
        return self._device

    def _resolve_device(self) -> torch.device:
        """
        Resolve the device based on preference and availability.

        Returns:
            torch.device: The resolved device
        """
        if self.preference == "auto":
            return self._auto_select_device()
        elif self.preference == "cpu":
            return torch.device("cpu")
        elif self.preference.startswith("cuda"):
            return self._resolve_cuda_device()
        else:
            logger.warning(
                f"Unknown device preference '{self.preference}', falling back to auto"
            )
            return self._auto_select_device()

    def _auto_select_device(self) -> torch.device:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            # Check if CUDA is actually working
            try:
                # Test CUDA device
                test_tensor = torch.tensor([1.0], device="cuda:0")
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("CUDA device detected and working")
                return torch.device("cuda:0")
            except Exception as e:
                logger.warning(f"CUDA device detected but not working: {e}")
                if self.fallback:
                    logger.info("Falling back to CPU")
                    return torch.device("cpu")
                else:
                    raise RuntimeError(
                        f"CUDA device not working and fallback disabled: {e}"
                    ) from e
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

    def _resolve_cuda_device(self) -> torch.device:
        """Resolve CUDA device specification."""
        if not torch.cuda.is_available():
            if self.fallback:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            else:
                raise RuntimeError(
                    "CUDA requested but not available, and fallback disabled"
                )

        if self.preference == "cuda":
            device = torch.device("cuda:0")
        elif self.preference.startswith("cuda:"):
            try:
                device_idx = int(self.preference.split(":")[1])
                if device_idx >= torch.cuda.device_count():
                    if self.fallback:
                        logger.warning(
                            f"CUDA device {device_idx} not available, falling back to cuda:0"
                        )
                        device = torch.device("cuda:0")
                    else:
                        raise RuntimeError(
                            f"CUDA device {device_idx} not available and fallback disabled"
                        )
                else:
                    device = torch.device(f"cuda:{device_idx}")
            except (ValueError, IndexError) as e:
                if self.fallback:
                    logger.warning(
                        f"Invalid CUDA device specification '{self.preference}', falling back to cuda:0"
                    )
                    device = torch.device("cuda:0")
                else:
                    raise RuntimeError(
                        f"Invalid CUDA device specification and fallback disabled: {e}"
                    ) from e
        else:
            device = torch.device("cuda:0")

        # Test the selected CUDA device
        try:
            test_tensor = torch.tensor([1.0], device=device)
            del test_tensor
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            if self.fallback:
                logger.warning(
                    f"CUDA device {device} not working, falling back to CPU: {e}"
                )
                return torch.device("cpu")
            else:
                raise RuntimeError(
                    f"CUDA device {device} not working and fallback disabled: {e}"
                ) from e

    def _configure_device(self) -> None:
        """Configure device-specific settings."""
        if self._device and self._device.type == "cuda":
            if self.memory_fraction is not None:
                if 0.0 <= self.memory_fraction <= 1.0:
                    try:
                        torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                        logger.info(
                            f"Set CUDA memory fraction to {self.memory_fraction}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to set CUDA memory fraction: {e}")
                else:
                    logger.warning(
                        f"Invalid memory fraction {self.memory_fraction}, ignoring"
                    )

            # Log device information
            device_props = torch.cuda.get_device_properties(self._device)
            logger.info(f"Using CUDA device: {device_props.name}")
            logger.info(f"CUDA memory: {device_props.total_memory // (1024**3)} GB")

    def validate_tensor_compatibility(
        self, tensor: torch.Tensor, target_device: torch.device
    ) -> bool:
        """
        Validate tensor compatibility with target device.

        Args:
            tensor: Tensor to validate
            target_device: Target device for compatibility check

        Returns:
            bool: True if compatible, False otherwise
        """
        if not self.validate_compatibility:
            return True

        try:
            # Check if tensor can be moved to target device
            if tensor.device == target_device:
                return True

            # Try to move a small test tensor
            test_tensor = torch.tensor([1.0], dtype=tensor.dtype, device=tensor.device)
            moved_tensor = test_tensor.to(target_device)
            del test_tensor, moved_tensor
            return True
        except Exception as e:
            logger.warning(f"Tensor compatibility validation failed: {e}")
            return False

    def safe_tensor_to_device(
        self, tensor: torch.Tensor, target_device: torch.device
    ) -> torch.Tensor:
        """
        Safely move tensor to target device with validation.

        Args:
            tensor: Tensor to move
            target_device: Target device

        Returns:
            torch.Tensor: Tensor on target device

        Raises:
            RuntimeError: If tensor cannot be moved and validation is enabled
        """
        if not self.validate_tensor_compatibility(tensor, target_device):
            raise RuntimeError(f"Tensor cannot be moved to device {target_device}")

        try:
            return tensor.to(target_device)
        except Exception as e:
            raise RuntimeError(f"Failed to move tensor to device {target_device}: {e}") from e

    def get_optimal_device_for_model(
        self, model_size_mb: Optional[float] = None
    ) -> torch.device:
        """
        Get optimal device for model based on size and availability.

        Args:
            model_size_mb: Estimated model size in MB

        Returns:
            torch.device: Optimal device
        """
        device = self.get_device()

        if device.type == "cuda" and model_size_mb is not None:
            # Check if model fits in GPU memory
            try:
                device_props = torch.cuda.get_device_properties(device)
                available_memory_mb = device_props.total_memory / (1024**2)

                if self.memory_fraction:
                    available_memory_mb *= self.memory_fraction

                # Reserve some memory for operations
                effective_memory_mb = available_memory_mb * 0.8

                if model_size_mb > effective_memory_mb:
                    logger.warning(
                        f"Model size ({model_size_mb} MB) may exceed available GPU memory "
                        f"({effective_memory_mb:.1f} MB), consider using CPU or smaller model"
                    )
            except Exception as e:
                logger.warning(f"Failed to check GPU memory: {e}")

        return device

    def reset(self) -> None:
        """Reset device manager state."""
        self._device = None
        self._initialized = False


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    preference: str = "auto",
    fallback: bool = True,
    memory_fraction: Optional[float] = None,
    validate_compatibility: bool = True,
) -> DeviceManager:
    """
    Get or create a global device manager instance.

    Args:
        preference: Device preference
        fallback: Whether to fallback to CPU
        memory_fraction: GPU memory fraction
        validate_compatibility: Whether to validate compatibility

    Returns:
        DeviceManager: Global device manager instance
    """
    global _global_device_manager

    if _global_device_manager is None:
        _global_device_manager = DeviceManager(
            preference=preference,
            fallback=fallback,
            memory_fraction=memory_fraction,
            validate_compatibility=validate_compatibility,
        )
    else:
        # Update configuration if different
        if (
            _global_device_manager.preference != preference
            or _global_device_manager.fallback != fallback
            or _global_device_manager.memory_fraction != memory_fraction
            or _global_device_manager.validate_compatibility != validate_compatibility
        ):
            _global_device_manager.reset()
            _global_device_manager.preference = preference
            _global_device_manager.fallback = fallback
            _global_device_manager.memory_fraction = memory_fraction
            _global_device_manager.validate_compatibility = validate_compatibility

    return _global_device_manager


def get_device(
    preference: str = "auto",
    fallback: bool = True,
    memory_fraction: Optional[float] = None,
    validate_compatibility: bool = True,
) -> torch.device:
    """
    Convenience function to get device directly.

    Args:
        preference: Device preference
        fallback: Whether to fallback to CPU
        memory_fraction: GPU memory fraction
        validate_compatibility: Whether to validate compatibility

    Returns:
        torch.device: The selected device
    """
    manager = get_device_manager(
        preference=preference,
        fallback=fallback,
        memory_fraction=memory_fraction,
        validate_compatibility=validate_compatibility,
    )
    return manager.get_device()


def safe_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convenience function for safe tensor device movement.

    Args:
        tensor: Tensor to move
        device: Target device

    Returns:
        torch.Tensor: Tensor on target device
    """
    manager = get_device_manager()
    return manager.safe_tensor_to_device(tensor, device)


def create_device_from_config(config) -> torch.device:
    """
    Create device from configuration object.

    Args:
        config: Configuration object with device settings

    Returns:
        torch.device: Configured device
    """
    # Support nested SimulationConfig.device as well as flat configs
    device_cfg = getattr(config, "device", None)
    preference = (
        getattr(device_cfg, "device_preference", None)
        if device_cfg is not None
        else None
    ) or getattr(config, "device_preference", "auto")

    fallback = (
        getattr(device_cfg, "device_fallback", None)
        if device_cfg is not None
        else None
    )
    if fallback is None:
        fallback = getattr(config, "device_fallback", True)

    memory_fraction = (
        getattr(device_cfg, "device_memory_fraction", None)
        if device_cfg is not None
        else None
    )
    if memory_fraction is None:
        memory_fraction = getattr(config, "device_memory_fraction", None)

    validate_compatibility = (
        getattr(device_cfg, "device_validate_compatibility", None)
        if device_cfg is not None
        else None
    )
    if validate_compatibility is None:
        validate_compatibility = getattr(
            config, "device_validate_compatibility", True
        )

    return get_device(
        preference=preference,
        fallback=fallback,
        memory_fraction=memory_fraction,
        validate_compatibility=validate_compatibility,
    )
