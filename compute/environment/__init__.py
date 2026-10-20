"""Runtime device, memory, and compute-environment resolution."""

from .capabilities import discover_device_capabilities
from .device import get_device, scale_training_cfg
from .precision import (
    configure_accelerator_for_training,
    configure_cuda_for_training,
    get_autocast_context,
    get_autocast_dtype,
)

__all__ = [
    "configure_accelerator_for_training",
    "configure_cuda_for_training",
    "discover_device_capabilities",
    "get_autocast_context",
    "get_autocast_dtype",
    "get_device",
    "scale_training_cfg",
]
