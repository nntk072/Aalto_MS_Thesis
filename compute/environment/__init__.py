"""Runtime device, memory, and compute-environment resolution."""

from .device import configure_cuda, get_device, scale_training_cfg

__all__ = ["configure_cuda", "get_device", "scale_training_cfg"]
