"""Discover neutral hardware capabilities available to the current process."""

from __future__ import annotations

import platform

import torch

from compute.contracts import DeviceCapabilities


def discover_device_capabilities(device_hint: str | None = None) -> DeviceCapabilities:
    """Describe the selected visible accelerator without exposing model policy."""
    if not device_hint or device_hint.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(device_hint)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid device value {device_hint!r}") from exc

    has_accelerator = device.type == "cuda" and torch.cuda.is_available()
    accelerator_count = torch.cuda.device_count() if has_accelerator else 0
    index = device.index if device.index is not None else 0
    properties = torch.cuda.get_device_properties(index) if has_accelerator else None
    memory = int(properties.total_memory) if properties is not None else None
    supports_bfloat16 = bool(has_accelerator and torch.cuda.is_bf16_supported())
    capability = (
        (int(properties.major), int(properties.minor))
        if properties is not None and hasattr(properties, "major") and hasattr(properties, "minor")
        else None
    )
    backend = "rocm" if has_accelerator and torch.version.hip is not None else "cuda"
    return DeviceCapabilities(
        architecture=platform.machine(),
        accelerator_kind=backend if has_accelerator else None,
        accelerator_count=accelerator_count,
        accelerator_memory_bytes=memory,
        supports_bfloat16=supports_bfloat16,
        accelerator_backend=backend if has_accelerator else None,
        device_name=str(properties.name) if properties is not None else None,
        compute_capability=capability,
    )


__all__ = ["discover_device_capabilities"]
