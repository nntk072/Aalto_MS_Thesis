"""Hardware-aware mixed precision and accelerator setup policy."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from compute.environment.capabilities import discover_device_capabilities


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device value {device!r}") from exc


def get_autocast_dtype(device: str | torch.device | None = None) -> torch.dtype | None:
    """Choose fp16 for pre-Ampere GPUs and bf16 when a newer GPU supports it."""
    resolved = _resolve_device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    capabilities = discover_device_capabilities(str(resolved))
    if capabilities.compute_capability is not None and capabilities.compute_capability[0] < 8:
        return torch.float16
    if capabilities.supports_bfloat16:
        return torch.bfloat16
    return torch.float16


def get_autocast_context(
    device: str | torch.device | None = None,
    *,
    dtype: torch.dtype | None = None,
    enabled: bool | None = None,
) -> AbstractContextManager[Any]:
    """Return the selected accelerator autocast context, or a CPU no-op."""
    resolved = _resolve_device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available() or enabled is False:
        return nullcontext()
    chosen_dtype = dtype or get_autocast_dtype(resolved)
    if chosen_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=chosen_dtype, enabled=True)


def configure_accelerator_for_training(device_hint: str | torch.device | None = None) -> None:
    """Configure allocator and backend kernels before PPO's first GPU allocation."""
    if isinstance(device_hint, torch.device) and device_hint.type == "cpu":
        return
    if isinstance(device_hint, str) and device_hint.lower() == "cpu":
        return
    is_rocm = torch.version.hip is not None
    if not is_rocm:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    resolved = _resolve_device(device_hint)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return

    capabilities = discover_device_capabilities(str(resolved))
    if capabilities.accelerator_backend == "cuda":
        if capabilities.compute_capability is not None and capabilities.compute_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def configure_cuda_for_training(device_hint: str | torch.device | None = None) -> None:
    """Backward-compatible alias for accelerator-wide training setup."""
    configure_accelerator_for_training(device_hint)


__all__ = [
    "configure_accelerator_for_training",
    "configure_cuda_for_training",
    "get_autocast_context",
    "get_autocast_dtype",
]
