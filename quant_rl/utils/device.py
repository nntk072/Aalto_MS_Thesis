"""Shared PyTorch device resolution.

Prefer GPU when available, fall back to CPU. Also accepts explicit device
strings from config so users can pin a device without changing code.
"""

from __future__ import annotations

import torch


def get_device(config_device: str | None = None) -> torch.device:
    """Resolve the torch device to use.

    Parameters
    ----------
    config_device:
        Optional device hint from config. Accepted values:

        * ``None`` or ``"auto"`` — pick CUDA if available, else CPU.
        * ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ... — use as-is.
        * Any other string raises ``ValueError``.

    Returns
    -------
    torch.device
    """
    if not config_device or config_device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        return torch.device(config_device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device value {config_device!r}") from exc
