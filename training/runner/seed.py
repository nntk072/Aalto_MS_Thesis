"""Seed and deterministic run helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _setup_rngs(seed: int) -> None:
    """Seed numpy, random, and torch for reproducible training."""
    np.random.seed(seed)
    import random

    random.seed(seed)
    # SB3 seeds its own generator but does *not* seed torch.manual_seed — the
    # caller's responsibility. Network init/dropout draw from torch's global
    # RNG, so without this line two --seed 42 runs are not reproducible.
    torch.manual_seed(seed)


def _direction_summary(trades: Any) -> dict[str, int]:
    """Count open directions from an evaluate_model trade log."""
    if trades is None or getattr(trades, "empty", True):
        return {"long": 0, "short": 0, "opens": 0}
    opens = trades[trades["type"] == "open"] if "type" in trades.columns else trades
    if opens.empty or "direction" not in opens.columns:
        return {"long": 0, "short": 0, "opens": 0}
    dirs = opens["direction"].astype(int)
    return {
        "long": int((dirs == 1).sum()),
        "short": int((dirs == -1).sum()),
        "opens": int(len(opens)),
    }
