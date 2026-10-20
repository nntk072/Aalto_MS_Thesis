"""Market-clock semantics shared by environment services."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_MINUTES_PER_STEP = 1.0
BAR_TIME_TZ = "Etc/GMT-3"


def _minutes_per_step(index: pd.DatetimeIndex) -> float:
    """Return median positive bar spacing, with a one-minute fallback."""
    if index is None or len(index) < 2:
        return _DEFAULT_MINUTES_PER_STEP
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    deltas = np.diff(nanos)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return _DEFAULT_MINUTES_PER_STEP
    minutes = float(np.median(positive)) / 60_000_000_000.0
    return minutes if minutes > 0.0 else _DEFAULT_MINUTES_PER_STEP


__all__ = ["BAR_TIME_TZ", "_DEFAULT_MINUTES_PER_STEP", "_minutes_per_step"]
