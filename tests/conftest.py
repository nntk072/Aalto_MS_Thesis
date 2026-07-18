"""Pytest fixtures for quant_rl tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def m1_bars() -> pd.DataFrame:
    """Small synthetic M1 bar DataFrame (500 bars, NY session, UTC+3)."""
    n = 500
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(42)
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
        {
            "open":    close - rng.uniform(0, 1, n),
            "high":    close + rng.uniform(0, 2, n),
            "low":     close - rng.uniform(0, 2, n),
            "close":   close,
            "tickvol": rng.integers(10, 200, n),
            "vol":     np.zeros(n, dtype=int),
            "spread":  np.full(n, 0.6),
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df
