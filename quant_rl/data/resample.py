"""Resample M1 bars to higher timeframes.

All mislabeled multi-TF CSV files in the raw data folder are IGNORED.
Every timeframe is built from the true M1 source via pandas resample().
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

TF = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

_TF_TO_RULE: dict[TF, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tickvol": "sum",
    "vol": "sum",
    "spread": "mean",
}


def resample(m1: pd.DataFrame, tf: TF) -> pd.DataFrame:
    """Return *tf* bars built from *m1* (1-minute) bars.

    Parameters
    ----------
    m1:
        DataFrame produced by :func:`quant_rl.data.loader.load_bars`
        with a DatetimeIndex.
    tf:
        Target timeframe string.
    """
    if tf == "M1":
        return m1.copy()
    rule = _TF_TO_RULE[tf]
    # label='left', closed='left' → bar timestamp is the bar open time
    agg_spec: Any = _OHLCV_AGG
    resampled = m1.resample(rule, label="left", closed="left").agg(agg_spec)
    resampled = resampled.dropna(subset=["open"])
    return resampled


def build_all_timeframes(m1: pd.DataFrame) -> dict[TF, pd.DataFrame]:
    """Build all standard timeframes from M1 data."""
    return {tf: resample(m1, tf) for tf in _TF_TO_RULE}
