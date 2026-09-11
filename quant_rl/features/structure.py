"""Causal swing high/low structure features for SL/TP pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.session import get_session
from .swings import (
    TIMEFRAME_CONFIG,
    classify_structure,
    detect_pivots,
    detect_swings,
    swing_features,
)

__all__ = [
    "TIMEFRAME_CONFIG",
    "classify_structure",
    "detect_pivots",
    "detect_session_levels",
    "detect_swings",
    "get_session",
    "structure_levels",
    "swing_features",
]


def structure_levels(
    bars: pd.DataFrame,
    swing_period: int = 5,
    atr_mult: float = 0.0,
) -> pd.DataFrame:
    """Compute causal swing price levels for structure-based SL/TP.

    Thin wrapper over :func:`detect_pivots` / :func:`detect_swings`. Default
    ``atr_mult=0`` accepts every confirmed fractal so SL/TP and liquidity
    keep fractal timing; pass a positive multiplier for ATR zigzag filtering.

    Args:
        bars: OHLC DataFrame with high/low/close and a DatetimeIndex.
        swing_period: Bars on each side of the fractal (left = right).
        atr_mult: Zigzag reversal in ATR units; 0 accepts on confirmation.
    """
    pivots = detect_pivots(bars, left=swing_period, right=swing_period)
    swings = detect_swings(bars, pivots, atr_mult=atr_mult)
    times = np.asarray(bars.index, dtype="datetime64[ns]")
    loc_h = swings["swing_high_location"].to_numpy(dtype=float)
    loc_l = swings["swing_low_location"].to_numpy(dtype=float)
    sh_times = np.full(len(bars), np.datetime64("NaT"), dtype="datetime64[ns]")
    sl_times = np.full(len(bars), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i, loc in enumerate(loc_h):
        if np.isfinite(loc):
            sh_times[i] = times[int(loc)]
    for i, loc in enumerate(loc_l):
        if np.isfinite(loc):
            sl_times[i] = times[int(loc)]
    return pd.DataFrame(
        {
            "last_swing_high": swings["swing_high_extreme"].to_numpy(dtype=float),
            "last_swing_low": swings["swing_low_extreme"].to_numpy(dtype=float),
            "last_swing_high_time": pd.Series(sh_times, index=bars.index, dtype=bars.index.dtype),
            "last_swing_low_time": pd.Series(sl_times, index=bars.index, dtype=bars.index.dtype),
        },
        index=bars.index,
    )


def detect_session_levels(
    df: pd.DataFrame,
    asian_start: str = "01:05",
    asian_end: str = "09:00",
    london_end: str = "16:30",
    swing_period: int = 50,
    min_bars_per_session: int = 10,
) -> pd.DataFrame:
    """Detect Asian High/Low and London High/Low from pre-NY data.

    Uses rolling swing high/low detection within each session window.
    Levels are forward-filled to the NY session for use in real-time trading.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with timezone-aware DatetimeIndex.
        Required columns: ['open', 'high', 'low', 'close', 'volume']
    asian_start : str
        Asian session start time (HH:MM format, UTC+3)
    asian_end : str
        Asian session end time (HH:MM format, UTC+3)
    london_end : str
        London session end time (HH:MM format, UTC+3)
    swing_period : int
        Rolling window for swing high/low detection
    min_bars_per_session : int
        Minimum bars required in a session to compute levels

    Returns
    -------
    pd.DataFrame
        Same index as input, with added columns:
        - asian_high: Rolling swing high from Asian session
        - asian_low: Rolling swing low from Asian session
        - london_high: Rolling swing high from London session
        - london_low: Rolling swing low from London session
        - prev_day_close: Previous day's close price
        - prev_day_high: Previous calendar day's high price
        - prev_day_low: Previous calendar day's low price
    """
    df = df.copy()
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex), "Index must be DatetimeIndex"

    asian_start_time = pd.Timestamp(f"2000-01-01 {asian_start}").time()
    asian_end_time = pd.Timestamp(f"2000-01-01 {asian_end}").time()
    london_end_time = pd.Timestamp(f"2000-01-01 {london_end}").time()

    asian_mask = (idx.time >= asian_start_time) & (idx.time < asian_end_time)
    london_mask = (idx.time >= asian_end_time) & (idx.time < london_end_time)

    result = pd.DataFrame(index=idx)

    asian_data = df[asian_mask].copy()
    if len(asian_data) >= min_bars_per_session:
        asian_high = asian_data["high"].rolling(swing_period, min_periods=1).max()
        asian_low = asian_data["low"].rolling(swing_period, min_periods=1).min()
        result["asian_high"] = asian_high.reindex(idx).ffill()
        result["asian_low"] = asian_low.reindex(idx).ffill()
    else:
        result["asian_high"] = np.nan
        result["asian_low"] = np.nan

    london_data = df[london_mask].copy()
    if len(london_data) >= min_bars_per_session:
        london_high = london_data["high"].rolling(swing_period, min_periods=1).max()
        london_low = london_data["low"].rolling(swing_period, min_periods=1).min()
        result["london_high"] = london_high.reindex(idx).ffill()
        result["london_low"] = london_low.reindex(idx).ffill()
    else:
        result["london_high"] = np.nan
        result["london_low"] = np.nan

    prev_close = df["close"].shift(1)
    result["prev_day_close"] = prev_close

    # Previous day's high/low (causal: only uses completed prior calendar days).
    idx_dt = pd.DatetimeIndex(df.index)
    daily_high = df["high"].groupby(idx_dt.date).max()
    daily_low = df["low"].groupby(idx_dt.date).min()
    prev_day_high = daily_high.shift(1).reindex(idx).ffill()
    prev_day_low = daily_low.shift(1).reindex(idx).ffill()
    result["prev_day_high"] = prev_day_high
    result["prev_day_low"] = prev_day_low

    result = result.ffill()

    return result
