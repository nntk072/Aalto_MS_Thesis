"""Causal swing high/low structure features for SL/TP pricing.

Swing detection uses only past bars (causal), suitable for real-time trading.
Extends the swing detection logic from smt.py with explicit price levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _swing_highs(s: pd.Series, period: int) -> pd.Series:
    """1 where s[t] is a local max over ±period bars (causal: look back only)."""
    roll_max = s.rolling(2 * period + 1, center=False).max()
    # Shift so we don't look forward
    lag_max = roll_max.shift(period)
    return (s.shift(period) == lag_max).astype(int)


def _swing_lows(s: pd.Series, period: int) -> pd.Series:
    """1 where s[t] is a local min over ±period bars (causal: look back only)."""
    roll_min = s.rolling(2 * period + 1, center=False).min()
    lag_min = roll_min.shift(period)
    return (s.shift(period) == lag_min).astype(int)


def structure_levels(
    bars: pd.DataFrame,
    swing_period: int = 5,
) -> pd.DataFrame:
    """Compute causal swing price levels for structure-based SL/TP.

    Parameters
    ----------
    bars:
        OHLC DataFrame with 'high', 'low' columns and DatetimeIndex.
    swing_period:
        Bars on each side to detect a swing high/low.

    Returns
    -------
    DataFrame with columns:
      - ``last_swing_high``: price of most recent confirmed swing high (NaN if none yet)
      - ``last_swing_low``: price of most recent confirmed swing low (NaN if none yet)
      - ``last_swing_high_time``: timestamp of last swing high
      - ``last_swing_low_time``: timestamp of last swing low
    """
    sh = _swing_highs(bars["high"], swing_period)
    sl = _swing_lows(bars["low"], swing_period)

    # Extract prices at swing points
    swing_high_prices = bars["high"][sh == 1]
    swing_low_prices = bars["low"][sl == 1]

    # Forward-fill to get "last" level at each bar
    last_sh = pd.Series(np.nan, index=bars.index)
    last_sh_time = pd.Series(pd.NaT, index=bars.index, dtype="datetime64[ns]")

    last_sl = pd.Series(np.nan, index=bars.index)
    last_sl_time = pd.Series(pd.NaT, index=bars.index, dtype="datetime64[ns]")

    if len(swing_high_prices) > 0:
        for ts, price in swing_high_prices.items():
            # All bars at or after this swing point get this level until next swing
            mask = bars.index >= ts
            last_sh[mask] = price
            last_sh_time[mask] = ts

    if len(swing_low_prices) > 0:
        for ts, price in swing_low_prices.items():
            mask = bars.index >= ts
            last_sl[mask] = price
            last_sl_time[mask] = ts

    return pd.DataFrame(
        {
            "last_swing_high": last_sh,
            "last_swing_low": last_sl,
            "last_swing_high_time": last_sh_time,
            "last_swing_low_time": last_sl_time,
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

    result = result.ffill()

    return result
