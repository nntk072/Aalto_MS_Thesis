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
