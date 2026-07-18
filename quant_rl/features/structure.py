"""Structure-based SL/TP: swing highs/lows for per-trade price levels.

Causal swing detection: identifies price swing extrema using only past bars,
suitable for determining SL/TP levels at trade entry.

Usage in features pipeline:
  structure_df = compute_structure_features(bars, swing_period=5, buffer_pts=1.0)
  # Output columns: last_swing_low_price, last_swing_high_price
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_swing_highs(high: pd.Series, period: int = 5) -> pd.Series:
    """Identify bars where price is a local high.
    
    Returns 1.0 where high[t-period] is a local max in [t-2*period : t], else 0.0.
    (Causal: only uses past bars, no look-ahead.)
    
    Parameters
    ----------
    high : pd.Series
        High prices indexed by time.
    period : int
        Number of bars on each side for swing detection.
    
    Returns
    -------
    pd.Series
        Float series, 1.0 where swing high detected, 0.0 otherwise.
    """
    roll_max = high.rolling(window=2 * period + 1, center=False).max()
    # Shift so we look at the max value from the past
    lag_max = roll_max.shift(period)
    is_swing = (high.shift(period) == lag_max).astype(float)
    return is_swing


def _causal_swing_lows(low: pd.Series, period: int = 5) -> pd.Series:
    """Identify bars where price is a local low.
    
    Returns 1.0 where low[t-period] is a local min in [t-2*period : t], else 0.0.
    (Causal: only uses past bars, no look-ahead.)
    
    Parameters
    ----------
    low : pd.Series
        Low prices indexed by time.
    period : int
        Number of bars on each side for swing detection.
    
    Returns
    -------
    pd.Series
        Float series, 1.0 where swing low detected, 0.0 otherwise.
    """
    roll_min = low.rolling(window=2 * period + 1, center=False).min()
    lag_min = roll_min.shift(period)
    is_swing = (low.shift(period) == lag_min).astype(float)
    return is_swing


def compute_structure_features(
    bars: pd.DataFrame,
    swing_period: int = 5,
    buffer_pts: float = 1.0,
) -> pd.DataFrame:
    """Compute structure-based SL/TP price levels.
    
    For each bar, compute:
    - last_swing_low_price: most recent confirmed swing low (for long SL)
    - last_swing_high_price: most recent confirmed swing high (for short SL)
    
    These are causal: computed only from past bars, ready for entry logic.
    
    Parameters
    ----------
    bars : pd.DataFrame
        M1 bars with 'high' and 'low' columns, DatetimeIndex.
    swing_period : int
        Bars on each side for swing detection (default 5).
    buffer_pts : float
        Price offset beyond swing extreme (default 1.0 pt).
    
    Returns
    -------
    pd.DataFrame
        Columns: last_swing_low_price, last_swing_high_price
        Same index as bars.
    """
    sh = _causal_swing_highs(bars["high"], swing_period)
    sl = _causal_swing_lows(bars["low"], swing_period)
    
    # Find the most recent swing high and low
    last_swing_high = None
    last_swing_low = None
    last_swing_high_prices = []
    last_swing_low_prices = []
    
    for i in range(len(bars)):
        if sh.iloc[i] == 1.0:
            last_swing_high = bars["high"].iloc[i - swing_period] if i >= swing_period else bars["high"].iloc[i]
        if sl.iloc[i] == 1.0:
            last_swing_low = bars["low"].iloc[i - swing_period] if i >= swing_period else bars["low"].iloc[i]
        
        last_swing_high_prices.append(last_swing_high if last_swing_high is not None else np.nan)
        last_swing_low_prices.append(last_swing_low if last_swing_low is not None else np.nan)
    
    return pd.DataFrame(
        {
            "last_swing_low_price": last_swing_low_prices,
            "last_swing_high_price": last_swing_high_prices,
        },
        index=bars.index,
    )
