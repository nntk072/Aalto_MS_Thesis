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

    # The swing flag fires ``swing_period`` bars AFTER the extremum bar, so the
    # swing price is the raw series shifted back to the swing bar (using the
    # flag bar's own price would record the wrong level).
    conf_h = (sh == 1).to_numpy()
    conf_l = (sl == 1).to_numpy()
    price_h = bars["high"].shift(swing_period).to_numpy()
    price_l = bars["low"].shift(swing_period).to_numpy()
    times = np.asarray(bars.index, dtype="datetime64[ns]")

    n = len(bars)
    sh_vals = np.full(n, np.nan)
    sl_vals = np.full(n, np.nan)
    sh_times = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    sl_times = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    # Running "current confirmed level": updates at each confirmation bar and
    # persists forward (equivalent to forward-fill from the flag position).
    cur_hv, cur_ht = np.nan, np.datetime64("NaT")
    cur_lv, cur_lt = np.nan, np.datetime64("NaT")
    for i in range(n):
        if conf_h[i]:
            cur_hv, cur_ht = price_h[i], times[i - swing_period]
        if conf_l[i]:
            cur_lv, cur_lt = price_l[i], times[i - swing_period]
        sh_vals[i] = cur_hv
        sl_vals[i] = cur_lv
        sh_times[i] = cur_ht
        sl_times[i] = cur_lt

    return pd.DataFrame(
        {
            "last_swing_high": sh_vals,
            "last_swing_low": sl_vals,
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


def get_session(
    timestamp: pd.Timestamp | str,
    tz: str = "Etc/GMT-3",
) -> str:
    """Determine trading session for a given timestamp.

    Session times (UTC+3):
    - Asia:   01:05 – 09:00
    - London: 09:00 – 16:30
    - NY:     16:30 – 23:50 (or next day 00:00)

    Parameters
    ----------
    timestamp:
        Timestamp to classify (timezone-aware or naive).
    tz:
        Timezone for session times (default: broker timezone UTC+3).

    Returns
    -------
    Literal[\"asia\", \"london\", \"ny\"]
        Session name for the timestamp.
    """
    if isinstance(timestamp, str):
        ts = pd.Timestamp(timestamp)
    else:
        ts = timestamp

    # Localize to session timezone if naive
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)

    t = ts.time()

    # Session boundaries (UTC+3)
    asia_start = pd.Timestamp("2000-01-01 01:05").time()
    asia_end = pd.Timestamp("2000-01-01 09:00").time()
    london_end = pd.Timestamp("2000-01-01 16:30").time()
    ny_end = pd.Timestamp("2000-01-01 23:50").time()

    if asia_start <= t < asia_end:
        return "asia"
    elif asia_end <= t < london_end:
        return "london"
    elif london_end <= t < ny_end:
        return "ny"
    else:
        # Outside all sessions (e.g., 23:50-01:05)
        return "ny"  # Still counts as NY session (overnight)
