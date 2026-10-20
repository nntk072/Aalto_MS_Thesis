"""Candle checks and on-chart level placement for order plots.

MAE/MFE use only bars that pass the OHLC check and are not an isolated print.
A wide candle whose neighbors stay at the new price is news and is kept.
A fill a few spread points off the close is not a bad bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from trading.orders.levels import deviation_levels as deviation_levels
from trading.orders.levels import level_near_candles as level_near_candles

_SPIKE_ATR = 8.0


def ohlc_ok_mask(bars: pd.DataFrame) -> NDArray[np.bool_]:
    """True where open/high/low/close are finite and ordered."""
    o = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
    ok &= h + 1e-9 >= low
    ok &= (o >= low - 1e-9) & (o <= h + 1e-9)
    ok &= (c >= low - 1e-9) & (c <= h + 1e-9)
    return np.asarray(ok, dtype=bool)


def isolated_spike_mask(bars: pd.DataFrame, atr_mult: float = _SPIKE_ATR) -> NDArray[np.bool_]:
    """True for a one-bar print that both neighbors reject.

    If the next bar stays at the new price, the move is news and the mask
    stays false.
    """
    n = len(bars)
    spike = np.zeros(n, dtype=bool)
    if n < 3:
        return spike
    h = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    tr = np.where(np.isfinite(h) & np.isfinite(low), h - low, np.nan)
    raw = pd.Series(tr).rolling(20, min_periods=5).median().to_numpy(dtype=float)
    fallback = float(np.nanmedian(tr)) if np.isfinite(tr).any() else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    scale = np.asarray(np.where(np.isfinite(raw) & (raw > 0.0), raw, fallback), dtype=float)
    thr = atr_mult * scale
    hi_jump = h[1:-1] - np.maximum(h[:-2], h[2:])
    hi_back = (np.abs(h[2:] - h[:-2]) < thr[1:-1]) & (h[2:] < h[1:-1] - 0.5 * thr[1:-1])
    lo_jump = np.minimum(low[:-2], low[2:]) - low[1:-1]
    lo_back = (np.abs(low[2:] - low[:-2]) < thr[1:-1]) & (low[2:] > low[1:-1] + 0.5 * thr[1:-1])
    spike[1:-1] = ((hi_jump > thr[1:-1]) & hi_back) | ((lo_jump > thr[1:-1]) & lo_back)
    return spike


def tradable_mask(bars: pd.DataFrame) -> NDArray[np.bool_]:
    """Bars that may set MAE/MFE or the price-axis limit."""
    if bars.empty:
        return np.zeros(0, dtype=bool)
    return np.asarray(ohlc_ok_mask(bars) & ~isolated_spike_mask(bars), dtype=bool)


def wick_time(bars: pd.DataFrame, price: float, column: str) -> pd.Timestamp | None:
    """Timestamp of the bar whose ``column`` printed ``price``, if any."""
    if bars.empty or column not in bars.columns or not np.isfinite(price):
        return None
    vals = pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype=float)
    tol = max(1.0, abs(float(price)) * 1e-4)
    hit = np.isfinite(vals) & (np.abs(vals - float(price)) <= tol)
    if not hit.any():
        return None
    idx = np.flatnonzero(hit)
    best = int(idx[int(np.argmin(np.abs(vals[idx] - float(price))))])
    return pd.Timestamp(bars.index[best])


def at_sample_extreme(
    bars: pd.DataFrame,
    t_open: pd.Timestamp,
    direction: int,
    recent_bars: int = 400,
) -> bool:
    """True when the recent candles themselves hold the sample high or low.

    A reward-ratio target above the high does not count. Only a long whose
    recent bars printed the highest high up to entry (or a short at the lowest
    low) is at the extreme.
    """
    if bars.empty or direction not in (1, -1):
        return False
    hist = bars.loc[bars.index <= t_open]
    if hist.empty:
        return False
    recent = hist.iloc[-min(len(hist), recent_bars) :]
    span = max(float(recent["high"].max()) - float(recent["low"].min()), 1e-6)
    if direction == 1:
        sample = float(hist["high"].max())
        if float(recent["high"].max()) < sample - 1e-6:
            return False
        return float(recent["high"].iloc[-1]) >= sample - 0.05 * span
    sample_lo = float(hist["low"].min())
    if float(recent["low"].min()) > sample_lo + 1e-6:
        return False
    return float(recent["low"].iloc[-1]) <= sample_lo + 0.05 * span


def chart_ylim(
    candle_lo: float,
    candle_hi: float,
    extra: list[float] | None = None,
) -> tuple[float, float]:
    """Y limits from the candles, plus nearby levels, with a small margin."""
    lo = float(candle_lo)
    hi = float(candle_hi)
    for price in extra or []:
        if level_near_candles(price, candle_lo, candle_hi):
            lo = min(lo, float(price))
            hi = max(hi, float(price))
    span = max(hi - lo, 1e-6)
    pad = 0.08 * span
    return lo - pad, hi + pad
