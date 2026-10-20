"""Market-candle validity and in-trade wick semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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


__all__ = ["isolated_spike_mask", "ohlc_ok_mask", "tradable_mask", "wick_time"]
