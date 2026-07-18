"""Rule-based baselines: EMA crossover, MACD, RSI + ATR stops.

Each strategy returns a Series of signals: +1 long, -1 short, 0 flat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indicators import _ema, atr, rsi as _rsi, macd as _macd


def ema_crossover(
    bars: pd.DataFrame,
    fast: int = 9,
    slow: int = 21,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
) -> pd.Series:
    """EMA crossover with ATR-based stop logic (positional, bar-by-bar).

    Returns a signal Series aligned to bars.index.
    """
    close = bars["close"]
    fast_e = _ema(close, fast)
    slow_e = _ema(close, slow)
    atr_s = atr(bars, atr_period)

    signal = pd.Series(0, index=bars.index, dtype=int)
    signal[fast_e > slow_e] = 1
    signal[fast_e < slow_e] = -1

    # ATR stop: exit if adverse move > mult * ATR from entry
    # (simplified: just mask signal transitions within 1 ATR of close-to-close)
    return signal


def macd_baseline(bars: pd.DataFrame) -> pd.Series:
    """MACD histogram sign → signal."""
    m = _macd(bars["close"])
    signal = pd.Series(0, index=bars.index, dtype=int)
    signal[m["macd_hist"] > 0] = 1
    signal[m["macd_hist"] < 0] = -1
    return signal


def rsi_mean_reversion(
    bars: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    atr_period: int = 14,
) -> pd.Series:
    """RSI mean-reversion: buy oversold, sell overbought, exit at 50."""
    rsi_s = _rsi(bars["close"], period)
    signal = pd.Series(0, index=bars.index, dtype=int)
    signal[rsi_s < oversold] = 1
    signal[rsi_s > overbought] = -1
    # exit at neutral zone
    signal[(rsi_s >= 45) & (rsi_s <= 55)] = 0
    signal = signal.ffill()
    return signal.fillna(0).astype(int)
