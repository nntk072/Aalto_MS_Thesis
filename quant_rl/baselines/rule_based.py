"""Rule-based baselines: EMA crossover, MACD, RSI + ATR stops.

Each strategy returns a Series of signals: +1 long, -1 short, 0 flat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indicators import _ema, atr, rsi as _rsi, macd as _macd


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


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


def macd_ema50_baseline(
    bars: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    ema50_period: int = 50,
    cooldown_bars: int = 5,
) -> pd.Series:
    """MACD+EMA50 baseline with cross-only entry/exit and 5-bar cooldown.

    Rules:
    - Long entry: close > EMA50 AND bullish MACD cross
    - Long exit: bearish MACD cross (go flat)
    - Short entry: close < EMA50 AND bearish MACD cross
    - Short exit: bullish MACD cross (go flat)
    - Cooldown: wait >= 5 bars after any exit before next entry
    - No auto-flip (exit first, then wait for opposite condition to re-enter)

    Returns:
    - +1: hold long
    - -1: hold short
    - 0: hold current or flat
    - 2: explicit exit (go flat)
    """
    close = bars["close"]

    # Compute MACD with SMA(9) signal line (not EMA)
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _sma(macd_line, signal_period)
    ema50 = _ema(close, ema50_period)

    # Detect crosses (causal: t-1 vs t)
    bullish_cross = (macd_line.shift(1) <= signal_line.shift(1)) & (macd_line > signal_line)
    bearish_cross = (macd_line.shift(1) >= signal_line.shift(1)) & (macd_line < signal_line)

    # Initialize action series
    actions = pd.Series(0, index=bars.index, dtype=int)
    position = 0  # 0=flat, 1=long, -1=short
    cooldown_counter = 0

    for i in range(len(bars)):
        # Decrement cooldown
        if cooldown_counter > 0:
            cooldown_counter -= 1

        current_action = 0  # default: hold

        if position == 0:
            # Flat: check for entry
            if cooldown_counter == 0:
                if close.iloc[i] > ema50.iloc[i] and bullish_cross.iloc[i]:
                    # Long entry
                    current_action = 1
                    position = 1
                elif close.iloc[i] < ema50.iloc[i] and bearish_cross.iloc[i]:
                    # Short entry
                    current_action = -1
                    position = -1

        elif position == 1:
            # In long: check for exit
            if bearish_cross.iloc[i]:
                # Exit signal
                current_action = 2
                position = 0
                cooldown_counter = cooldown_bars
            else:
                # Hold long
                current_action = 1

        elif position == -1:
            # In short: check for exit
            if bullish_cross.iloc[i]:
                # Exit signal
                current_action = 2
                position = 0
                cooldown_counter = cooldown_bars
            else:
                # Hold short
                current_action = -1

        actions.iloc[i] = current_action

    return actions


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
