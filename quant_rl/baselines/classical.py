"""Classical trading baselines: buy-and-hold, EMA/MACD/RSI, multi-level breakout.

All strategies emit continuous actions in ``[-1, 1]`` and rely on the
``TradingEnv`` entry gate and SL/TP logic for risk enforcement, so the
comparison with RL agents is apples-to-apples.
"""

from __future__ import annotations

import pandas as pd

from ..features.indicators import macd, rsi
from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Always-long baseline: emits the maximum long fraction every bar."""

    def __init__(self, n_bars: int, size: float = 1.0) -> None:
        """Store the constant long fraction.

        Args:
            n_bars: Number of bars in the evaluation window.
            size: Constant action in ``(0, 1]``.
        """
        super().__init__(n_bars)
        self._size = float(min(max(size, 0.0), 1.0))

    def _signal(self, idx: int) -> float:
        return self._size


class EMAMACDRSIStrategy(BaseStrategy):
    """Trend-following baseline combining EMA cross, MACD and RSI filter.

    Long when EMA(fast) > EMA(slow), MACD > signal and RSI is not
    overbought; short on the mirror condition; hold otherwise.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        ema_fast: int = 12,
        ema_slow: int = 26,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        """Pre-compute indicator signals from close prices.

        Args:
            bars: OHLCV DataFrame with a ``close`` column.
            ema_fast: Fast EMA span.
            ema_slow: Slow EMA span.
            rsi_period: RSI lookback.
            rsi_overbought: Block new longs above this RSI.
            rsi_oversold: Block new shorts below this RSI.
        """
        close = bars["close"].astype(float)
        ema_f = close.ewm(span=ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=ema_slow, adjust=False).mean()
        macd_df = macd(close, fast=ema_fast, slow=ema_slow)
        rsi_vals = rsi(close, period=rsi_period)

        trend_up = (ema_f > ema_s) & (macd_df["macd"] > macd_df["macd_signal"])
        trend_down = (ema_f < ema_s) & (macd_df["macd"] < macd_df["macd_signal"])
        not_overbought = rsi_vals < rsi_overbought
        not_oversold = rsi_vals > rsi_oversold

        signal = pd.Series(0.0, index=bars.index)
        signal[trend_up & not_overbought] = 1.0
        signal[trend_down & not_oversold] = -1.0
        self._signals = signal.fillna(0.0).to_numpy(dtype=float)
        super().__init__(len(self._signals))

    def _signal(self, idx: int) -> float:
        return float(self._signals[idx])
