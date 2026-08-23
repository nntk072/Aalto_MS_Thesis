"""Multi-level breakout baseline over Asian/London liquidity levels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..features.indicators import atr
from ..features.structure import detect_session_levels
from .base import BaseStrategy


class MultiLevelBreakoutStrategy(BaseStrategy):
    """Breakout baseline over Asian/London liquidity levels with ATR stops.

    Enters long when price closes above the Asian high with a volume spike;
    short on the mirror condition. Exits when price crosses back by
    ``atr_stop_mult`` ATR from entry.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        volume_spike_threshold: float = 1.5,
        atr_period: int = 5,
        atr_stop_mult: float = 2.0,
        size: float = 1.0,
    ) -> None:
        """Pre-compute breakout signals from bars and session levels.

        Args:
            bars: OHLCV DataFrame with ``close`` and optionally ``volume``.
            volume_spike_threshold: Minimum spike ratio for entries.
            atr_period: Lookback for the trailing stop distance.
            atr_stop_mult: Stop distance in ATR multiples.
            size: Fraction of capital allocated per entry.
        """
        levels = detect_session_levels(bars)
        atr_vals = atr(bars, period=atr_period).ffill().fillna(0.0)
        if "volume" in bars.columns:
            vol_ma = bars["volume"].rolling(20).mean()
            spike = (bars["volume"] / vol_ma.replace(0.0, np.nan)).fillna(np.inf)
        else:
            spike = pd.Series(np.inf, index=bars.index)

        closes = bars["close"].astype(float).to_numpy()
        highs = levels.get("asian_high")
        lows = levels.get("asian_low")
        asian_high = (
            highs.to_numpy(dtype=float) if highs is not None else np.full(len(closes), np.nan)
        )
        asian_low = lows.to_numpy(dtype=float) if lows is not None else np.full(len(closes), np.nan)

        self._signals = self._simulate(
            closes,
            asian_high,
            asian_low,
            spike.to_numpy(dtype=float),
            atr_vals.to_numpy(dtype=float),
            volume_spike_threshold,
            atr_stop_mult,
            size,
        )
        super().__init__(len(self._signals))

    @staticmethod
    def _simulate(
        closes: NDArray[np.floating[Any]],
        asian_high: NDArray[np.floating[Any]],
        asian_low: NDArray[np.floating[Any]],
        spike: NDArray[np.floating[Any]],
        atr_vals: NDArray[np.floating[Any]],
        spike_thr: float,
        stop_mult: float,
        size: float,
    ) -> NDArray[np.floating[Any]]:
        """Walk forward once emitting per-bar actions."""
        actions = np.zeros(len(closes), dtype=float)
        pos_dir = 0.0
        entry_price = 0.0
        for i, price in enumerate(closes):
            if pos_dir != 0.0:
                stop_dist = stop_mult * atr_vals[i]
                hit_stop = pos_dir > 0 and price <= entry_price - stop_dist
                hit_stop = hit_stop or (pos_dir < 0 and price >= entry_price + stop_dist)
                if hit_stop:
                    pos_dir = 0.0
                else:
                    actions[i] = pos_dir * size
                    continue

            if not np.isfinite(asian_high[i]) or not np.isfinite(asian_low[i]):
                continue
            spiked = bool(spike[i] >= spike_thr)
            if price > asian_high[i] and spiked:
                pos_dir, entry_price = 1.0, price
                actions[i] = size
            elif price < asian_low[i] and spiked:
                pos_dir, entry_price = -1.0, price
                actions[i] = -size
        return actions

    def _signal(self, idx: int) -> float:
        return float(self._signals[idx])
