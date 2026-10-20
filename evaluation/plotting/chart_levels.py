"""Candle checks and on-chart level placement for order plots.

MAE/MFE use only bars that pass the OHLC check and are not an isolated print.
A wide candle whose neighbors stay at the new price is news and is kept.
A fill a few spread points off the close is not a bad bar.
"""

from __future__ import annotations

import pandas as pd

from evaluation.metrics.candle_quality import (
    isolated_spike_mask as isolated_spike_mask,
)
from evaluation.metrics.candle_quality import (
    ohlc_ok_mask as ohlc_ok_mask,
)
from evaluation.metrics.candle_quality import (
    tradable_mask as tradable_mask,
)
from evaluation.metrics.candle_quality import (
    wick_time as wick_time,
)
from trading.orders.levels import deviation_levels as deviation_levels
from trading.orders.levels import level_near_candles as level_near_candles


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
