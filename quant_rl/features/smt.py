"""SMT (Smart Money Technique) divergence between US500 and US100."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .swings import detect_pivots


def smt_divergence(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    swing_period: int = 5,
    corr_window: int = 20,
) -> pd.DataFrame:
    """Compute SMT divergence features.

    Args:
        primary: US100 bar DataFrame (reference instrument).
        secondary: US500 bar DataFrame aligned to primary index.
        swing_period: Bars on each side to detect a confirmed fractal pivot.
        corr_window: Rolling correlation window between closes.

    Returns:
        DataFrame with smt_bearish, smt_bullish, smt_corr, smt_spread.
    """
    sec_close = secondary["close"].reindex(primary.index, method="ffill")
    sec_aligned = secondary.reindex(primary.index, method="ffill")

    pr_piv = detect_pivots(primary, left=swing_period, right=swing_period)
    sec_piv = detect_pivots(sec_aligned, left=swing_period, right=swing_period)

    smt_bearish = (pr_piv["pivot_high_event"] & ~sec_piv["pivot_high_event"]).astype(float)
    smt_bullish = (pr_piv["pivot_low_event"] & ~sec_piv["pivot_low_event"]).astype(float)

    smt_corr = primary["close"].rolling(corr_window).corr(sec_close)
    smt_spread = (primary["close"] - sec_close) / primary["close"].replace(0, np.nan)

    return pd.DataFrame(
        {
            "smt_bearish": smt_bearish,
            "smt_bullish": smt_bullish,
            "smt_corr": smt_corr,
            "smt_spread": smt_spread,
        },
        index=primary.index,
    )
