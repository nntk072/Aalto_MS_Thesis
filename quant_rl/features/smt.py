"""SMT (Smart Money Technique) divergence between US500 and US100.

SMT divergence: when one instrument makes a new swing high/low but the
correlated instrument does NOT confirm → bearish/bullish divergence signal.
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
    roll_min = s.rolling(2 * period + 1, center=False).min()
    lag_min = roll_min.shift(period)
    return (s.shift(period) == lag_min).astype(int)


def smt_divergence(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    swing_period: int = 5,
    corr_window: int = 20,
) -> pd.DataFrame:
    """Compute SMT divergence features.

    Parameters
    ----------
    primary:
        US100 bar DataFrame (reference instrument).
    secondary:
        US500 bar DataFrame aligned to primary index (via merge_asof).
    swing_period:
        Bars on each side to detect a swing high/low (causal lag applied).
    corr_window:
        Rolling correlation window between primary and secondary closes.

    Returns
    -------
    DataFrame with columns:
      - ``smt_bearish``: primary new swing-high but secondary does NOT confirm
      - ``smt_bullish``: primary new swing-low but secondary does NOT confirm
      - ``smt_corr``: rolling correlation
      - ``smt_spread``: normalised close spread (primary - secondary) / primary
    """
    # Align secondary to primary index causally
    sec_close = secondary["close"].reindex(primary.index, method="ffill")

    pr_sh = _swing_highs(primary["high"], swing_period)
    sec_sh = _swing_highs(secondary["high"].reindex(primary.index, method="ffill"), swing_period)

    pr_sl = _swing_lows(primary["low"], swing_period)
    sec_sl = _swing_lows(secondary["low"].reindex(primary.index, method="ffill"), swing_period)

    smt_bearish = ((pr_sh == 1) & (sec_sh == 0)).astype(float)
    smt_bullish = ((pr_sl == 1) & (sec_sl == 0)).astype(float)

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
