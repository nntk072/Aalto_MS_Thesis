from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FVGConfig:
    """Configuration for FVG detection.

    Attributes
    ----------
    min_imbalance_pts : float
        Minimum price imbalance required to count as FVG (in price units).
    lookback_bars : int
        Number of bars to look back for FVG detection.
    """

    min_imbalance_pts: float = 0.0
    lookback_bars: int = 3


def detect_fvg(
    bars: pd.DataFrame,
    config: FVGConfig | None = None,
) -> pd.DataFrame:
    """Detect Fair Value Gaps (FVG) using 3-bar imbalance rule.

    Bullish FVG: bar3.low > bar1.high (gap between bar1 high and bar3 low)
    Bearish FVG: bar3.high < bar1.low (gap between bar1 low and bar3 high)

    The FVG is detected **during the manipulation leg itself** (i.e., on bar 3).

    Parameters
    ----------
    bars : pd.DataFrame
        OHLC DataFrame with 'high', 'low', 'close' columns and DatetimeIndex.
    config : FVGConfig, optional
        Configuration for FVG detection. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index as input, containing:
        - fvg_bullish: 1 if bullish FVG detected at this bar, else 0
        - fvg_bearish: 1 if bearish FVG detected at this bar, else 0
        - fvg_bullish_low: Low price of the FVG zone (for bullish)
        - fvg_bullish_high: High price of the FVG zone (for bullish)
        - fvg_bearish_low: Low price of the FVG zone (for bearish)
        - fvg_bearish_high: High price of the FVG zone (for bearish)
    """
    if config is None:
        config = FVGConfig()

    high = bars["high"]
    low = bars["low"]

    # bar1 = bar at i-2, bar3 = current bar (i). shift(2) aligns bar1 with bar3.
    bar1_high = high.shift(2)
    bar1_low = low.shift(2)
    bar3_high = high
    bar3_low = low

    # Bullish FVG: bar3.low > bar1.high (with min imbalance threshold)
    bullish_imbalance = bar3_low - bar1_high
    bullish_mask = (bullish_imbalance > 0) & (bullish_imbalance >= config.min_imbalance_pts)

    # Bearish FVG: bar3.high < bar1.low (with min imbalance threshold)
    bearish_imbalance = bar1_low - bar3_high
    bearish_mask = (bearish_imbalance > 0) & (bearish_imbalance >= config.min_imbalance_pts)

    fvg_bullish = bullish_mask.astype(int)
    fvg_bearish = bearish_mask.astype(int)

    result = pd.DataFrame(
        {
            "fvg_bullish": fvg_bullish,
            "fvg_bearish": fvg_bearish,
            "fvg_bullish_low": bar1_high.where(bullish_mask),
            "fvg_bullish_high": bar3_low.where(bullish_mask),
            "fvg_bearish_low": bar3_high.where(bearish_mask),
            "fvg_bearish_high": bar1_low.where(bearish_mask),
        },
        index=bars.index,
    )

    return result
