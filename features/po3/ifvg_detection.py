from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IFVGConfig:
    """Configuration for IFVG confirmation.

    Attributes
    ----------
    close_through_threshold : float
        How far price must close beyond the FVG zone to confirm IFVG
        (as fraction of FVG size, e.g., 0.5 = 50% through the gap).
    inverse_opposing_fvg : bool
        When true, a bullish IFVG is a close above an existing bearish FVG;
        a bearish IFVG is a close below an existing bullish FVG. False keeps
        the historical same-orientation confirmation behavior.
    """

    close_through_threshold: float = 0.5
    inverse_opposing_fvg: bool = False


def detect_ifvg_confirmation(
    bars: pd.DataFrame,
    fvg_df: pd.DataFrame,
    config: IFVGConfig | None = None,
) -> pd.DataFrame:
    """Detect IFVG (Imbalanced FVG) confirmation.

    An FVG becomes an IFVG when price **closes through** the FVG zone,
    indicating rejection of the imbalance and potential reversal.

    Bullish IFVG confirmation: Price closes above the FVG zone (above bar1_high)
    Bearish IFVG confirmation: Price closes below the FVG zone (below bar1_low)

    Parameters
    ----------
    bars : pd.DataFrame
        OHLC DataFrame with 'high', 'low', 'close' columns.
    fvg_df : pd.DataFrame
        FVG detection output from detect_fvg().
    config : IFVGConfig, optional
        Configuration for IFVG confirmation. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index, containing:
        - ifvg_bullish_confirmed: 1 if bullish IFVG confirmed at this bar
        - ifvg_bearish_confirmed: 1 if bearish IFVG confirmed at this bar
        - ifvg_bullish_low: Low price of the IFVG zone (for bullish)
        - ifvg_bullish_high: High price of the IFVG zone (for bullish)
        - ifvg_bearish_low: Low price of the IFVG zone (for bearish)
        - ifvg_bearish_high: High price of the IFVG zone (for bearish)
    """
    if config is None:
        config = IFVGConfig()

    df = bars.copy()
    n = len(df)

    # Track active FVG zones
    active_fvg_bullish = []  # List of (start_idx, fvg_low, fvg_high)
    active_fvg_bearish = []

    # Pre-extract numpy arrays once: scalar access inside the loop via
    # pandas .iloc is ~10x slower than ndarray indexing over 100k+ bars.
    close_arr = df["close"].to_numpy(dtype=float)
    bull_flag = fvg_df["fvg_bullish"].to_numpy(dtype=int)
    bear_flag = fvg_df["fvg_bearish"].to_numpy(dtype=int)
    bull_low_arr = fvg_df["fvg_bullish_low"].to_numpy(dtype=float)
    bull_high_arr = fvg_df["fvg_bullish_high"].to_numpy(dtype=float)
    bear_low_arr = fvg_df["fvg_bearish_low"].to_numpy(dtype=float)
    bear_high_arr = fvg_df["fvg_bearish_high"].to_numpy(dtype=float)

    confirm_out_bull = np.zeros(n, dtype=int)
    confirm_out_bear = np.zeros(n, dtype=int)
    out_bull_low = np.full(n, np.nan)
    out_bull_high = np.full(n, np.nan)
    out_bear_low = np.full(n, np.nan)
    out_bear_high = np.full(n, np.nan)
    origin_bull = np.full(n, np.nan)
    origin_bear = np.full(n, np.nan)

    threshold = config.close_through_threshold
    max_age = 50  # bars; zones older than this are dropped

    for i in range(n):
        # Check for new FVGs
        if i < len(fvg_df):
            if bull_flag[i] == 1 and i >= 2:
                active_fvg_bullish.append((i, bull_low_arr[i], bull_high_arr[i]))
            if bear_flag[i] == 1 and i >= 2:
                active_fvg_bearish.append((i, bear_low_arr[i], bear_high_arr[i]))

        # Check for IFVG confirmation (close-through)
        close_price = close_arr[i]

        # Revised mode inverts an opposing-direction gap. The false branch
        # retains the pre-Idea same-orientation behavior for reproduction.
        bullish_sources = active_fvg_bearish if config.inverse_opposing_fvg else active_fvg_bullish
        bearish_sources = active_fvg_bullish if config.inverse_opposing_fvg else active_fvg_bearish
        for fvg_start, fvg_low, fvg_high in bullish_sources[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (close_price - fvg_high) / fvg_size
                if close_price > fvg_high and close_through_pct >= threshold:
                    confirm_out_bull[i] = 1
                    out_bull_low[i] = fvg_low
                    out_bull_high[i] = fvg_high
                    origin_bull[i] = float(fvg_start)
                    bullish_sources.remove((fvg_start, fvg_low, fvg_high))

        for fvg_start, fvg_low, fvg_high in bearish_sources[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (fvg_low - close_price) / fvg_size
                if close_price < fvg_low and close_through_pct >= threshold:
                    confirm_out_bear[i] = 1
                    out_bear_low[i] = fvg_low
                    out_bear_high[i] = fvg_high
                    origin_bear[i] = float(fvg_start)
                    bearish_sources.remove((fvg_start, fvg_low, fvg_high))

        # Remove old FVGs (older than lookback period)
        active_fvg_bullish = [(s, lo, h) for s, lo, h in active_fvg_bullish if i - s < max_age]
        active_fvg_bearish = [(s, lo, h) for s, lo, h in active_fvg_bearish if i - s < max_age]

    result = pd.DataFrame(
        {
            "ifvg_bullish_confirmed": confirm_out_bull,
            "ifvg_bearish_confirmed": confirm_out_bear,
            "ifvg_bullish_low": out_bull_low,
            "ifvg_bullish_high": out_bull_high,
            "ifvg_bearish_low": out_bear_low,
            "ifvg_bearish_high": out_bear_high,
            "ifvg_bullish_origin_bar": origin_bull,
            "ifvg_bearish_origin_bar": origin_bear,
        },
        index=df.index,
    )

    return result
