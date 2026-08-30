"""PO3 (AMD) + IFVG configuration and rules.

This module defines the single source of truth for:
- Fair Value Gap (FVG) detection rules
- Imbalanced Fair Value Gap (IFVG) confirmation rules
- Entry trigger types (retest, LTF-FVG, close-through)

All rules are based on the PO3 methodology (Manipulation, Observation, Distribution)
with HTF/LTF IFVG confirmation, excluding MSS/BOS detection steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
        - fvg_bearish_high: High price of the FVG zone (for bearish)
    """
    if config is None:
        config = FVGConfig()

    df = bars.copy()
    n = len(df)

    fvg_bullish = pd.Series(0, index=df.index, dtype=int)
    fvg_bearish = pd.Series(0, index=df.index, dtype=int)
    fvg_bullish_low = pd.Series(float('nan'), index=df.index)
    fvg_bearish_high = pd.Series(float('nan'), index=df.index)

    # Detect FVG using 3-bar rule
    for i in range(2, n):
        bar1_high = df['high'].iloc[i - 2]
        bar1_low = df['low'].iloc[i - 2]
        bar3_high = df['high'].iloc[i]
        bar3_low = df['low'].iloc[i]

        # Bullish FVG: bar3.low > bar1.high
        if bar3_low > bar1_high:
            imbalance = bar3_low - bar1_high
            if imbalance >= config.min_imbalance_pts:
                fvg_bullish.iloc[i] = 1
                fvg_bullish_low.iloc[i] = bar1_high  # Bottom of FVG zone

        # Bearish FVG: bar3.high < bar1.low
        if bar3_high < bar1_low:
            imbalance = bar1_low - bar3_high
            if imbalance >= config.min_imbalance_pts:
                fvg_bearish.iloc[i] = 1
                fvg_bearish_high.iloc[i] = bar1_low  # Top of FVG zone

    result = pd.DataFrame(
        {
            'fvg_bullish': fvg_bullish,
            'fvg_bearish': fvg_bearish,
            'fvg_bullish_low': fvg_bullish_low,
            'fvg_bearish_high': fvg_bearish_high,
        },
        index=df.index,
    )

    return result


@dataclass
class IFVGConfig:
    """Configuration for IFVG confirmation.

    Attributes
    ----------
    close_through_threshold : float
        How far price must close beyond the FVG zone to confirm IFVG
        (as fraction of FVG size, e.g., 0.5 = 50% through the gap).
    """

    close_through_threshold: float = 0.5


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
    """
    if config is None:
        config = IFVGConfig()

    df = bars.copy()
    n = len(df)

    ifvg_bullish_confirmed = pd.Series(0, index=df.index, dtype=int)
    ifvg_bearish_confirmed = pd.Series(0, index=df.index, dtype=int)

    # Track active FVG zones
    active_fvg_bullish = []  # List of (start_idx, fvg_low, fvg_high)
    active_fvg_bearish = []

    for i in range(n):
        # Check for new FVGs
        if i < len(fvg_df) and fvg_df['fvg_bullish'].iloc[i] == 1:
            # FVG zone: from bar1_high to bar3_low
            fvg_low = fvg_df['fvg_bullish_low'].iloc[i]
            # Find the bar1 (2 bars ago)
            if i >= 2:
                fvg_high = df['high'].iloc[i - 2]
                active_fvg_bullish.append((i, fvg_low, fvg_high))

        if i < len(fvg_df) and fvg_df['fvg_bearish'].iloc[i] == 1:
            # FVG zone: from bar3_high to bar1_low
            fvg_high = fvg_df['fvg_bearish_high'].iloc[i]
            if i >= 2:
                fvg_low = df['low'].iloc[i - 2]
                active_fvg_bearish.append((i, fvg_low, fvg_high))

        # Check for IFVG confirmation (close-through)
        close_price = df['close'].iloc[i]

        # Bullish confirmation: close above FVG zone
        for fvg_start, fvg_low, fvg_high in active_fvg_bullish[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (close_price - fvg_high) / fvg_size
                if close_through_pct >= config.close_through_threshold:
                    ifvg_bullish_confirmed.iloc[i] = 1
                    active_fvg_bullish.remove((fvg_start, fvg_low, fvg_high))

        # Bearish confirmation: close below FVG zone
        for fvg_start, fvg_low, fvg_high in active_fvg_bearish[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (fvg_low - close_price) / fvg_size
                if close_through_pct >= config.close_through_threshold:
                    ifvg_bearish_confirmed.iloc[i] = 1
                    active_fvg_bearish.remove((fvg_start, fvg_low, fvg_high))

        # Remove old FVGs (older than lookback period)
        max_age = 50  # bars
        active_fvg_bullish = [(s, l, h) for s, l, h in active_fvg_bullish if i - s < max_age]
        active_fvg_bearish = [(s, l, h) for s, l, h in active_fvg_bearish if i - s < max_age]

    result = pd.DataFrame(
        {
            'ifvg_bullish_confirmed': ifvg_bullish_confirmed,
            'ifvg_bearish_confirmed': ifvg_bearish_confirmed,
        },
        index=df.index,
    )

    return result


EntryTriggerType = Literal['retest', 'ltf_fvg', 'close_through']


@dataclass
class EntryConfig:
    """Configuration for entry triggers.

    Attributes
    ----------
    retest_threshold : float
        How close price must get to IFVG zone to count as retest
        (as fraction of FVG size, e.g., 0.1 = within 10% of zone).
    ltf_timeframe : str
        Lower timeframe for LTF-FVG detection (e.g., 'M1', 'M5').
    ltf_fvg_min_imbalance : float
        Minimum imbalance for LTF FVG to trigger entry.
    """

    retest_threshold: float = 0.1
    ltf_timeframe: str = 'M1'
    ltf_fvg_min_imbalance: float = 0.0


def detect_entry_trigger(
    bars: pd.DataFrame,
    ifvg_df: pd.DataFrame,
    entry_type: EntryTriggerType = 'retest',
    config: EntryConfig | None = None,
) -> pd.DataFrame:
    """Detect entry triggers based on IFVG confirmation.

    Three entry trigger types:

    1. **retest**: Price returns to test the IFVG zone after confirmation
    2. **ltf_fvg**: LTF FVG forms in direction of the trade after confirmation
    3. **close_through**: Strong close completely through the IFVG zone

    Parameters
    ----------
    bars : pd.DataFrame
        OHLC DataFrame with 'high', 'low', 'close' columns.
    ifvg_df : pd.DataFrame
        IFVG confirmation output from detect_ifvg_confirmation().
    entry_type : EntryTriggerType
        Type of entry trigger to detect.
    config : EntryConfig, optional
        Configuration for entry detection. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index, containing:
        - entry_long: 1 if long entry trigger detected
        - entry_short: 1 if short entry trigger detected
        - entry_trigger_type: Type of trigger ('retest', 'ltf_fvg', 'close_through')
    """
    if config is None:
        config = EntryConfig()

    df = bars.copy()
    n = len(df)

    entry_long = pd.Series(0, index=df.index, dtype=int)
    entry_short = pd.Series(0, index=df.index, dtype=int)
    entry_trigger_type = pd.Series('', index=df.index, dtype=object)

    # Track confirmed IFVG zones
    confirmed_bullish_ifvg = []  # List of (idx, fvg_low, fvg_high)
    confirmed_bearish_ifvg = []

    for i in range(n):
        # Record new IFVG confirmations
        if i < len(ifvg_df):
            if ifvg_df['ifvg_bullish_confirmed'].iloc[i] == 1:
                if i >= 2:
                    fvg_low = df['low'].iloc[i - 2]
                    fvg_high = df['high'].iloc[i - 2]
                    confirmed_bullish_ifvg.append((i, fvg_low, fvg_high))

            if ifvg_df['ifvg_bearish_confirmed'].iloc[i] == 1:
                if i >= 2:
                    fvg_low = df['low'].iloc[i - 2]
                    fvg_high = df['high'].iloc[i - 2]
                    confirmed_bearish_ifvg.append((i, fvg_low, fvg_high))

        # Detect entry triggers based on type
        close_price = df['close'].iloc[i]

        if entry_type == 'retest':
            # Long entry: price retests bullish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_high) / fvg_size
                    if distance <= config.retest_threshold and close_price >= fvg_low:
                        entry_long.iloc[i] = 1
                        entry_trigger_type.iloc[i] = 'retest'
                        break

            # Short entry: price retests bearish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_low) / fvg_size
                    if distance <= config.retest_threshold and close_price <= fvg_high:
                        entry_short.iloc[i] = 1
                        entry_trigger_type.iloc[i] = 'retest'
                        break

        elif entry_type == 'close_through':
            # Long entry: strong close above bearish IFVG
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                if close_price > fvg_high:
                    entry_long.iloc[i] = 1
                    entry_trigger_type.iloc[i] = 'close_through'
                    break

            # Short entry: strong close below bullish IFVG
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                if close_price < fvg_low:
                    entry_short.iloc[i] = 1
                    entry_trigger_type.iloc[i] = 'close_through'
                    break

        elif entry_type == 'ltf_fvg':
            # LTF FVG detection would require multi-timeframe data
            # For now, use a simplified version based on M1 bars
            # This is a placeholder - full implementation needs LTF data
            pass

        # Clean up old IFVGs
        max_age = 50
        confirmed_bullish_ifvg = [(idx, l, h) for idx, l, h in confirmed_bullish_ifvg if i - idx < max_age]
        confirmed_bearish_ifvg = [(idx, l, h) for idx, l, h in confirmed_bearish_ifvg if i - idx < max_age]

    result = pd.DataFrame(
        {
            'entry_long': entry_long,
            'entry_short': entry_short,
            'entry_trigger_type': entry_trigger_type,
        },
        index=df.index,
    )

    return result
