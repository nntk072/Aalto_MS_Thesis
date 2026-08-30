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
    ifvg_bullish_low = pd.Series(float('nan'), index=df.index)
    ifvg_bearish_high = pd.Series(float('nan'), index=df.index)

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
                    ifvg_bullish_low.iloc[i] = fvg_low
                    active_fvg_bullish.remove((fvg_start, fvg_low, fvg_high))

        # Bearish confirmation: close below FVG zone
        for fvg_start, fvg_low, fvg_high in active_fvg_bearish[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (fvg_low - close_price) / fvg_size
                if close_through_pct >= config.close_through_threshold:
                    ifvg_bearish_confirmed.iloc[i] = 1
                    ifvg_bearish_high.iloc[i] = fvg_high
                    active_fvg_bearish.remove((fvg_start, fvg_low, fvg_high))

        # Remove old FVGs (older than lookback period)
        max_age = 50  # bars
        active_fvg_bullish = [(s, l, h) for s, l, h in active_fvg_bullish if i - s < max_age]
        active_fvg_bearish = [(s, l, h) for s, l, h in active_fvg_bearish if i - s < max_age]

    result = pd.DataFrame(
        {
            'ifvg_bullish_confirmed': ifvg_bullish_confirmed,
            'ifvg_bearish_confirmed': ifvg_bearish_confirmed,
            'ifvg_bullish_low': ifvg_bullish_low,
            'ifvg_bearish_high': ifvg_bearish_high,
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


def detect_htf_fvg(
    m1_bars: pd.DataFrame,
    htf: str = 'M15',
    config: FVGConfig | None = None,
) -> pd.DataFrame:
    """Detect FVG on higher timeframe and map back to M1 bars.

    This function resamples M1 data to the specified HTF, detects FVG on the
    HTF bars, then forward-fills the HTF FVG signals to each M1 bar within
    that HTF period.

    Parameters
    ----------
    m1_bars : pd.DataFrame
        M1 OHLC DataFrame with DatetimeIndex.
    htf : str
        Higher timeframe label (e.g., 'M5', 'M15', 'M30', 'H1').
    config : FVGConfig, optional
        Configuration for FVG detection. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index as M1 input, containing:
        - htf_fvg_bullish: 1 if bullish FVG detected on HTF
        - htf_fvg_bearish: 1 if bearish FVG detected on HTF
        - htf_fvg_bullish_low: Low price of HTF FVG zone
        - htf_fvg_bearish_high: High price of HTF FVG zone
    """
    from quant_rl.data.resample import resample

    # Resample to HTF
    htf_bars = resample(m1_bars, htf)  # type: ignore[arg-type]

    # Detect FVG on HTF
    if config is None:
        config = FVGConfig()
    htf_fvg = detect_fvg(htf_bars, config=config)

    # Map HTF FVG signals back to M1 bars
    # For each M1 bar, find which HTF bar it belongs to and use that HTF bar's FVG
    n = len(m1_bars)
    htf_fvg_bullish = pd.Series(0, index=m1_bars.index, dtype=int)
    htf_fvg_bearish = pd.Series(0, index=m1_bars.index, dtype=int)
    htf_fvg_bullish_low = pd.Series(float('nan'), index=m1_bars.index)
    htf_fvg_bearish_high = pd.Series(float('nan'), index=m1_bars.index)

    # For each HTF bar, forward-fill its FVG signal to all M1 bars in that HTF period
    for htf_idx in htf_bars.index:
        # Find M1 bars that belong to this HTF bar
        # HTF bar at time T covers M1 bars from T to T+htf_period-1min
        if htf_idx in htf_fvg.index:
            # Get the row position of this HTF bar in the HTF bars DataFrame
            htf_loc = htf_bars.index.get_loc(htf_idx)
            # Next HTF bar (if exists)
            next_htf_idx = htf_bars.index[htf_loc + 1] if htf_loc + 1 < len(htf_bars) else None

            # Select M1 bars in this HTF period
            if next_htf_idx is not None:
                m1_mask = (m1_bars.index >= htf_idx) & (m1_bars.index < next_htf_idx)
            else:
                m1_mask = m1_bars.index >= htf_idx

            # Assign HTF FVG values to these M1 bars
            htf_fvg_bullish.loc[m1_mask] = htf_fvg['fvg_bullish'].loc[htf_idx]
            htf_fvg_bearish.loc[m1_mask] = htf_fvg['fvg_bearish'].loc[htf_idx]
            htf_fvg_bullish_low.loc[m1_mask] = htf_fvg['fvg_bullish_low'].loc[htf_idx]
            htf_fvg_bearish_high.loc[m1_mask] = htf_fvg['fvg_bearish_high'].loc[htf_idx]

    result = pd.DataFrame(
        {
            'htf_fvg_bullish': htf_fvg_bullish,
            'htf_fvg_bearish': htf_fvg_bearish,
            'htf_fvg_bullish_low': htf_fvg_bullish_low,
            'htf_fvg_bearish_high': htf_fvg_bearish_high,
        },
        index=m1_bars.index,
    )

    return result


def detect_ltf_ifvg(
    m1_bars: pd.DataFrame,
    primary_tf: str = 'M5',
    ltf: str = 'M1',
    fvg_config: FVGConfig | None = None,
    ifvg_config: IFVGConfig | None = None,
) -> pd.DataFrame:
    """Detect IFVG on lower timeframe and map back to primary timeframe bars.

    This function resamples M1 data to the primary timeframe, detects FVG and
    IFVG confirmation on that primary TF, then maps the IFVG signals back to
    each M1 bar within that primary TF period.

    Parameters
    ----------
    m1_bars : pd.DataFrame
        M1 OHLC DataFrame with DatetimeIndex.
    primary_tf : str
        Primary timeframe label (e.g., 'M5', 'M15', 'M30').
    ltf : str
        Lower timeframe label (e.g., 'M1'). Default is 'M1'.
    fvg_config : FVGConfig, optional
        Configuration for FVG detection. Uses defaults if None.
    ifvg_config : IFVGConfig, optional
        Configuration for IFVG confirmation. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index as M1 input, containing:
        - ltf_ifvg_bullish_confirmed: 1 if bullish IFVG confirmed on LTF
        - ltf_ifvg_bearish_confirmed: 1 if bearish IFVG confirmed on LTF
        - ltf_ifvg_bullish_low: Low price of LTF IFVG zone
        - ltf_ifvg_bearish_high: High price of LTF IFVG zone
    """
    from quant_rl.data.resample import resample

    # Resample M1 to primary TF
    primary_bars = resample(m1_bars, primary_tf)  # type: ignore[arg-type]

    # Detect FVG on primary TF
    if fvg_config is None:
        fvg_config = FVGConfig()
    primary_fvg = detect_fvg(primary_bars, config=fvg_config)

    # Detect IFVG confirmation on primary TF
    if ifvg_config is None:
        ifvg_config = IFVGConfig()
    primary_ifvg = detect_ifvg_confirmation(primary_bars, primary_fvg, config=ifvg_config)

    # Map primary TF IFVG signals back to M1 bars
    n = len(m1_bars)
    ltf_ifvg_bullish = pd.Series(0, index=m1_bars.index, dtype=int)
    ltf_ifvg_bearish = pd.Series(0, index=m1_bars.index, dtype=int)
    ltf_ifvg_bullish_low = pd.Series(float('nan'), index=m1_bars.index)
    ltf_ifvg_bearish_high = pd.Series(float('nan'), index=m1_bars.index)

    # For each primary TF bar, forward-fill its IFVG signal to all M1 bars in that period
    for primary_idx in primary_bars.index:
        if primary_idx in primary_ifvg.index:
            # Get the row position of this primary bar
            primary_loc = primary_bars.index.get_loc(primary_idx)
            # Next primary bar (if exists)
            next_primary_idx = primary_bars.index[primary_loc + 1] if primary_loc + 1 < len(primary_bars) else None

            # Select M1 bars in this primary TF period
            if next_primary_idx is not None:
                m1_mask = (m1_bars.index >= primary_idx) & (m1_bars.index < next_primary_idx)
            else:
                m1_mask = m1_bars.index >= primary_idx

            # Assign primary TF IFVG values to these M1 bars
            ltf_ifvg_bullish.loc[m1_mask] = primary_ifvg['ifvg_bullish_confirmed'].loc[primary_idx]
            ltf_ifvg_bearish.loc[m1_mask] = primary_ifvg['ifvg_bearish_confirmed'].loc[primary_idx]
            ltf_ifvg_bullish_low.loc[m1_mask] = primary_ifvg['ifvg_bullish_low'].loc[primary_idx]
            ltf_ifvg_bearish_high.loc[m1_mask] = primary_ifvg['ifvg_bearish_high'].loc[primary_idx]

    result = pd.DataFrame(
        {
            'ltf_ifvg_bullish_confirmed': ltf_ifvg_bullish,
            'ltf_ifvg_bearish_confirmed': ltf_ifvg_bearish,
            'ltf_ifvg_bullish_low': ltf_ifvg_bullish_low,
            'ltf_ifvg_bearish_high': ltf_ifvg_bearish_high,
        },
        index=m1_bars.index,
    )

    return result


def detect_po3_entries(
    m1_bars: pd.DataFrame,
    htf: str = 'M15',
    primary_tf: str = 'M5',
    entry_type: EntryTriggerType = 'retest',
    fvg_config: FVGConfig | None = None,
    ifvg_config: IFVGConfig | None = None,
    entry_config: EntryConfig | None = None,
) -> pd.DataFrame:
    """Unified PO3 entry detection combining HTF FVG, LTF IFVG, and entry triggers.

    This function integrates the full PO3/IFVG detection pipeline:
    1. Detect HTF FVG (e.g., M15)
    2. Detect LTF IFVG confirmation (e.g., M5)
    3. Generate entry triggers based on specified type

    Parameters
    ----------
    m1_bars : pd.DataFrame
        M1 OHLC DataFrame with DatetimeIndex.
    htf : str
        Higher timeframe for FVG detection (default: 'M15').
    primary_tf : str
        Primary timeframe for IFVG confirmation (default: 'M5').
    entry_type : EntryTriggerType
        Type of entry trigger: 'retest', 'close_through', or 'ltf_fvg'.
    fvg_config : FVGConfig, optional
        Configuration for FVG detection.
    ifvg_config : IFVGConfig, optional
        Configuration for IFVG confirmation.
    entry_config : EntryConfig, optional
        Configuration for entry triggers.

    Returns
    -------
    pd.DataFrame
        DataFrame with same index as M1 input, containing:
        - All HTF FVG columns
        - All LTF IFVG columns
        - entry_long: 1 if long entry signal
        - entry_short: 1 if short entry signal
        - entry_trigger_type: Type of entry trigger that fired
    """
    # Step 1: Detect HTF FVG
    htf_fvg = detect_htf_fvg(m1_bars, htf=htf, config=fvg_config)

    # Step 2: Detect LTF IFVG confirmation
    ltf_ifvg = detect_ltf_ifvg(
        m1_bars,
        primary_tf=primary_tf,
        fvg_config=fvg_config,
        ifvg_config=ifvg_config,
    )

    # Step 3: Generate entry triggers based on LTF IFVG
    # Rename LTF IFVG columns to match expected format for detect_entry_trigger
    ifvg_for_entry = ltf_ifvg.rename(columns={
        'ltf_ifvg_bullish_confirmed': 'ifvg_bullish_confirmed',
        'ltf_ifvg_bearish_confirmed': 'ifvg_bearish_confirmed',
        'ltf_ifvg_bullish_low': 'ifvg_bullish_low',
        'ltf_ifvg_bearish_high': 'ifvg_bearish_high',
    })
    entry_result = detect_entry_trigger(
        m1_bars,
        ifvg_for_entry,
        entry_type=entry_type,
        config=entry_config,
    )

    # Combine all results
    result = pd.concat([htf_fvg, ltf_ifvg, entry_result], axis=1)

    return result
