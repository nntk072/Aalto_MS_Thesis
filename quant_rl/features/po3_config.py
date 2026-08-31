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
        - fvg_bullish_high: High price of the FVG zone (for bullish)
        - fvg_bearish_low: Low price of the FVG zone (for bearish)
        - fvg_bearish_high: High price of the FVG zone (for bearish)
    """
    if config is None:
        config = FVGConfig()

    df = bars.copy()
    n = len(df)

    fvg_bullish = pd.Series(0, index=df.index, dtype=int)
    fvg_bearish = pd.Series(0, index=df.index, dtype=int)
    fvg_bullish_low = pd.Series(float("nan"), index=df.index)
    fvg_bullish_high = pd.Series(float("nan"), index=df.index)
    fvg_bearish_low = pd.Series(float("nan"), index=df.index)
    fvg_bearish_high = pd.Series(float("nan"), index=df.index)

    # Detect FVG using 3-bar rule
    for i in range(2, n):
        bar1_high = df["high"].iloc[i - 2]
        bar1_low = df["low"].iloc[i - 2]
        bar3_high = df["high"].iloc[i]
        bar3_low = df["low"].iloc[i]

        # Bullish FVG: bar3.low > bar1.high
        if bar3_low > bar1_high:
            imbalance = bar3_low - bar1_high
            if imbalance >= config.min_imbalance_pts:
                fvg_bullish.iloc[i] = 1
                fvg_bullish_low.iloc[i] = bar1_high  # Bottom of FVG zone
                fvg_bullish_high.iloc[i] = bar3_low  # Top of FVG zone

        # Bearish FVG: bar3.high < bar1.low
        if bar3_high < bar1_low:
            imbalance = bar1_low - bar3_high
            if imbalance >= config.min_imbalance_pts:
                fvg_bearish.iloc[i] = 1
                fvg_bearish_low.iloc[i] = bar3_high  # Bottom of FVG zone
                fvg_bearish_high.iloc[i] = bar1_low  # Top of FVG zone

    result = pd.DataFrame(
        {
            "fvg_bullish": fvg_bullish,
            "fvg_bearish": fvg_bearish,
            "fvg_bullish_low": fvg_bullish_low,
            "fvg_bullish_high": fvg_bullish_high,
            "fvg_bearish_low": fvg_bearish_low,
            "fvg_bearish_high": fvg_bearish_high,
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
        - ifvg_bullish_low: Low price of the IFVG zone (for bullish)
        - ifvg_bullish_high: High price of the IFVG zone (for bullish)
        - ifvg_bearish_low: Low price of the IFVG zone (for bearish)
        - ifvg_bearish_high: High price of the IFVG zone (for bearish)
    """
    if config is None:
        config = IFVGConfig()

    df = bars.copy()
    n = len(df)

    ifvg_bullish_confirmed = pd.Series(0, index=df.index, dtype=int)
    ifvg_bearish_confirmed = pd.Series(0, index=df.index, dtype=int)
    ifvg_bullish_low = pd.Series(float("nan"), index=df.index)
    ifvg_bullish_high = pd.Series(float("nan"), index=df.index)
    ifvg_bearish_low = pd.Series(float("nan"), index=df.index)
    ifvg_bearish_high = pd.Series(float("nan"), index=df.index)

    # Track active FVG zones
    active_fvg_bullish = []  # List of (start_idx, fvg_low, fvg_high)
    active_fvg_bearish = []

    for i in range(n):
        # Check for new FVGs
        if i < len(fvg_df) and fvg_df["fvg_bullish"].iloc[i] == 1:
            # FVG zone: from bar1_high to bar3_low
            fvg_low = fvg_df["fvg_bullish_low"].iloc[i]
            fvg_high = fvg_df["fvg_bullish_high"].iloc[i]
            if i >= 2:
                active_fvg_bullish.append((i, fvg_low, fvg_high))

        if i < len(fvg_df) and fvg_df["fvg_bearish"].iloc[i] == 1:
            # FVG zone: from bar3_high to bar1_low
            fvg_high = fvg_df["fvg_bearish_high"].iloc[i]
            fvg_low = fvg_df["fvg_bearish_low"].iloc[i]
            if i >= 2:
                active_fvg_bearish.append((i, fvg_low, fvg_high))

        # Check for IFVG confirmation (close-through)
        close_price = df["close"].iloc[i]

        # Bullish confirmation: close above FVG zone
        for fvg_start, fvg_low, fvg_high in active_fvg_bullish[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (close_price - fvg_high) / fvg_size
                if close_through_pct >= config.close_through_threshold:
                    ifvg_bullish_confirmed.iloc[i] = 1
                    ifvg_bullish_low.iloc[i] = fvg_low
                    ifvg_bullish_high.iloc[i] = fvg_high
                    active_fvg_bullish.remove((fvg_start, fvg_low, fvg_high))

        # Bearish confirmation: close below FVG zone
        for fvg_start, fvg_low, fvg_high in active_fvg_bearish[:]:
            fvg_size = fvg_high - fvg_low
            if fvg_size > 0:
                close_through_pct = (fvg_low - close_price) / fvg_size
                if close_through_pct >= config.close_through_threshold:
                    ifvg_bearish_confirmed.iloc[i] = 1
                    ifvg_bearish_low.iloc[i] = fvg_low
                    ifvg_bearish_high.iloc[i] = fvg_high
                    active_fvg_bearish.remove((fvg_start, fvg_low, fvg_high))

        # Remove old FVGs (older than lookback period)
        max_age = 50  # bars
        active_fvg_bullish = [(s, lo, h) for s, lo, h in active_fvg_bullish if i - s < max_age]
        active_fvg_bearish = [(s, lo, h) for s, lo, h in active_fvg_bearish if i - s < max_age]

    result = pd.DataFrame(
        {
            "ifvg_bullish_confirmed": ifvg_bullish_confirmed,
            "ifvg_bearish_confirmed": ifvg_bearish_confirmed,
            "ifvg_bullish_low": ifvg_bullish_low,
            "ifvg_bullish_high": ifvg_bullish_high,
            "ifvg_bearish_low": ifvg_bearish_low,
            "ifvg_bearish_high": ifvg_bearish_high,
        },
        index=df.index,
    )

    return result


EntryTriggerType = Literal["retest", "ltf_fvg", "close_through"]


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
    ltf_timeframe: str = "M1"
    ltf_fvg_min_imbalance: float = 0.0


def detect_entry_trigger(
    bars: pd.DataFrame,
    ifvg_df: pd.DataFrame,
    entry_type: EntryTriggerType = "retest",
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
    entry_trigger_type = pd.Series("", index=df.index, dtype=object)

    # Track confirmed IFVG zones
    confirmed_bullish_ifvg = []  # List of (idx, fvg_low, fvg_high)
    confirmed_bearish_ifvg = []

    for i in range(n):
        # Record new IFVG confirmations
        if i < len(ifvg_df):
            if ifvg_df["ifvg_bullish_confirmed"].iloc[i] == 1:
                if "ifvg_bullish_low" in ifvg_df.columns:
                    fvg_low = ifvg_df["ifvg_bullish_low"].iloc[i]
                else:
                    fvg_low = df["low"].iloc[i - 2] if i >= 2 else float("nan")
                if "ifvg_bullish_high" in ifvg_df.columns:
                    fvg_high = ifvg_df["ifvg_bullish_high"].iloc[i]
                else:
                    fvg_high = df["high"].iloc[i - 2] if i >= 2 else float("nan")
                if not (pd.isna(fvg_low) or pd.isna(fvg_high)):
                    confirmed_bullish_ifvg.append((i, fvg_low, fvg_high))

            if ifvg_df["ifvg_bearish_confirmed"].iloc[i] == 1:
                if "ifvg_bearish_low" in ifvg_df.columns:
                    fvg_low = ifvg_df["ifvg_bearish_low"].iloc[i]
                else:
                    fvg_low = df["low"].iloc[i - 2] if i >= 2 else float("nan")
                if "ifvg_bearish_high" in ifvg_df.columns:
                    fvg_high = ifvg_df["ifvg_bearish_high"].iloc[i]
                else:
                    fvg_high = df["high"].iloc[i - 2] if i >= 2 else float("nan")
                if not (pd.isna(fvg_low) or pd.isna(fvg_high)):
                    confirmed_bearish_ifvg.append((i, fvg_low, fvg_high))

        # Detect entry triggers based on type
        close_price = df["close"].iloc[i]

        if entry_type == "retest":
            # Long entry: price retests bullish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_high) / fvg_size
                    if distance <= config.retest_threshold and close_price >= fvg_low:
                        entry_long.iloc[i] = 1
                        entry_trigger_type.iloc[i] = "retest"
                        break

            # Short entry: price retests bearish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_low) / fvg_size
                    if distance <= config.retest_threshold and close_price <= fvg_high:
                        entry_short.iloc[i] = 1
                        entry_trigger_type.iloc[i] = "retest"
                        break

        elif entry_type == "close_through":
            # Long entry: strong close above bearish IFVG
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                if close_price > fvg_high:
                    entry_long.iloc[i] = 1
                    entry_trigger_type.iloc[i] = "close_through"
                    break

            # Short entry: strong close below bullish IFVG
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                if close_price < fvg_low:
                    entry_short.iloc[i] = 1
                    entry_trigger_type.iloc[i] = "close_through"
                    break

        elif entry_type == "ltf_fvg":
            # LTF FVG entry: price re-enters a confirmed inverted-FVG zone after
            # the IFVG confirms. The confirmed zone bounds come from the tracked
            # confirmed_*_ifvg lists (populated from ifvg_df's zone columns, which
            # detect_po3_entries renames from 'ltf_ifvg_*' to 'ifvg_*').
            for fvg_low, fvg_high in ((lo, hi) for _idx, lo, hi in confirmed_bullish_ifvg[:]):
                if fvg_low <= close_price <= fvg_high:
                    entry_long.iloc[i] = 1
                    entry_trigger_type.iloc[i] = "ltf_fvg"
                    break

            for fvg_low, fvg_high in ((lo, hi) for _idx, lo, hi in confirmed_bearish_ifvg[:]):
                if fvg_low <= close_price <= fvg_high:
                    entry_short.iloc[i] = 1
                    entry_trigger_type.iloc[i] = "ltf_fvg"
                    break

        # Clean up old IFVGs
        max_age = 50
        confirmed_bullish_ifvg = [
            (idx, lo, h) for idx, lo, h in confirmed_bullish_ifvg if i - idx < max_age
        ]
        confirmed_bearish_ifvg = [
            (idx, lo, h) for idx, lo, h in confirmed_bearish_ifvg if i - idx < max_age
        ]

    result = pd.DataFrame(
        {
            "entry_long": entry_long,
            "entry_short": entry_short,
            "entry_trigger_type": entry_trigger_type,
        },
        index=df.index,
    )

    return result


def detect_htf_fvg(
    m1_bars: pd.DataFrame,
    htf: str = "M15",
    config: FVGConfig | None = None,
) -> pd.DataFrame:
    """Detect FVG on higher timeframe and map back to M1 bars.

    This function resamples M1 data to the specified HTF, detects FVG on
    the HTF bars, then maps the signals onto the M1 spine with zero
    within-period lookahead: the signal of an HTF bar spanning [T, T+period)
    is only observable from the next HTF open (the moment the bar closes)
    onward.

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
        - htf_fvg_bullish_high: High price of HTF FVG zone
        - htf_fvg_bearish_low: Low price of HTF FVG zone
        - htf_fvg_bearish_high: High price of HTF FVG zone
    """
    from quant_rl.data.resample import resample

    # Resample to HTF
    htf_bars = resample(m1_bars, htf)  # type: ignore[arg-type]

    # Detect FVG on HTF
    if config is None:
        config = FVGConfig()
    htf_fvg = detect_fvg(htf_bars, config=config)

    # Causally map HTF FVG signals back to M1 bars. An HTF bar spanning
    # [T, T+period) only closes at the NEXT HTF open, so its signal may only
    # be observed by M1 bars from that instant onward. shift(1) moves each
    # bar's signal to its successor's open (the moment it completes) and
    # reindex(method="ffill") carries each completed bar's signal onto the M1
    # spine — M1 bar t sees only HTF bars *completed* at or before t, never an
    # HTF bar still forming. The final HTF bar's signal is dropped because it
    # never completes inside this sample.
    htf_aligned = htf_fvg.shift(1).reindex(m1_bars.index, method="ffill")

    result = pd.DataFrame(
        {
            # Binary signals: NaN only occurs before the first completed HTF
            # bar; treat that as "no signal yet" (0) to keep the 0/1 contract.
            "htf_fvg_bullish": htf_aligned["fvg_bullish"].fillna(0),
            "htf_fvg_bearish": htf_aligned["fvg_bearish"].fillna(0),
            "htf_fvg_bullish_low": htf_aligned["fvg_bullish_low"],
            "htf_fvg_bullish_high": htf_aligned["fvg_bullish_high"],
            "htf_fvg_bearish_low": htf_aligned["fvg_bearish_low"],
            "htf_fvg_bearish_high": htf_aligned["fvg_bearish_high"],
        },
        index=m1_bars.index,
    )

    return result


def detect_ltf_ifvg(
    m1_bars: pd.DataFrame,
    primary_tf: str = "M5",
    ltf: str = "M1",
    fvg_config: FVGConfig | None = None,
    ifvg_config: IFVGConfig | None = None,
) -> pd.DataFrame:
    """Detect IFVG on lower timeframe and map back to primary timeframe bars.

    This function resamples M1 data to the primary timeframe, detects FVG and
    IFVG confirmation on that primary TF, then maps the IFVG signals back to
    the M1 spine with zero within-period lookahead: a confirmation signal is
    only observable from the next primary-TF open (the moment its bar closes)
    onward.

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
        - ltf_ifvg_bullish_high: High price of LTF IFVG zone
        - ltf_ifvg_bearish_low: Low price of LTF IFVG zone
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

    # Causally map primary-TF IFVG signals back to M1 bars, with the same
    # one-period delay as detect_htf_fvg: a primary bar [T, T+period) only
    # completes at the next primary open, so confirmation signals are shifted
    # one period forward and forward-filled onto the M1 spine. The final
    # primary bar's signal is dropped since it never completes in-sample.
    primary_aligned = primary_ifvg.shift(1).reindex(m1_bars.index, method="ffill")

    result = pd.DataFrame(
        {
            "ltf_ifvg_bullish_confirmed": primary_aligned["ifvg_bullish_confirmed"].fillna(0),
            "ltf_ifvg_bearish_confirmed": primary_aligned["ifvg_bearish_confirmed"].fillna(0),
            "ltf_ifvg_bullish_low": primary_aligned["ifvg_bullish_low"],
            "ltf_ifvg_bullish_high": primary_aligned["ifvg_bullish_high"],
            "ltf_ifvg_bearish_low": primary_aligned["ifvg_bearish_low"],
            "ltf_ifvg_bearish_high": primary_aligned["ifvg_bearish_high"],
        },
        index=m1_bars.index,
    )

    return result


def detect_po3_entries(
    m1_bars: pd.DataFrame,
    htf: str = "M15",
    primary_tf: str = "M5",
    entry_type: EntryTriggerType = "retest",
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
    ifvg_for_entry = ltf_ifvg.rename(
        columns={
            "ltf_ifvg_bullish_confirmed": "ifvg_bullish_confirmed",
            "ltf_ifvg_bearish_confirmed": "ifvg_bearish_confirmed",
            "ltf_ifvg_bullish_low": "ifvg_bullish_low",
            "ltf_ifvg_bullish_high": "ifvg_bullish_high",
            "ltf_ifvg_bearish_low": "ifvg_bearish_low",
            "ltf_ifvg_bearish_high": "ifvg_bearish_high",
        }
    )
    entry_result = detect_entry_trigger(
        m1_bars,
        ifvg_for_entry,
        entry_type=entry_type,
        config=entry_config,
    )

    # Combine all results
    result = pd.concat([htf_fvg, ltf_ifvg, entry_result], axis=1)

    return result


# ---------------------------------------------------------------------------
# Zone building for visualization
# ---------------------------------------------------------------------------

_ZONE_SPECS: list[tuple[str, str, str, str, str]] = [
    # (kind, side, signal_col, zone_low_col, zone_high_col)
    ("htf_fvg", "bullish", "htf_fvg_bullish", "htf_fvg_bullish_low", "htf_fvg_bullish_high"),
    ("htf_fvg", "bearish", "htf_fvg_bearish", "htf_fvg_bearish_low", "htf_fvg_bearish_high"),
    (
        "ltf_ifvg",
        "bullish",
        "ltf_ifvg_bullish_confirmed",
        "ltf_ifvg_bullish_low",
        "ltf_ifvg_bullish_high",
    ),
    (
        "ltf_ifvg",
        "bearish",
        "ltf_ifvg_bearish_confirmed",
        "ltf_ifvg_bearish_low",
        "ltf_ifvg_bearish_high",
    ),
]


@dataclass
class FVGZone:
    """A drawable FVG/IFVG zone rectangle.

    Attributes
    ----------
    kind : str
        'htf_fvg' for a higher-timeframe FVG or 'ltf_ifvg' for a confirmed IFVG.
    side : str
        'bullish' or 'bearish'.
    start_ts : pd.Timestamp
        First bar timestamp of the zone.
    end_ts : pd.Timestamp
        Last bar timestamp: the bar where the zone was filled (invalidated) or
        where the horizon expired.
    zone_low : float
        Lower price bound of the zone.
    zone_high : float
        Upper price bound of the zone.
    confirmed : bool
        True if the zone was confirmed as an IFVG.
    invalidated : bool
        True if price traded back through the gap (zone filled) before expiry.
    """

    kind: str
    side: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    zone_low: float
    zone_high: float
    confirmed: bool
    invalidated: bool = False


def build_fvg_zones(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    max_zone_bars: int = 50,
) -> list[FVGZone]:
    """Build drawable zone rectangles from PO3 signal columns.

    Each contiguous run of a ``*_bullish``/``*_bearish`` signal (0 -> 1 edge)
    produces exactly one :class:`FVGZone`.  A zone ends when price trades back
    through the gap (``invalidated=True``) or after ``max_zone_bars`` bars.

    Parameters
    ----------
    bars : pd.DataFrame
        M1 OHLC bars (DatetimeIndex, 'high'/'low' columns) that produced the signals.
    signals : pd.DataFrame
        Output of :func:`detect_htf_fvg`, :func:`detect_ltf_ifvg`, or the
        combined output of :func:`detect_po3_entries`.  Must share the index of
        ``bars`` and contain a subset of the well-known column names.
    max_zone_bars : int
        Maximum number of bars a zone stays active before it is considered expired.

    Returns
    -------
    list[FVGZone]
        One zone per formation event, ordered by start timestamp.
    """
    zones: list[FVGZone] = []

    if signals.empty or len(bars) == 0:
        return zones

    for kind, side, sig_col, low_col, high_col in _ZONE_SPECS:
        if sig_col not in signals.columns:
            continue

        sig = signals[sig_col]
        # Only start a zone on the 0 -> 1 edge (handles forward-filled runs).
        starts = (sig == 1) & (sig.shift(1, fill_value=0) == 0)
        if low_col not in signals.columns or high_col not in signals.columns:
            continue

        for i in range(len(bars)):
            if not bool(starts.iloc[i]):
                continue
            zone_low = signals[low_col].iloc[i]
            zone_high = signals[high_col].iloc[i]
            if pd.isna(zone_low) or pd.isna(zone_high) or zone_high <= zone_low:
                continue

            start_ts = bars.index[i]
            end_iloc = min(len(bars) - 1, i + max_zone_bars)
            invalidated = False
            end_ts = bars.index[end_iloc]

            for j in range(i + 1, end_iloc + 1):
                if side == "bullish" and bars["low"].iloc[j] <= zone_low:
                    invalidated = True
                    end_ts = bars.index[j]
                    break
                if side == "bearish" and bars["high"].iloc[j] >= zone_high:
                    invalidated = True
                    end_ts = bars.index[j]
                    break

            zones.append(
                FVGZone(
                    kind=kind,
                    side=side,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    zone_low=float(zone_low),
                    zone_high=float(zone_high),
                    confirmed=(kind == "ltf_ifvg"),
                    invalidated=invalidated,
                )
            )

    zones.sort(key=lambda z: z.start_ts)
    return zones
