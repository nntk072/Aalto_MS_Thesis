from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .fvg_detection import FVGConfig, detect_fvg
from .ifvg_detection import IFVGConfig, detect_ifvg_confirmation

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

    # Track confirmed IFVG zones
    confirmed_bullish_ifvg = []  # List of (idx, fvg_low, fvg_high)
    confirmed_bearish_ifvg = []

    # Pre-extract numpy arrays once: scalar access inside the loop via
    # pandas .iloc is ~10x slower than ndarray indexing over 100k+ bars.
    close_arr = df["close"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)

    have_bull_zone = (
        "ifvg_bullish_low" in ifvg_df.columns and "ifvg_bullish_high" in ifvg_df.columns
    )
    have_bear_zone = (
        "ifvg_bearish_low" in ifvg_df.columns and "ifvg_bearish_high" in ifvg_df.columns
    )
    bull_confirm = (
        ifvg_df["ifvg_bullish_confirmed"].to_numpy(dtype=int)
        if "ifvg_bullish_confirmed" in ifvg_df.columns
        else np.zeros(n, dtype=int)
    )
    bear_confirm = (
        ifvg_df["ifvg_bearish_confirmed"].to_numpy(dtype=int)
        if "ifvg_bearish_confirmed" in ifvg_df.columns
        else np.zeros(n, dtype=int)
    )
    if have_bull_zone:
        bull_zone_low = ifvg_df["ifvg_bullish_low"].to_numpy(dtype=float)
        bull_zone_high = ifvg_df["ifvg_bullish_high"].to_numpy(dtype=float)
    else:
        # Fallback (legacy contract): zone bounds = bar at i-2 (bar1 of the FVG).
        bull_zone_low = np.full(n, np.nan)
        bull_zone_high = np.full(n, np.nan)
        bull_zone_low[2:] = low_arr[:-2]
        bull_zone_high[2:] = high_arr[:-2]
    if have_bear_zone:
        bear_zone_low = ifvg_df["ifvg_bearish_low"].to_numpy(dtype=float)
        bear_zone_high = ifvg_df["ifvg_bearish_high"].to_numpy(dtype=float)
    else:
        bear_zone_low = np.full(n, np.nan)
        bear_zone_high = np.full(n, np.nan)
        bear_zone_low[2:] = low_arr[:-2]
        bear_zone_high[2:] = high_arr[:-2]

    out_entry_long = np.zeros(n, dtype=int)
    out_entry_short = np.zeros(n, dtype=int)
    out_trigger_type: np.ndarray = np.empty(n, dtype=object)  # type: ignore[type-arg]
    out_trigger_type[:] = ""

    for i in range(n):
        # Record new IFVG confirmations
        if i < len(ifvg_df):
            if bull_confirm[i] == 1:
                fvg_low = bull_zone_low[i] if i < len(bull_zone_low) else float("nan")
                fvg_high = bull_zone_high[i] if i < len(bull_zone_high) else float("nan")
                if not (pd.isna(fvg_low) or pd.isna(fvg_high)):
                    confirmed_bullish_ifvg.append((i, fvg_low, fvg_high))

            if bear_confirm[i] == 1:
                fvg_low = bear_zone_low[i] if i < len(bear_zone_low) else float("nan")
                fvg_high = bear_zone_high[i] if i < len(bear_zone_high) else float("nan")
                if not (pd.isna(fvg_low) or pd.isna(fvg_high)):
                    confirmed_bearish_ifvg.append((i, fvg_low, fvg_high))

        # Detect entry triggers based on type
        close_price = close_arr[i]

        if entry_type == "retest":
            # Long entry: price retests bullish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_high) / fvg_size
                    if distance <= config.retest_threshold and close_price >= fvg_low:
                        out_entry_long[i] = 1
                        out_trigger_type[i] = "retest"
                        break

            # Short entry: price retests bearish IFVG zone
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                fvg_size = fvg_high - fvg_low
                if fvg_size > 0:
                    distance = abs(close_price - fvg_low) / fvg_size
                    if distance <= config.retest_threshold and close_price <= fvg_high:
                        out_entry_short[i] = 1
                        out_trigger_type[i] = "retest"
                        break

        elif entry_type == "close_through":
            # Long entry: strong close above bearish IFVG
            for idx, fvg_low, fvg_high in confirmed_bearish_ifvg[:]:
                if close_price > fvg_high:
                    out_entry_long[i] = 1
                    out_trigger_type[i] = "close_through"
                    break

            # Short entry: strong close below bullish IFVG
            for idx, fvg_low, fvg_high in confirmed_bullish_ifvg[:]:
                if close_price < fvg_low:
                    out_entry_short[i] = 1
                    out_trigger_type[i] = "close_through"
                    break

        elif entry_type == "ltf_fvg":
            # LTF FVG entry: price re-enters a confirmed inverted-FVG zone after
            # the IFVG confirms. The confirmed zone bounds come from the tracked
            # confirmed_*_ifvg lists (populated from ifvg_df's zone columns, which
            # detect_po3_entries renames from 'ltf_ifvg_*' to 'ifvg_*').
            for fvg_low, fvg_high in ((lo, hi) for _idx, lo, hi in confirmed_bullish_ifvg[:]):
                if fvg_low <= close_price <= fvg_high:
                    out_entry_long[i] = 1
                    out_trigger_type[i] = "ltf_fvg"
                    break

            for fvg_low, fvg_high in ((lo, hi) for _idx, lo, hi in confirmed_bearish_ifvg[:]):
                if fvg_low <= close_price <= fvg_high:
                    out_entry_short[i] = 1
                    out_trigger_type[i] = "ltf_fvg"
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
            "entry_long": out_entry_long,
            "entry_short": out_entry_short,
            "entry_trigger_type": out_trigger_type,
        },
        index=df.index,
    )

    return result


def detect_htf_fvg(
    m1_bars: pd.DataFrame,
    htf: str = "M15",
    config: FVGConfig | None = None,
    timeframe_bars: pd.DataFrame | None = None,
    availability_times: pd.Series | None = None,
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
        Higher timeframe label (e.g., 'M5', 'M10', 'M15', 'M30', 'H1').
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
    from market_data.transforms.resample import resample

    # Resample to HTF
    htf_bars = timeframe_bars
    if htf_bars is None:
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
    if availability_times is None:
        htf_aligned = htf_fvg.shift(1).reindex(m1_bars.index, method="ffill")
    else:
        from market_data.transforms.align import align_timeframes

        htf_aligned = align_timeframes(
            m1_bars.iloc[:, 0:0],
            {htf: htf_fvg},
            availability_times={htf: availability_times},
        ).rename(columns={f"{htf}_{column}": column for column in htf_fvg.columns})

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
    timeframe_bars: pd.DataFrame | None = None,
    availability_times: pd.Series | None = None,
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
        Primary timeframe label (e.g., 'M5', 'M10', 'M15', 'M30').
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
    from market_data.transforms.resample import resample

    # Resample M1 to primary TF
    primary_bars = timeframe_bars
    if primary_bars is None:
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
    if availability_times is None:
        primary_aligned = primary_ifvg.shift(1).reindex(m1_bars.index, method="ffill")
    else:
        from market_data.transforms.align import align_timeframes

        primary_aligned = align_timeframes(
            m1_bars.iloc[:, 0:0],
            {primary_tf: primary_ifvg},
            availability_times={primary_tf: availability_times},
        ).rename(columns={f"{primary_tf}_{column}": column for column in primary_ifvg.columns})

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
    timeframe_bars: dict[str, pd.DataFrame] | None = None,
    availability_times: dict[str, pd.Series] | None = None,
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
    htf = str(htf).upper()
    primary_tf = str(primary_tf).upper()
    timeframe_bars = timeframe_bars or {}
    availability_times = availability_times or {}
    htf_fvg = detect_htf_fvg(
        m1_bars,
        htf=htf,
        config=fvg_config,
        timeframe_bars=timeframe_bars.get(htf),
        availability_times=availability_times.get(htf),
    )

    # Step 2: Detect LTF IFVG confirmation
    ltf_ifvg = detect_ltf_ifvg(
        m1_bars,
        primary_tf=primary_tf,
        fvg_config=fvg_config,
        ifvg_config=ifvg_config,
        timeframe_bars=timeframe_bars.get(primary_tf),
        availability_times=availability_times.get(primary_tf),
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
