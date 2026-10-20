from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fvg_detection import FVGConfig, detect_fvg
from .ifvg_detection import IFVGConfig, detect_ifvg_confirmation

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


def _invalidate_zone_on_tf(
    bars: pd.DataFrame,
    *,
    side: str,
    start_i: int,
    zone_low: float,
    zone_high: float,
) -> tuple[pd.Timestamp, bool]:
    """Scan ``bars`` after ``start_i`` for gap fill; return (end_ts, invalidated)."""
    n = len(bars)
    end_ts = bars.index[n - 1]
    invalidated = False
    for j in range(start_i + 1, n):
        if side == "bullish" and float(bars["low"].iloc[j]) <= zone_low:
            return bars.index[j], True
        if side == "bearish" and float(bars["high"].iloc[j]) >= zone_high:
            return bars.index[j], True
    return end_ts, invalidated


def build_fvg_zones_for_plot(
    m1_bars: pd.DataFrame,
    *,
    htf: str = "M15",
    ltf: str = "M5",
    fvg_config: FVGConfig | None = None,
    ifvg_config: IFVGConfig | None = None,
) -> list[FVGZone]:
    """Build drawable FVG/IFVG zones from native HTF/LTF candles (plot-only).

    Unlike :func:`build_fvg_zones`, which starts boxes at the causally delayed
    M1 0→1 edge (next HTF open), this places the left edge on the imbalance
    candle between bar1 and bar3 (bar2 open) and ends at true gap fill with
    no M1 ``max_zone_bars`` cap. Agent feature causality is unchanged.
    """
    from market_data.transforms.resample import resample

    zones: list[FVGZone] = []
    if m1_bars.empty:
        return zones

    if fvg_config is None:
        fvg_config = FVGConfig()
    if ifvg_config is None:
        ifvg_config = IFVGConfig()

    # --- HTF FVG: start at bar2 open (gap between bar1 and bar3) ---
    htf_bars = resample(m1_bars, htf)  # type: ignore[arg-type]
    if len(htf_bars) >= 3:
        htf_fvg = detect_fvg(htf_bars, config=fvg_config)
        for side, flag_col, low_col, high_col in (
            ("bullish", "fvg_bullish", "fvg_bullish_low", "fvg_bullish_high"),
            ("bearish", "fvg_bearish", "fvg_bearish_low", "fvg_bearish_high"),
        ):
            flags = htf_fvg[flag_col].to_numpy(dtype=int)
            for i in range(2, len(htf_bars)):
                if flags[i] != 1:
                    continue
                # Pulse only: skip forward-filled runs if any (detect_fvg is per-bar).
                zone_low = float(htf_fvg[low_col].iloc[i])
                zone_high = float(htf_fvg[high_col].iloc[i])
                if not np.isfinite(zone_low) or not np.isfinite(zone_high) or zone_high <= zone_low:
                    continue
                start_ts = htf_bars.index[i - 1]  # bar2 = edge between candles
                end_ts, invalidated = _invalidate_zone_on_tf(
                    htf_bars,
                    side=side,
                    start_i=i,
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                zones.append(
                    FVGZone(
                        kind="htf_fvg",
                        side=side,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        zone_low=zone_low,
                        zone_high=zone_high,
                        confirmed=False,
                        invalidated=invalidated,
                    )
                )

    # --- LTF IFVG: start at confirmation candle on LTF ---
    ltf_bars = resample(m1_bars, ltf)  # type: ignore[arg-type]
    if len(ltf_bars) >= 3:
        ltf_fvg = detect_fvg(ltf_bars, config=fvg_config)
        ltf_ifvg = detect_ifvg_confirmation(ltf_bars, ltf_fvg, config=ifvg_config)
        for side, flag_col, low_col, high_col in (
            (
                "bullish",
                "ifvg_bullish_confirmed",
                "ifvg_bullish_low",
                "ifvg_bullish_high",
            ),
            (
                "bearish",
                "ifvg_bearish_confirmed",
                "ifvg_bearish_low",
                "ifvg_bearish_high",
            ),
        ):
            flags = ltf_ifvg[flag_col].to_numpy(dtype=int)
            for i in range(len(ltf_bars)):
                if flags[i] != 1:
                    continue
                zone_low = float(ltf_ifvg[low_col].iloc[i])
                zone_high = float(ltf_ifvg[high_col].iloc[i])
                if not np.isfinite(zone_low) or not np.isfinite(zone_high) or zone_high <= zone_low:
                    continue
                # Two LTF candles left of the confirmation bar, so the box
                # covers the imbalance pair and not only the close-through.
                start_ts = ltf_bars.index[i - 2] if i >= 2 else ltf_bars.index[0]
                end_ts, invalidated = _invalidate_zone_on_tf(
                    ltf_bars,
                    side=side,
                    start_i=i,
                    zone_low=zone_low,
                    zone_high=zone_high,
                )
                zones.append(
                    FVGZone(
                        kind="ltf_ifvg",
                        side=side,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        zone_low=zone_low,
                        zone_high=zone_high,
                        confirmed=True,
                        invalidated=invalidated,
                    )
                )

    zones.sort(key=lambda z: z.start_ts)
    return zones
