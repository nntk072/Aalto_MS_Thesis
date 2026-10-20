"""Causal FVG zone feature construction."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from features.po3.fvg_detection import FVGConfig, detect_fvg
from features.po3.zones import build_fvg_zones
from features.technical.indicators import atr

_FVG_DIST_CAP = 5.0


def build_fvg_zone_features(
    bars: pd.DataFrame,
    fvg_config: FVGConfig | None = None,
    max_zone_bars: int = 50,
) -> pd.DataFrame:
    """Model-facing FVG zone features for one timeframe's own bars.

    Uses :func:`features.po3.config.detect_fvg` +
    :func:`features.po3.config.build_fvg_zones`, then per bar reports:

    - ``fvg_in_bull`` / ``fvg_in_bear``: 1 if close is inside an active zone
    - ``fvg_bull_dist`` / ``fvg_bear_dist``: distance from close to the nearest
      active zone edge, normalised by ATR and capped at 5.0 (cap value = no
      active zone nearby)
    - ``fvg_retest_bull`` / ``fvg_retest_bear``: 1 only when price has left a
      still-live zone, tags the near edge, and closes back on the trade side.
      The fill bar is not a retest.
    - ``fvg_*_low`` / ``fvg_*_high``: live zone bounds. ``fvg_*_origin`` is the
      low (bull) or high (bear) of candle 1 in the 1-2-3 that left the gap.
      Candle 2 is the displacement candle the gap belongs to.

    All values are computed from the current and earlier bars only (zone
    activity windows end at fill/expire time), so the block is causal.
    """
    n = len(bars)
    idx = bars.index
    out = pd.DataFrame(
        0.0,
        index=idx,
        columns=[
            "fvg_in_bull",
            "fvg_in_bear",
            "fvg_bull_dist",
            "fvg_bear_dist",
            "fvg_retest_bull",
            "fvg_retest_bear",
            "fvg_bull_active",
            "fvg_bear_active",
            "fvg_bull_low",
            "fvg_bull_high",
            "fvg_bear_low",
            "fvg_bear_high",
            "fvg_bull_origin",
            "fvg_bear_origin",
        ],
    )
    out["fvg_bull_dist"] = _FVG_DIST_CAP
    out["fvg_bear_dist"] = _FVG_DIST_CAP
    for col in (
        "fvg_bull_low",
        "fvg_bull_high",
        "fvg_bear_low",
        "fvg_bear_high",
        "fvg_bull_origin",
        "fvg_bear_origin",
    ):
        out[col] = np.nan

    if n == 0:
        return out

    signals = detect_fvg(bars, fvg_config)
    # build_fvg_zones() matches on the htf_fvg_* column family; rename the
    # plain detect_fvg() output so the zone builder recognises it.
    signals = signals.rename(
        columns={
            "fvg_bullish": "htf_fvg_bullish",
            "fvg_bullish_low": "htf_fvg_bullish_low",
            "fvg_bullish_high": "htf_fvg_bullish_high",
            "fvg_bearish": "htf_fvg_bearish",
            "fvg_bearish_low": "htf_fvg_bearish_low",
            "fvg_bearish_high": "htf_fvg_bearish_high",
        }
    )
    zones = build_fvg_zones(bars, signals, max_zone_bars=max_zone_bars)
    if not zones:
        return out

    atr_s = atr(bars, period=14).to_numpy()
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    bull_in = cast(int, out.columns.get_loc("fvg_in_bull"))
    bear_in = cast(int, out.columns.get_loc("fvg_in_bear"))
    bull_dist = cast(int, out.columns.get_loc("fvg_bull_dist"))
    bear_dist = cast(int, out.columns.get_loc("fvg_bear_dist"))
    bull_retest = cast(int, out.columns.get_loc("fvg_retest_bull"))
    bear_retest = cast(int, out.columns.get_loc("fvg_retest_bear"))
    bull_active = cast(int, out.columns.get_loc("fvg_bull_active"))
    bear_active = cast(int, out.columns.get_loc("fvg_bear_active"))
    bull_lo = cast(int, out.columns.get_loc("fvg_bull_low"))
    bull_hi = cast(int, out.columns.get_loc("fvg_bull_high"))
    bear_lo = cast(int, out.columns.get_loc("fvg_bear_low"))
    bear_hi = cast(int, out.columns.get_loc("fvg_bear_high"))
    bull_origin = cast(int, out.columns.get_loc("fvg_bull_origin"))
    bear_origin = cast(int, out.columns.get_loc("fvg_bear_origin"))

    # The zone/bar loop can touch millions of cells on a full M1 history.
    # DataFrame.iat performs indexer and block-manager work for every scalar
    # access, so keep the same ordered updates in a local homogeneous array
    # and wrap it back into the original column schema once at the end.
    values = out.to_numpy(copy=True)

    for zone in zones:
        start_i = int(idx.searchsorted(zone.start_ts, side="left"))
        end_i = int(idx.searchsorted(zone.end_ts, side="right"))  # exclusive
        if start_i >= n:
            continue
        fill_i = int(idx.searchsorted(zone.end_ts, side="left")) if zone.invalidated else None
        origin_i = start_i - 2
        mid = 0.5 * (zone.zone_low + zone.zone_high)
        left = False
        bull = zone.side == "bullish"
        for i in range(start_i, min(end_i, n)):
            if bull:
                edge = zone.zone_high if close[i] >= mid else zone.zone_low
            else:
                edge = zone.zone_low if close[i] <= mid else zone.zone_high
            d = abs(close[i] - edge)
            a = atr_s[i] if not np.isnan(atr_s[i]) and atr_s[i] > 0 else 1.0
            in_col, dist_col = (bull_in, bull_dist) if bull else (bear_in, bear_dist)
            dist = min(d / a, values[i, dist_col])
            values[i, dist_col] = dist
            if zone.zone_low <= close[i] <= zone.zone_high:
                values[i, in_col] = 1.0
            is_fill = fill_i is not None and i == fill_i
            if bull:
                values[i, bull_active] = 1.0
                values[i, bull_lo] = zone.zone_low
                values[i, bull_hi] = zone.zone_high
                if origin_i >= 0:
                    values[i, bull_origin] = low[origin_i]
                if (
                    not is_fill
                    and left
                    and zone.zone_low < low[i] <= zone.zone_high
                    and close[i] >= zone.zone_high
                ):
                    values[i, bull_retest] = 1.0
                if low[i] > zone.zone_high:
                    left = True
            else:
                values[i, bear_active] = 1.0
                values[i, bear_lo] = zone.zone_low
                values[i, bear_hi] = zone.zone_high
                if origin_i >= 0:
                    values[i, bear_origin] = high[origin_i]
                if (
                    not is_fill
                    and left
                    and zone.zone_low <= high[i] < zone.zone_high
                    and close[i] <= zone.zone_low
                ):
                    values[i, bear_retest] = 1.0
                if high[i] < zone.zone_low:
                    left = True
    return pd.DataFrame(values, index=idx, columns=out.columns)
