"""Swing, sweep, and SMT events for an order chart.

These are inputs to the window and to deviation levels. Drawing lives in
``chart_overlays``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quant_rl.features.liquidity import detect_liquidity_sweeps
from quant_rl.features.smt import smt_divergence
from quant_rl.features.structure import structure_levels

Side = Literal["high", "low"]


@dataclass(frozen=True)
class SwingRay:
    """Horizontal ray from a confirmed swing high/low going forward.

    ``origin`` is the pivot candle. ``t0`` is the visible start and may be
    later than ``origin`` after the ray is clipped to a chart window. The
    pivot dot is drawn only when ``origin`` still lies on the visible segment.
    """

    t0: pd.Timestamp
    t1: pd.Timestamp
    price: float
    side: Side
    origin: pd.Timestamp | None = None
    mark_pivot: bool = True

    def pivot_time(self) -> pd.Timestamp:
        """Candle that printed the swing, falling back to the visible start."""
        return self.origin if self.origin is not None else self.t0


@dataclass(frozen=True)
class SweepLine:
    """Horizontal line spanning the swept swing candle and the sweep candle."""

    t0: pd.Timestamp
    t1: pd.Timestamp
    price: float
    side: Side
    label: str


@dataclass(frozen=True)
class SmtSegment:
    """Line connecting two consecutive swing highs or two swing lows."""

    t0: pd.Timestamp
    p0: float
    t1: pd.Timestamp
    p1: float
    side: Side
    label: str


@dataclass
class OverlayEvents:
    """Structure events aligned to the bar index timezone."""

    swings: list[SwingRay] = field(default_factory=list)
    sweeps: list[SweepLine] = field(default_factory=list)
    smt: list[SmtSegment] = field(default_factory=list)

    def clip(self, start: pd.Timestamp, end: pd.Timestamp) -> OverlayEvents:
        """Keep events whose pivot candles lie inside ``[start, end]``.

        A swing that only passes through the window is dropped. Cutting it
        at the edge draws a dash that does not sit on its pivot. A sweep
        needs both ends on the chart. The forward end of a swing may still
        be trimmed to ``end`` so the ray does not leave the axis.
        """
        swings: list[SwingRay] = []
        for r in self.swings:
            origin = r.pivot_time()
            if origin < start or origin > end or r.t1 <= origin:
                continue
            t1 = min(r.t1, end)
            if t1 <= origin:
                continue
            swings.append(
                SwingRay(
                    t0=origin,
                    t1=t1,
                    price=r.price,
                    side=r.side,
                    origin=origin,
                    mark_pivot=r.mark_pivot,
                )
            )
        sweeps: list[SweepLine] = []
        for s in self.sweeps:
            if s.t0 < start or s.t1 > end or s.t1 <= s.t0:
                continue
            sweeps.append(s)
        smt: list[SmtSegment] = []
        for seg in self.smt:
            # Keep the segment only when both swing candles are on the chart.
            # Clipping a week-long line to the window edge leaves SMT off any candle.
            if seg.t0 < start or seg.t1 > end:
                continue
            smt.append(seg)
        return OverlayEvents(swings=swings, sweeps=sweeps, smt=smt)


def _bar_pad(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Extend sweep lines ~0.75 bars past each of the two candles."""
    if len(index) < 2:
        return pd.Timedelta("45s")
    delta = pd.Timedelta(index[1] - index[0])
    if delta <= pd.Timedelta(0):
        return pd.Timedelta("45s")
    return delta * 0.75


def _align_origin_ts(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Match ``ts`` to the timezone of ``index``."""
    out = pd.Timestamp(ts)
    if index.tz is None:
        return out.tz_localize(None) if out.tzinfo is not None else out
    if out.tzinfo is None:
        return out.tz_localize(index.tz)
    return out.tz_convert(index.tz)


def _bar_that_prints(
    wick: pd.Series,
    origin_i: int,
    price: float,
    swing_period: int,
) -> int:
    """Index of the candle whose wick is ``price``, or -1.

    The structure timestamp can land on the confirmation bar. Search a few
    bars around it for the candle that actually printed the level.
    """
    if origin_i < 0 or origin_i >= len(wick) or np.isnan(price):
        return -1
    lo = max(0, origin_i - swing_period)
    hi = min(len(wick), origin_i + swing_period + 1)
    values = wick.to_numpy(dtype=float)
    for j in range(lo, hi):
        if np.isfinite(values[j]) and np.isclose(values[j], price, atol=0.05, rtol=0.0):
            return j
    return -1


def _swing_origins(
    levels: pd.DataFrame,
    wick: pd.Series,
    index: pd.DatetimeIndex,
    swing_period: int,
    side: Side,
) -> list[tuple[int, float]]:
    """Pivot-bar origins: (swing-bar integer position, swing price).

    The pivot is ``last_swing_*_time`` on the bar where the level changes.
    ``confirmation_bar - swing_period`` is only the fallback when that time
    is missing.
    """
    pcol = "last_swing_high" if side == "high" else "last_swing_low"
    tcol = "last_swing_high_time" if side == "high" else "last_swing_low_time"
    changed = levels[pcol].ne(levels[pcol].shift()) & levels[pcol].notna()
    out: list[tuple[int, float]] = []
    flags = changed.to_numpy()
    prices = levels[pcol].to_numpy(dtype=float)
    times = levels[tcol] if tcol in levels.columns else None
    for i, flag in enumerate(flags):
        if not flag:
            continue
        price = float(prices[i])
        if np.isnan(price):
            continue
        origin_i = -1
        if times is not None and pd.notna(times.iloc[i]):
            ts = _align_origin_ts(pd.Timestamp(times.iloc[i]), index)
            loc = int(index.get_indexer(pd.Index([ts]), method="nearest")[0])
            if loc >= 0 and abs(pd.Timestamp(index[loc]) - ts) <= pd.Timedelta("1min"):
                origin_i = loc
        if origin_i < 0:
            origin_i = i - swing_period
        if origin_i < 0 or origin_i >= len(wick):
            continue
        # A stored level that misses the wick still belongs on this pivot.
        # Draw the candle's high or low so the ray is not thrown away.
        matched = _bar_that_prints(wick, origin_i, price, swing_period)
        if matched >= 0:
            origin_i = matched
            draw_px = price
        else:
            draw_px = float(wick.iloc[origin_i])
            if not np.isfinite(draw_px):
                continue
        out.append((origin_i, draw_px))
    return out


def _rays_from_origins(
    index: pd.DatetimeIndex,
    origins: list[tuple[int, float]],
    side: Side,
) -> list[SwingRay]:
    last_i = len(index) - 1
    rays: list[SwingRay] = []
    for k, (origin_i, price) in enumerate(origins):
        end_i = origins[k + 1][0] if k + 1 < len(origins) else last_i
        if end_i <= origin_i:
            end_i = last_i
        origin_ts = pd.Timestamp(index[origin_i])
        rays.append(
            SwingRay(
                t0=origin_ts,
                t1=pd.Timestamp(index[end_i]),
                price=price,
                side=side,
                origin=origin_ts,
            )
        )
    return rays


def _sweep_lines(
    index: pd.DatetimeIndex,
    sweeps: pd.DataFrame,
    origins: list[tuple[int, float]],
    side: Side,
    pad: pd.Timedelta,
) -> list[SweepLine]:
    col = "sweep_high" if side == "high" else "sweep_low"
    lvl_col = "sweep_high_level" if side == "high" else "sweep_low_level"
    label = "X"
    events = sweeps[col].to_numpy(dtype=float)
    levels = sweeps[lvl_col].to_numpy(dtype=float)
    lines: list[SweepLine] = []
    origin_i_arr = np.array([o[0] for o in origins], dtype=int)
    for i, fired in enumerate(events):
        if fired != 1.0:
            continue
        price = float(levels[i])
        if np.isnan(price):
            continue
        prior = origin_i_arr[origin_i_arr < i]
        origin_i = int(prior[-1]) if len(prior) else i
        # Endpoints are the candles themselves. A time pad draws the X
        # between bars, off the wick the sweep took.
        t0 = pd.Timestamp(index[origin_i])
        t1 = pd.Timestamp(index[i])
        if t1 <= t0:
            t1 = t0 + pad * 2
        lines.append(SweepLine(t0=t0, t1=t1, price=price, side=side, label=label))
    return lines


def _smt_segments(
    index: pd.DatetimeIndex,
    smt: pd.DataFrame,
    origins: list[tuple[int, float]],
    prices: pd.Series,
    swing_period: int,
    side: Side,
) -> list[SmtSegment]:
    col = "smt_bearish" if side == "high" else "smt_bullish"
    label = "SMT"
    flags = smt[col].to_numpy(dtype=float)
    segs: list[SmtSegment] = []
    origin_i_arr = np.array([o[0] for o in origins], dtype=int)
    for i, fired in enumerate(flags):
        if fired != 1.0:
            continue
        cur_i = i - swing_period
        if cur_i < 0:
            continue
        prior = origin_i_arr[origin_i_arr < cur_i]
        if len(prior) == 0:
            continue
        prev_i = int(prior[-1])
        segs.append(
            SmtSegment(
                t0=pd.Timestamp(index[prev_i]),
                p0=float(prices.iloc[prev_i]),
                t1=pd.Timestamp(index[cur_i]),
                p1=float(prices.iloc[cur_i]),
                side=side,
                label=label,
            )
        )
    return segs


def build_overlay_events(
    bars: pd.DataFrame,
    secondary: pd.DataFrame | None = None,
    swing_period: int = 5,
) -> OverlayEvents:
    """Compute swing rays, liquidity-sweep lines, and SMT segments on ``bars``."""
    if bars.empty or len(bars) < swing_period * 2 + 2:
        return OverlayEvents()
    idx = pd.DatetimeIndex(bars.index)
    pad = _bar_pad(idx)
    levels = structure_levels(bars, swing_period)
    sweeps = detect_liquidity_sweeps(bars, swing_period=swing_period)
    high_origins = _swing_origins(levels, bars["high"], idx, swing_period, "high")
    low_origins = _swing_origins(levels, bars["low"], idx, swing_period, "low")

    events = OverlayEvents(
        swings=_rays_from_origins(idx, high_origins, "high")
        + _rays_from_origins(idx, low_origins, "low"),
        sweeps=_sweep_lines(idx, sweeps, high_origins, "high", pad)
        + _sweep_lines(idx, sweeps, low_origins, "low", pad),
    )
    if secondary is not None and not secondary.empty:
        smt = smt_divergence(bars, secondary, swing_period=swing_period)
        events.smt = _smt_segments(
            idx, smt, high_origins, bars["high"], swing_period, "high"
        ) + _smt_segments(idx, smt, low_origins, bars["low"], swing_period, "low")
    return events


def confirmed_pivots(
    events: OverlayEvents, bars: pd.DataFrame, atol: float = 0.05
) -> OverlayEvents:
    """Drop the pivot dot when that candle's high or low is not the ray price.

    The ray itself stays. A clipped window must not grow a dot on its first bar.
    """
    if bars.empty:
        return events
    idx = pd.DatetimeIndex(bars.index)
    swings: list[SwingRay] = []
    for ray in events.swings:
        origin = ray.pivot_time()
        ts = _align_origin_ts(origin, idx)
        loc = int(idx.get_indexer(pd.Index([ts]), method="nearest")[0])
        mark = False
        if loc >= 0 and abs(pd.Timestamp(idx[loc]) - ts) <= pd.Timedelta("1min"):
            col = "high" if ray.side == "high" else "low"
            mark = bool(np.isclose(float(bars[col].iloc[loc]), ray.price, atol=atol, rtol=0.0))
        swings.append(
            SwingRay(
                t0=ray.t0,
                t1=ray.t1,
                price=ray.price,
                side=ray.side,
                origin=origin,
                mark_pivot=mark,
            )
        )
    return OverlayEvents(swings=swings, sweeps=list(events.sweeps), smt=list(events.smt))
