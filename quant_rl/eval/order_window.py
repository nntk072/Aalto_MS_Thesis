"""Which candles an order chart includes.

The position box (which prices, the fill, the marks) stays in ``order_chart``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_rl.eval.chart_indicators import slice_overlays
from quant_rl.eval.order_chart import OrderLevels, order_levels
from quant_rl.eval.overlay_events import OverlayEvents, confirmed_pivots
from quant_rl.eval.trade_metrics import TradeChartMetrics, compute_trade_metrics


@dataclass
class PreparedOrderChart:
    """One trade ready to draw. ``events`` are clipped to ``window``."""

    window: pd.DataFrame
    metrics: TradeChartMetrics
    events: OverlayEvents
    overlays: dict[str, pd.Series]
    levels: OrderLevels
    t_open: pd.Timestamp
    t_close: pd.Timestamp


def _align_ts(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Localize/convert ``ts`` to match ``index`` timezone."""
    out = pd.Timestamp(ts)
    if index.tz is None:
        return out.tz_localize(None) if out.tzinfo is not None else out
    if out.tzinfo is None:
        return out.tz_localize(index.tz)
    return out.tz_convert(index.tz)


def _ny_session_positions(
    index: pd.DatetimeIndex,
    t_open: pd.Timestamp,
    t_close: pd.Timestamp,
) -> np.ndarray[Any, Any]:
    """Integer positions of NY bars on calendar days spanned by the trade."""
    from quant_rl.data.session import ny_session_mask

    t0 = _align_ts(t_open, index)
    t1 = _align_ts(t_close, index)
    day0 = t0.normalize()
    day1 = t1.normalize()
    ny = ny_session_mask(index).to_numpy()
    norms = pd.DatetimeIndex(index).normalize()
    same_days = (norms == day0) | (norms == day1)
    return np.flatnonzero(ny & np.asarray(same_days))


def _extract_window(
    bars: pd.DataFrame,
    t_open: pd.Timestamp,
    t_close: pd.Timestamp,
    context: int,
) -> pd.DataFrame:
    """Return M1 bars around a trade, with the order away from the axis edge.

    ``context`` bars before the entry and after the exit. A trade that opens
    at 16:30 still gets earlier candles that same day, so the order sits in
    the middle. The slice does not cross into the previous or next calendar
    day.
    """
    idx = pd.DatetimeIndex(bars.index)
    i_o = int(idx.get_indexer(pd.Index([_align_ts(t_open, idx)]), method="nearest")[0])
    i_c = int(idx.get_indexer(pd.Index([_align_ts(t_close, idx)]), method="nearest")[0])
    if i_o < 0 or i_c < 0:
        return pd.DataFrame(columns=bars.columns)

    i_s = max(0, min(i_o, i_c) - context)
    i_e = min(len(bars) - 1, max(i_o, i_c) + context)
    t0 = _align_ts(t_open, idx)
    t1 = _align_ts(t_close, idx)
    day_start = t0.normalize()
    day_end = t1.normalize() + pd.Timedelta(days=1)
    while i_s < i_e and pd.Timestamp(idx[i_s]) < day_start:
        i_s += 1
    while i_e > i_s and pd.Timestamp(idx[i_e]) >= day_end:
        i_e -= 1
    return bars.iloc[i_s : i_e + 1]


def _structure_anchors(
    events: OverlayEvents,
    t_open: pd.Timestamp,
    t_close: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Pivot times of swings, sweeps, and SMT that touch the trade."""
    anchors: list[pd.Timestamp] = []
    for ray in events.swings:
        origin = ray.pivot_time()
        if ray.t1 >= t_open and origin <= t_close:
            anchors.append(origin)
    for sweep in events.sweeps:
        if sweep.t1 >= t_open and sweep.t0 <= t_close:
            anchors.append(sweep.t0)
    for seg in events.smt:
        # A segment that runs to a later week must not set the chart axis.
        # Both pivots have to sit on this session or the previous day.
        if not (seg.t1 >= t_open and seg.t0 <= t_close):
            continue
        if _on_trade_or_previous_day(seg.t0, t_open) and _on_trade_or_previous_day(
            seg.t1, t_open
        ):
            anchors.extend([seg.t0, seg.t1])
    return anchors


def _on_trade_or_previous_day(ts: pd.Timestamp, t_open: pd.Timestamp) -> bool:
    """True when ``ts`` is on the trade's calendar day or the day before."""
    ref = pd.Timestamp(t_open)
    cur = pd.Timestamp(ts)
    if ref.tzinfo is not None:
        cur = cur.tz_localize(ref.tzinfo) if cur.tzinfo is None else cur.tz_convert(ref.tzinfo)
    day = cur.normalize()
    trade_day = ref.normalize()
    return bool(day == trade_day or day == trade_day - pd.Timedelta(days=1))


def _extend_trade_window(
    bars: pd.DataFrame,
    window: pd.DataFrame,
    anchors: list[pd.Timestamp],
    t_open: pd.Timestamp,
    pad: int = 15,
) -> pd.DataFrame:
    """Include candles that structure lines are drawn from.

    Same trading day from Asia 01:05 is continuous. An anchor on the previous
    calendar day is a short pad. Older pivots are left off the chart.
    """
    if window.empty or not anchors:
        return window
    idx = pd.DatetimeIndex(bars.index)
    trade_day = _align_ts(t_open, idx).normalize()
    asia = trade_day + pd.Timedelta(hours=1, minutes=5)
    prev_day = trade_day - pd.Timedelta(days=1)
    i_end = int(idx.get_indexer(pd.Index([window.index[-1]]), method="nearest")[0])
    i_start = int(idx.get_indexer(pd.Index([window.index[0]]), method="nearest")[0])
    extra: list[pd.DataFrame] = []
    for raw in anchors:
        ts = _align_ts(raw, idx)
        loc = int(idx.get_indexer(pd.Index([ts]))[0])
        if loc < 0:
            loc = int(idx.get_indexer(pd.Index([ts]), method="nearest")[0])
            if loc < 0 or abs(pd.Timestamp(idx[loc]) - ts) > pd.Timedelta("1min"):
                continue
        matched = pd.Timestamp(idx[loc])
        day = matched.normalize()
        if day == trade_day and matched >= asia:
            if loc < i_start:
                i_start = loc
        elif day == prev_day or (day == trade_day and matched < asia):
            a = max(0, loc - pad)
            b = min(len(bars) - 1, loc + pad)
            extra.append(bars.iloc[a : b + 1])
    parts = [bars.iloc[i_start : i_end + 1], *extra]
    out = pd.concat(parts).sort_index()
    return out.loc[~out.index.duplicated(keep="first")]


def prepare_order_chart(
    bars: pd.DataFrame,
    open_row: pd.Series,
    close_row: pd.Series,
    *,
    overlay_events: OverlayEvents,
    full_overlays: dict[str, pd.Series],
    context_bars: int,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    lots: float = 1.0,
    contract_size: float = 1.0,
    show_mae_mfe: bool = True,
    show_sl_tp: bool = True,
) -> PreparedOrderChart | None:
    """Build the window, metrics, clipped events, and position box for one trade.

    The window is the trade's NY session plus ``context_bars`` before the
    entry and after the exit. Structure pivots and SL/TP/MAE/MFE times stay
    as levels on those candles. They do not pull earlier sessions onto the axis.
    ``_extend_trade_window`` is left in place for that older behaviour.

    Returns ``None`` when fewer than three bars remain. Does not draw.
    """
    t_open = pd.Timestamp(open_row["time"])
    t_close = pd.Timestamp(close_row["time"])
    window = _extract_window(bars, t_open, t_close, context_bars)
    if len(window) < 3:
        return None
    metrics = compute_trade_metrics(
        bars,
        open_row,
        close_row,
        max_loss_per_trade_usd=max_loss_per_trade_usd,
        take_profit_per_trade_usd=take_profit_per_trade_usd,
        lots=lots,
        contract_size=contract_size,
    )
    marked = confirmed_pivots(overlay_events, bars)
    levels = order_levels(
        window,
        metrics,
        marked,
        bars,
        t_open,
        t_close,
        open_row,
        show_mae_mfe=show_mae_mfe,
        show_sl_tp=show_sl_tp,
    )
    events = marked.clip(pd.Timestamp(window.index[0]), pd.Timestamp(window.index[-1]))
    return PreparedOrderChart(
        window=window,
        metrics=metrics,
        events=events,
        overlays=slice_overlays(full_overlays, window.index),
        levels=levels,
        t_open=t_open,
        t_close=t_close,
    )
