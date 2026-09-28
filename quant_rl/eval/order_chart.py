"""Entry-to-exit position box and candle-anchored MAE/MFE/SL/TP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import pandas as pd

from quant_rl.eval.chart_levels import (
    at_sample_extreme,
    chart_ylim,
    deviation_levels,
    level_near_candles,
    tradable_mask,
)
from quant_rl.eval.overlay_events import OverlayEvents
from quant_rl.eval.trade_metrics import TradeChartMetrics

_ENTRY_LINE = "#546e7a"
_RED = "#e53935"
_GREEN = "#2e7d32"
_MAE = "#ef9a9a"
_MFE = "#a5d6a7"


@dataclass
class LevelSeg:
    """Horizontal segment tied to a candle when one exists."""

    label: str
    price: float
    t0: pd.Timestamp
    t1: pd.Timestamp
    color: str
    linestyle: str
    mark: pd.Timestamp | None = None


@dataclass
class OrderLevels:
    """Geometry for the position box. Far targets stay out of ``ylim``."""

    ylim: tuple[float, float]
    t_open: pd.Timestamp
    t_close: pd.Timestamp
    entry: float
    direction: int
    exit_price: float
    red: tuple[float, float] | None = None
    green: tuple[float, float] | None = None
    segs: list[LevelSeg] = field(default_factory=list)
    deviations: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    offset: float = 1.0
    manip_spans: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    manip_ends: list[pd.Timestamp] = field(default_factory=list)


def _active_swing(events: OverlayEvents, t: pd.Timestamp, side: str) -> float | None:
    best_t: pd.Timestamp | None = None
    best_p: float | None = None
    for ray in events.swings:
        if ray.side != side:
            continue
        origin = ray.pivot_time()
        if origin <= t <= ray.t1 and (best_t is None or origin >= best_t):
            best_t = origin
            best_p = ray.price
    return best_p


def _candle_bounds(window: pd.DataFrame) -> tuple[float, float]:
    mask = tradable_mask(window)
    good = window.loc[mask] if mask.any() else window
    return float(good["low"].min()), float(good["high"].max())


def _in_window(ts: pd.Timestamp | None, window: pd.DataFrame) -> bool:
    if ts is None or window.empty:
        return False
    idx = pd.DatetimeIndex(window.index)
    loc = int(idx.get_indexer(pd.Index([pd.Timestamp(ts)]), method="nearest")[0])
    if loc < 0:
        return False
    return abs(pd.Timestamp(idx[loc]) - pd.Timestamp(ts)) <= pd.Timedelta("1min")


def _on_chart(
    price: float | None, when: pd.Timestamp | None, window: pd.DataFrame, lo: float, hi: float
) -> bool:
    if price is None:
        return False
    if _in_window(when, window):
        return True
    return level_near_candles(float(price), lo, hi)


def _finite_at(frame: pd.DataFrame, column: str, when: pd.Timestamp) -> float | None:
    if column not in frame.columns:
        return None
    if when not in frame.index:
        return None
    value = frame.at[when, column]
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _stamp(ts: object) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(str(ts))


def _manipulation_marks(
    features: pd.DataFrame,
    window: pd.DataFrame,
    t_open: pd.Timestamp,
    t_close: pd.Timestamp,
    direction: int,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[pd.Timestamp], list[LevelSeg]]:
    """Active spans, the end bar, and the prices already on the feature row."""
    aligned = features.reindex(window.index)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    ends: list[pd.Timestamp] = []
    if "po3_manipulation_active" in aligned.columns:
        active = pd.to_numeric(aligned["po3_manipulation_active"], errors="coerce").fillna(0.0)
        start: pd.Timestamp | None = None
        prev: pd.Timestamp | None = None
        for ts, flag in active.items():
            stamp = _stamp(ts)
            if float(flag) > 0.0:
                if start is None:
                    start = stamp
                prev = stamp
            elif start is not None and prev is not None:
                spans.append((start, prev))
                start = None
                prev = None
        if start is not None and prev is not None:
            spans.append((start, prev))
    if "po3_manipulation_end" in aligned.columns:
        ended = pd.to_numeric(aligned["po3_manipulation_end"], errors="coerce").fillna(0.0)
        ends = [_stamp(ts) for ts, flag in ended.items() if float(flag) > 0.0]
    names = (
        ("asian_high", "Asian high"),
        ("asian_low", "Asian low"),
        ("london_high", "London high"),
        ("london_low", "London low"),
        ("po3_manipulation_high", "manip high"),
        ("po3_manipulation_low", "manip low"),
        ("sweep_low_level" if direction == 1 else "sweep_high_level", "sweep"),
    )
    segs: list[LevelSeg] = []
    for column, label in names:
        price = _finite_at(aligned, column, t_open)
        if price is None:
            continue
        segs.append(LevelSeg(label, price, t_open, t_close, "#6a1b9a", "--"))
    return spans, ends, segs


def order_levels(
    window: pd.DataFrame,
    metrics: TradeChartMetrics,
    events: OverlayEvents,
    bars: pd.DataFrame,
    t_open: pd.Timestamp,
    t_close: pd.Timestamp,
    open_row: pd.Series,
    *,
    show_mae_mfe: bool = True,
    show_sl_tp: bool = True,
    features: pd.DataFrame | None = None,
) -> OrderLevels:
    """Position box, segments, and notes. A far RR target does not set the axis."""
    lo, hi = _candle_bounds(window)
    span = max(hi - lo, 1e-6)
    offset = max(span * 0.015, 0.5)
    segs: list[LevelSeg] = []
    extra: list[float] = [metrics.entry_price, metrics.exit_price]

    if (
        show_mae_mfe
        and metrics.mae_drawn
        and _on_chart(metrics.mae_price, metrics.mae_time, window, lo, hi)
    ):
        segs.append(
            LevelSeg(
                "MAE", metrics.mae_price, t_open, metrics.mae_time, _MAE, "--", metrics.mae_time
            )
        )
        extra.append(metrics.mae_price)
    if (
        show_mae_mfe
        and metrics.mfe_drawn
        and _on_chart(metrics.mfe_price, metrics.mfe_time, window, lo, hi)
    ):
        segs.append(
            LevelSeg(
                "MFE", metrics.mfe_price, t_open, metrics.mfe_time, _MFE, "--", metrics.mfe_time
            )
        )
        extra.append(metrics.mfe_price)

    sl_ref = str(open_row.get("sl_ref") or "")
    tp_ref = str(open_row.get("tp_ref") or "")
    exit_mode = str(open_row.get("exit_mode") or "")
    planned = open_row.get("planned_rr")
    planned_txt = ""
    if planned is not None and pd.notna(planned):
        planned_txt = f" ({float(planned):.2f}R)"
    sl_label = f"SL {sl_ref}" if sl_ref else "SL"
    tp_label = f"TP {tp_ref}{planned_txt}" if tp_ref else f"TP{planned_txt}"
    ema_exit = exit_mode == "ema_21" or tp_ref == "ema_21"

    sl_on = show_sl_tp and _on_chart(metrics.sl_price, metrics.sl_time, window, lo, hi)
    tp_on = (
        show_sl_tp and not ema_exit and _on_chart(metrics.tp_price, metrics.tp_time, window, lo, hi)
    )
    if sl_on and metrics.sl_price is not None:
        segs.append(
            LevelSeg(sl_label, metrics.sl_price, t_open, t_close, _RED, ":", metrics.sl_time)
        )
        extra.append(metrics.sl_price)
    if tp_on and metrics.tp_price is not None:
        segs.append(
            LevelSeg(tp_label, metrics.tp_price, t_open, t_close, _GREEN, ":", metrics.tp_time)
        )
        extra.append(metrics.tp_price)

    deviations: list[tuple[float, float]] = []
    if at_sample_extreme(bars, t_open, metrics.direction):
        swing_lo = _active_swing(events, t_open, "low")
        swing_hi = _active_swing(events, t_open, "high")
        if swing_lo is not None and swing_hi is not None:
            deviations = deviation_levels(swing_lo, swing_hi, lo, hi)
            extra.extend(price for _k, price in deviations)

    y0, y1 = chart_ylim(lo, hi, extra)
    # A target with no candle still paints a band, clipped to the candle scale.
    red = _clip_band(metrics.entry_price, metrics.sl_price, y0, y1) if show_sl_tp else None
    green = _clip_band(metrics.entry_price, metrics.tp_price, y0, y1) if show_sl_tp else None

    notes: list[str] = []
    if ema_exit and show_sl_tp:
        notes.append("exit ema_21")
    elif show_sl_tp and metrics.tp_price is not None and not tp_on:
        if tp_ref:
            notes.append(tp_label)
        else:
            rr = open_row.get("rr_ratio")
            rr_txt = f" ({float(rr):.2f}R)" if rr is not None and pd.notna(rr) else ""
            notes.append(f"TP {metrics.tp_price:.2f}{rr_txt}")
    if show_sl_tp and metrics.sl_price is not None and not sl_on:
        if sl_ref:
            notes.append(sl_label)
        else:
            notes.append(f"SL {metrics.sl_price:.2f}")

    manip_spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    manip_ends: list[pd.Timestamp] = []
    if features is not None and not features.empty:
        manip_spans, manip_ends, context = _manipulation_marks(
            features, window, t_open, t_close, metrics.direction
        )
        segs.extend(context)
        extra.extend(seg.price for seg in context)
        y0, y1 = chart_ylim(lo, hi, extra)
        red = _clip_band(metrics.entry_price, metrics.sl_price, y0, y1) if show_sl_tp else None
        green = _clip_band(metrics.entry_price, metrics.tp_price, y0, y1) if show_sl_tp else None

    return OrderLevels(
        ylim=(y0, y1),
        t_open=t_open,
        t_close=t_close,
        entry=metrics.entry_price,
        direction=metrics.direction,
        exit_price=metrics.exit_price,
        red=red,
        green=green,
        segs=segs,
        deviations=deviations,
        notes=notes,
        offset=offset,
        manip_spans=manip_spans,
        manip_ends=manip_ends,
    )


def _clip_band(
    entry: float, other: float | None, y0: float, y1: float
) -> tuple[float, float] | None:
    if other is None:
        return None
    a, b = (entry, other) if entry <= other else (other, entry)
    a = max(a, y0)
    b = min(b, y1)
    if b <= a:
        return None
    return a, b


def draw_order_levels_mpl(ax: Any, levels: OrderLevels) -> None:
    """Light SL/TP fills, the entry dash, candle marks, and offset arrows."""
    labeled_span = False
    for t0, t1 in levels.manip_spans:
        ax.axvspan(
            t0,
            t1,
            color="#7e57c2",
            alpha=0.12,
            linewidth=0,
            zorder=0,
            label="manipulation active" if not labeled_span else None,
        )
        labeled_span = True
    labeled_end = False
    for ts in levels.manip_ends:
        ax.axvline(
            ts,
            color="#4527a0",
            linewidth=1.0,
            linestyle="-.",
            zorder=2,
            label="manipulation end" if not labeled_end else None,
        )
        labeled_end = True
    if levels.red is not None:
        ax.fill_between(
            [levels.t_open, levels.t_close],
            levels.red[0],
            levels.red[1],
            color=_RED,
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    if levels.green is not None:
        ax.fill_between(
            [levels.t_open, levels.t_close],
            levels.green[0],
            levels.green[1],
            color=_GREEN,
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    ax.plot(
        [levels.t_open, levels.t_close],
        [levels.entry, levels.entry],
        color=_ENTRY_LINE,
        linewidth=1.0,
        linestyle="--",
        zorder=3,
    )
    seen: set[str] = set()
    for seg in levels.segs:
        ax.plot(
            [seg.t0, seg.t1],
            [seg.price, seg.price],
            color=seg.color,
            linewidth=1.0,
            linestyle=seg.linestyle,
            label=seg.label if seg.label not in seen else None,
            zorder=3,
        )
        seen.add(seg.label)
        if seg.mark is not None:
            ax.scatter([seg.mark], [seg.price], s=18, color=seg.color, zorder=4, marker="o")
    labeled_sd = False
    for k, price in levels.deviations:
        ax.plot(
            [levels.t_open, levels.t_close],
            [price, price],
            color="#90a4ae",
            linewidth=0.6,
            linestyle=":",
            label="SD" if not labeled_sd else None,
            zorder=1,
        )
        labeled_sd = True
        ax.annotate(
            f"{k:g}",
            xy=(levels.t_close, price),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            color="#607d8b",
        )
    _arrow(
        ax, levels.t_open, levels.entry, levels.direction == 1, "#00cc00", levels.offset, "Entry"
    )
    exit_up = levels.direction != 1
    _arrow(ax, levels.t_close, levels.exit_price, exit_up, "#ff0000", levels.offset, "Exit")
    ax.set_ylim(*levels.ylim)


def _arrow(
    ax: Any,
    when: pd.Timestamp,
    price: float,
    point_up: bool,
    color: str,
    offset: float,
    label: str,
) -> None:
    """Triangle whose tip sits on ``price``, a little off the candle."""
    text_y = price - offset if point_up else price + offset
    ax.annotate(
        "",
        xy=(when, price),
        xytext=(when, text_y),
        arrowprops={"arrowstyle": "-|>", "color": color, "mutation_scale": 12, "lw": 1.2},
        zorder=5,
    )
    ax.scatter([], [], marker="^" if point_up else "v", c=color, s=40, label=label)


def draw_order_levels_plotly(fig: Any, levels: OrderLevels) -> None:
    """Plotly twin of :func:`draw_order_levels_mpl`."""
    import plotly.graph_objects as go

    for t0, t1 in levels.manip_spans:
        fig.add_vrect(
            x0=t0,
            x1=t1,
            fillcolor="#7e57c2",
            opacity=0.12,
            line_width=0,
            row=1,
            col=1,
        )
    for ts in levels.manip_ends:
        fig.add_vline(
            x=ts,
            line_color="#4527a0",
            line_width=1,
            line_dash="dash",
            row=1,
            col=1,
        )

    def _band(band: tuple[float, float] | None, color: str) -> None:
        if band is None:
            return
        fig.add_shape(
            type="rect",
            x0=levels.t_open,
            x1=levels.t_close,
            y0=band[0],
            y1=band[1],
            fillcolor=color,
            opacity=1,
            line_width=0,
            row=1,
            col=1,
        )

    _band(levels.red, "rgba(229,57,53,0.12)")
    _band(levels.green, "rgba(46,125,50,0.12)")
    fig.add_trace(
        go.Scatter(
            x=[levels.t_open, levels.t_close],
            y=[levels.entry, levels.entry],
            mode="lines",
            name="Entry price",
            line=dict(color=_ENTRY_LINE, width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )
    for seg in levels.segs:
        fig.add_trace(
            go.Scatter(
                x=[seg.t0, seg.t1],
                y=[seg.price, seg.price],
                mode="lines",
                name=seg.label,
                line=dict(
                    color=seg.color, width=1, dash="dash" if seg.linestyle == "--" else "dot"
                ),
            ),
            row=1,
            col=1,
        )
        if seg.mark is not None:
            fig.add_trace(
                go.Scatter(
                    x=[seg.mark],
                    y=[seg.price],
                    mode="markers",
                    name=seg.label,
                    showlegend=False,
                    marker=dict(size=7, color=seg.color, symbol="circle"),
                ),
                row=1,
                col=1,
            )
    for k, price in levels.deviations:
        fig.add_trace(
            go.Scatter(
                x=[levels.t_open, levels.t_close],
                y=[price, price],
                mode="lines",
                name=f"SD {k:g}",
                line=dict(color="#90a4ae", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
    entry_up = levels.direction == 1
    fig.add_trace(
        go.Scatter(
            x=[levels.t_open],
            y=[levels.entry - levels.offset if entry_up else levels.entry + levels.offset],
            mode="markers",
            name="Entry",
            marker=dict(
                symbol="triangle-up" if entry_up else "triangle-down",
                size=12,
                color="#00cc00",
            ),
        ),
        row=1,
        col=1,
    )
    exit_up = levels.direction != 1
    fig.add_trace(
        go.Scatter(
            x=[levels.t_close],
            y=[levels.exit_price - levels.offset if exit_up else levels.exit_price + levels.offset],
            mode="markers",
            name="Exit",
            marker=dict(
                symbol="triangle-up" if exit_up else "triangle-down",
                size=12,
                color="#ff0000",
            ),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=list(levels.ylim), row=1, col=1)
