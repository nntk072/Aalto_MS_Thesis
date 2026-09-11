"""ICT-style structure overlays for per-trade order charts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant_rl.features.indicators import rsi as _rsi
from quant_rl.features.indicators import vwap_level
from quant_rl.features.liquidity import detect_liquidity_sweeps
from quant_rl.features.smt import smt_divergence
from quant_rl.features.structure import structure_levels

Side = Literal["high", "low"]

SWING_HIGH_COLOR = "#8e24aa"
SWING_LOW_COLOR = "#00838f"
SWEEP_COLOR = "#ef6c00"
SMT_BEAR_COLOR = "#c62828"
SMT_BULL_COLOR = "#2e7d32"
VWAP_COLOR = "#f9a825"
RSI_COLOR = "#6a1b9a"
RSI_LEVEL_COLOR = "#9e9e9e"


@dataclass(frozen=True)
class SwingRay:
    """Horizontal ray from a confirmed swing high/low going forward."""

    t0: pd.Timestamp
    t1: pd.Timestamp
    price: float
    side: Side


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
        """Keep events that intersect ``[start, end]`` and clip rays to the window."""
        swings = [
            SwingRay(t0=max(r.t0, start), t1=min(r.t1, end), price=r.price, side=r.side)
            for r in self.swings
            if r.t1 >= start and r.t0 <= end and min(r.t1, end) > max(r.t0, start)
        ]
        sweeps = [s for s in self.sweeps if s.t1 >= start and s.t0 <= end]
        smt = [s for s in self.smt if s.t1 >= start and s.t0 <= end]
        return OverlayEvents(swings=swings, sweeps=sweeps, smt=smt)


def compute_vwap_for_chart(bars: pd.DataFrame) -> pd.Series:
    """Session VWAP for charting; infers ``session_id`` / volume when missing."""
    df = bars.copy()
    if "session_id" not in df.columns:
        from quant_rl.data.session import add_session_id

        df = add_session_id(df)
    if "tickvol" not in df.columns:
        if "volume" in df.columns:
            df["tickvol"] = df["volume"]
        else:
            df["tickvol"] = 1.0
    df["tickvol"] = pd.to_numeric(df["tickvol"], errors="coerce").replace(0, np.nan).fillna(1.0)
    return vwap_level(df)


def _bar_pad(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Extend sweep lines ~0.75 bars past each of the two candles."""
    if len(index) < 2:
        return pd.Timedelta("45s")
    delta = pd.Timedelta(index[1] - index[0])
    if delta <= pd.Timedelta(0):
        return pd.Timedelta("45s")
    return delta * 0.75


def _swing_origins(levels: pd.DataFrame, swing_period: int, side: Side) -> list[tuple[int, float]]:
    """Confirmation-bar origins: (swing-bar integer position, swing price)."""
    pcol = "last_swing_high" if side == "high" else "last_swing_low"
    changed = levels[pcol].ne(levels[pcol].shift()) & levels[pcol].notna()
    out: list[tuple[int, float]] = []
    flags = changed.to_numpy()
    prices = levels[pcol].to_numpy(dtype=float)
    for i, flag in enumerate(flags):
        if not flag:
            continue
        origin_i = i - swing_period
        if origin_i < 0:
            continue
        price = float(prices[i])
        if np.isnan(price):
            continue
        out.append((origin_i, price))
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
        rays.append(
            SwingRay(
                t0=pd.Timestamp(index[origin_i]),
                t1=pd.Timestamp(index[end_i]),
                price=price,
                side=side,
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
        t0 = pd.Timestamp(index[origin_i]) - pad
        t1 = pd.Timestamp(index[i]) + pad
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
    high_origins = _swing_origins(levels, swing_period, "high")
    low_origins = _swing_origins(levels, swing_period, "low")

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


def draw_overlays_mpl(ax: Any, events: OverlayEvents) -> None:
    """Draw swing rays, sweep lines, and SMT segments on a matplotlib axis."""
    _draw_swings_mpl(ax, events.swings)
    _draw_sweeps_mpl(ax, events.sweeps)
    _draw_smt_mpl(ax, events.smt)


def _first_legend(ax: Any, label: str) -> bool:
    handles, labels = ax.get_legend_handles_labels()
    return label not in labels


def _draw_swings_mpl(ax: Any, swings: list[SwingRay]) -> None:
    for ray in swings:
        color = SWING_HIGH_COLOR if ray.side == "high" else SWING_LOW_COLOR
        label = "Swing High" if ray.side == "high" else "Swing Low"
        ax.plot(
            [ray.t0, ray.t1],
            [ray.price, ray.price],
            color=color,
            linewidth=1.0,
            linestyle="--",
            solid_capstyle="butt",
            label=label if _first_legend(ax, label) else None,
            zorder=2,
        )
        ax.scatter(
            [ray.t0],
            [ray.price],
            s=18,
            color=color,
            zorder=4,
            marker="o",
        )


def _draw_sweeps_mpl(ax: Any, sweeps: list[SweepLine]) -> None:
    for line in sweeps:
        ax.plot(
            [line.t0, line.t1],
            [line.price, line.price],
            color=SWEEP_COLOR,
            linewidth=1.6,
            linestyle="-",
            label=line.label if _first_legend(ax, line.label) else None,
            zorder=3,
        )
        mid = line.t0 + (line.t1 - line.t0) * 0.9
        ax.annotate(
            line.label,
            xy=(mid, line.price),
            xytext=(0, -3),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=SWEEP_COLOR,
            fontweight="bold",
            zorder=6,
        )


def _draw_smt_mpl(ax: Any, segments: list[SmtSegment]) -> None:
    for seg in segments:
        color = SMT_BEAR_COLOR if seg.side == "high" else SMT_BULL_COLOR
        ax.plot(
            [seg.t0, seg.t1],
            [seg.p0, seg.p1],
            color=color,
            linewidth=1.4,
            linestyle="-.",
            label=seg.label if _first_legend(ax, seg.label) else None,
            zorder=4,
        )
        mid_t = seg.t0 + (seg.t1 - seg.t0) * 0.5
        mid_p = (seg.p0 + seg.p1) / 2.0
        ax.annotate(
            seg.label,
            xy=(mid_t, mid_p),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
            zorder=6,
        )


def draw_overlays_plotly(fig: Any, events: OverlayEvents, *, row: int = 1, col: int = 1) -> None:
    """Draw swing rays, sweep lines, and SMT segments on a Plotly figure."""
    import plotly.graph_objects as go

    seen: set[str] = set()

    def _legend(name: str) -> bool:
        if name in seen:
            return False
        seen.add(name)
        return True

    for ray in events.swings:
        name = "Swing High" if ray.side == "high" else "Swing Low"
        color = SWING_HIGH_COLOR if ray.side == "high" else SWING_LOW_COLOR
        fig.add_trace(
            go.Scatter(
                x=[ray.t0, ray.t1],
                y=[ray.price, ray.price],
                mode="lines+markers",
                name=name,
                showlegend=_legend(name),
                line=dict(color=color, width=1.2, dash="dash"),
                marker=dict(size=6, color=color, symbol="circle"),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    for line in events.sweeps:
        fig.add_trace(
            go.Scatter(
                x=[line.t0, line.t1],
                y=[line.price, line.price],
                mode="lines",
                name=line.label,
                showlegend=_legend(line.label),
                line=dict(color=SWEEP_COLOR, width=2),
                hovertemplate=f"{line.label}: %{{y:.2f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        mid = line.t0 + (line.t1 - line.t0) * 0.9
        fig.add_annotation(
            x=mid,
            y=line.price,
            text=line.label,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            yshift=-4,
            font=dict(size=11, color=SWEEP_COLOR),
            row=row,
            col=col,
        )
    for seg in events.smt:
        color = SMT_BEAR_COLOR if seg.side == "high" else SMT_BULL_COLOR
        fig.add_trace(
            go.Scatter(
                x=[seg.t0, seg.t1],
                y=[seg.p0, seg.p1],
                mode="lines",
                name=seg.label,
                showlegend=_legend(seg.label),
                line=dict(color=color, width=1.6, dash="dashdot"),
                hovertemplate=f"{seg.label}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        mid_t = seg.t0 + (seg.t1 - seg.t0) * 0.5
        mid_p = (seg.p0 + seg.p1) / 2.0
        fig.add_annotation(
            x=mid_t,
            y=mid_p,
            text=seg.label,
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            yshift=-4,
            font=dict(size=8, color=color),
            row=row,
            col=col,
        )


def draw_macd_rsi_mpl(ax_macd: Any, window: pd.DataFrame, overlays: dict[str, pd.Series]) -> Any:
    """MACD on the left y-axis, RSI on the right y-axis of the same axes."""
    ax_macd.plot(window.index, overlays["macd"], color="#0066cc", linewidth=1.5, label="MACD")
    ax_macd.plot(window.index, overlays["signal"], color="#ff6600", linewidth=1.5, label="Signal")
    hist = overlays["histogram"]
    ax_macd.bar(
        window.index,
        hist,
        color=["#00cc00" if h > 0 else "#ff0000" for h in hist],
        alpha=0.3,
        label="Histogram",
        width=pd.Timedelta("0.8min"),
    )
    ax_macd.axhline(0, color="#000000", linewidth=0.5, linestyle="-", alpha=0.3)
    ax_macd.set_ylabel("MACD")
    ax_rsi = ax_macd.twinx()
    ax_rsi.plot(window.index, overlays["rsi"], color=RSI_COLOR, linewidth=1.2, label="RSI")
    ax_rsi.axhline(70, color=RSI_LEVEL_COLOR, linewidth=0.6, linestyle=":", alpha=0.8)
    ax_rsi.axhline(30, color=RSI_LEVEL_COLOR, linewidth=0.6, linestyle=":", alpha=0.8)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    h1, l1 = ax_macd.get_legend_handles_labels()
    h2, l2 = ax_rsi.get_legend_handles_labels()
    ax_macd.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax_macd.grid(True, alpha=0.3)
    return ax_rsi


def draw_macd_rsi_plotly(
    fig: Any, window: pd.DataFrame, overlays: dict[str, pd.Series], *, row: int = 2
) -> None:
    """MACD (left) and RSI (right) on a Plotly subplot with a secondary y-axis."""
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=window.index, y=overlays["macd"], name="MACD", line=dict(color="#0066cc", width=2)
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=window.index, y=overlays["signal"], name="Signal", line=dict(color="#ff6600", width=2)
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    colors = ["#00cc00" if h > 0 else "#ff0000" for h in overlays["histogram"]]
    fig.add_trace(
        go.Bar(
            x=window.index,
            y=overlays["histogram"],
            name="Histogram",
            marker=dict(color=colors),
            opacity=0.3,
            showlegend=True,
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_hline(y=0, line_color="#000000", line_width=1, line_dash="solid", row=row, col=1)
    fig.add_trace(
        go.Scatter(
            x=window.index, y=overlays["rsi"], name="RSI", line=dict(color=RSI_COLOR, width=1.5)
        ),
        row=row,
        col=1,
        secondary_y=True,
    )
    rsi_levels = pd.Series(70.0, index=window.index)
    rsi_os = pd.Series(30.0, index=window.index)
    fig.add_trace(
        go.Scatter(
            x=window.index,
            y=rsi_levels,
            name="RSI 70",
            line=dict(color=RSI_LEVEL_COLOR, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=window.index,
            y=rsi_os,
            name="RSI 30",
            line=dict(color=RSI_LEVEL_COLOR, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(title_text="MACD", row=row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1, secondary_y=True)


def rsi_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI on close; used by chart overlay builders."""
    return _rsi(bars["close"], period).rename("rsi")
