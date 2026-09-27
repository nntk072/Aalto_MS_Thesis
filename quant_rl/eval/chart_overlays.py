"""Draw swing, sweep, and SMT overlays. Event construction lives in overlay_events."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_rl.eval.overlay_events import OverlayEvents as OverlayEvents
from quant_rl.eval.overlay_events import SmtSegment as SmtSegment
from quant_rl.eval.overlay_events import SweepLine as SweepLine
from quant_rl.eval.overlay_events import SwingRay as SwingRay
from quant_rl.eval.overlay_events import build_overlay_events as build_overlay_events
from quant_rl.eval.overlay_events import confirmed_pivots as confirmed_pivots
from quant_rl.features.indicators import rsi as _rsi
from quant_rl.features.indicators import vwap_level

SWING_HIGH_COLOR = "#8e24aa"
SWING_LOW_COLOR = "#00838f"
SWEEP_COLOR = "#ef6c00"
SMT_BEAR_COLOR = "#c62828"
SMT_BULL_COLOR = "#2e7d32"
VWAP_COLOR = "#f9a825"
RSI_COLOR = "#6a1b9a"
RSI_LEVEL_COLOR = "#9e9e9e"

__all__ = [
    "OverlayEvents",
    "SmtSegment",
    "SweepLine",
    "SwingRay",
    "VWAP_COLOR",
    "build_overlay_events",
    "compute_vwap_for_chart",
    "confirmed_pivots",
    "draw_macd_rsi_mpl",
    "draw_macd_rsi_plotly",
    "draw_overlays_mpl",
    "draw_overlays_plotly",
    "rsi_series",
]


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
        origin = ray.origin
        if ray.mark_pivot and origin is not None and ray.t0 <= origin <= ray.t1:
            ax.scatter(
                [origin],
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
        ax.annotate(
            line.label,
            xy=(line.t1, line.price),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
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
        origin = ray.origin
        show_dot = bool(ray.mark_pivot and origin is not None and ray.t0 <= origin <= ray.t1)
        fig.add_trace(
            go.Scatter(
                x=[ray.t0, ray.t1],
                y=[ray.price, ray.price],
                mode="lines",
                name=name,
                showlegend=_legend(name),
                line=dict(color=color, width=1.2, dash="dash"),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        if show_dot:
            fig.add_trace(
                go.Scatter(
                    x=[origin],
                    y=[ray.price],
                    mode="markers",
                    name=name,
                    showlegend=False,
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
        fig.add_annotation(
            x=line.t1,
            y=line.price,
            text=line.label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=4,
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
    macd = overlays["macd"]
    signal = overlays["signal"]
    rsi = overlays["rsi"]
    ax_macd.plot(macd.index, macd.to_numpy(), color="#0066cc", linewidth=1.5, label="MACD")
    ax_macd.plot(signal.index, signal.to_numpy(), color="#ff6600", linewidth=1.5, label="Signal")
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
    ax_rsi.plot(rsi.index, rsi.to_numpy(), color=RSI_COLOR, linewidth=1.2, label="RSI")
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

    macd = overlays["macd"]
    signal = overlays["signal"]
    rsi = overlays["rsi"]
    fig.add_trace(
        go.Scatter(
            x=macd.index,
            y=macd.to_numpy(),
            name="MACD",
            connectgaps=False,
            line=dict(color="#0066cc", width=2),
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=signal.index,
            y=signal.to_numpy(),
            name="Signal",
            connectgaps=False,
            line=dict(color="#ff6600", width=2),
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
            x=rsi.index,
            y=rsi.to_numpy(),
            name="RSI",
            connectgaps=False,
            line=dict(color=RSI_COLOR, width=1.5),
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
