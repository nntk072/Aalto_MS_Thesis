"""Trade-level diagnostic charts: MAE/MFE, hold time, win rate, session heatmap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quant_rl.eval.plots import LONG_COLOR, SHORT_COLOR, _apply_style, _save
from quant_rl.eval.trade_diagnostics import (
    closed_trades_table,
    mae_mfe_table,
    rolling_win_rate,
    weekday_hour_pnl,
)

try:
    import plotly.graph_objects as go

    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _empty_mpl(title: str, msg: str, out_path: Path | str | None, dpi: int) -> Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontweight="bold")
    ax.set_axis_off()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_mae_mfe(
    table: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Scatter of MAE vs MFE in USD; marker shows whether SL/TP was reached."""
    if table.empty:
        return _empty_mpl("MAE vs MFE", "No MAE/MFE data", out_path, dpi)
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in table.iterrows():
        color = LONG_COLOR if row["win"] else SHORT_COLOR
        if row["sl_reached"]:
            marker = "x"
            size = 55
        elif row["tp_reached"]:
            marker = "^"
            size = 55
        else:
            marker = "o"
            size = 36
        ax.scatter(row["mae_usd"], row["mfe_usd"], c=color, marker=marker, s=size, zorder=3)
    lo = float(min(0.0, table["mae_usd"].min(), table["mfe_usd"].min()))
    hi = float(max(table["mae_usd"].max(), table["mfe_usd"].max(), 1.0))
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle=":", linewidth=0.9, label="MAE = MFE")
    ax.scatter([], [], c=LONG_COLOR, marker="o", label="Win")
    ax.scatter([], [], c=SHORT_COLOR, marker="o", label="Loss")
    ax.scatter([], [], c="#444444", marker="x", label="SL reached")
    ax.scatter([], [], c="#444444", marker="^", label="TP reached")
    ax.set_xlabel("MAE (USD, adverse)")
    ax.set_ylabel("MFE (USD, favorable)")
    ax.set_title("MAE vs MFE", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_hold_time_pnl(
    closed: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Hold duration versus realized PnL."""
    if closed.empty:
        return _empty_mpl("Hold time vs PnL", "No closed trades", out_path, dpi)
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [LONG_COLOR if w else SHORT_COLOR for w in closed["win"]]
    ax.scatter(closed["hold_min"], closed["pnl"], c=colors, s=28, zorder=3)
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("Hold time (minutes)")
    ax.set_ylabel("PnL (USD)")
    ax.set_title("Hold time vs PnL", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.scatter([], [], c=LONG_COLOR, label="Win")
    ax.scatter([], [], c=SHORT_COLOR, label="Loss")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_rolling_winrate(
    closed: pd.DataFrame,
    window: int = 20,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Rolling win rate across consecutive closed trades."""
    roll = rolling_win_rate(closed, window=window)
    if roll.empty:
        return _empty_mpl("Rolling win rate", "No closed trades", out_path, dpi)
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        roll.index,
        roll.to_numpy(),
        color="#1565c0",
        linewidth=1.6,
        label=f"{window}-trade win rate",
    )
    ax.axhline(50.0, color="#777777", linewidth=0.8, linestyle=":", label="50%")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Win rate (%)")
    ax.set_title(f"Rolling {window}-trade win rate", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_tod_heatmap(
    closed: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Weekday × hour-of-entry heatmap of summed PnL."""
    matrix, weekdays, hours = weekday_hour_pnl(closed)
    if closed.empty or len(hours) == 0:
        return _empty_mpl("Time-of-day PnL", "No closed trades", out_path, dpi)
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(hours) * 0.55), 4.5))
    finite = matrix[np.isfinite(matrix)]
    vabs = float(np.max(np.abs(finite))) if finite.size else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}" for h in hours])
    ax.set_yticks(range(len(weekdays)))
    ax.set_yticklabels(weekdays)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#000" if abs(v) < vabs * 0.6 else "#fff",
            )
    plt.colorbar(im, ax=ax, fraction=0.03, label="Sum PnL (USD)")
    ax.set_xlabel("Hour of entry")
    ax.set_title("Weekday × hour PnL", fontweight="bold")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _save_html(fig: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_mae_mfe_html(table: pd.DataFrame, out_path: Path | str | None = None) -> Any:
    """Interactive MAE vs MFE scatter."""
    if not _PLOTLY:
        raise ImportError("plotly is required")
    fig = go.Figure()
    if table.empty:
        fig.add_annotation(
            text="No MAE/MFE data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    else:
        for win, color, name in ((True, LONG_COLOR, "Win"), (False, SHORT_COLOR, "Loss")):
            sub = table[table["win"] == win]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["mae_usd"],
                    y=sub["mfe_usd"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=color,
                        size=9,
                        symbol=[
                            "x" if s else "triangle-up" if t else "circle"
                            for s, t in zip(sub["sl_reached"], sub["tp_reached"])
                        ],
                    ),
                    text=sub["close_type"],
                    hovertemplate="MAE $%{x:.1f}<br>MFE $%{y:.1f}<br>%{text}<extra>"
                    + name
                    + "</extra>",
                )
            )
        hi = float(max(table["mae_usd"].max(), table["mfe_usd"].max(), 1.0))
        fig.add_trace(
            go.Scatter(
                x=[0, hi],
                y=[0, hi],
                mode="lines",
                name="MAE = MFE",
                line=dict(color="#888", dash="dot", width=1),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="MAE vs MFE",
        xaxis_title="MAE (USD, adverse)",
        yaxis_title="MFE (USD, favorable)",
        height=520,
    )
    if out_path:
        _save_html(fig, out_path)
    return fig


def plot_hold_time_pnl_html(closed: pd.DataFrame, out_path: Path | str | None = None) -> Any:
    """Interactive hold time vs PnL scatter."""
    if not _PLOTLY:
        raise ImportError("plotly is required")
    fig = go.Figure()
    if closed.empty:
        fig.add_annotation(
            text="No closed trades", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    else:
        colors = [LONG_COLOR if w else SHORT_COLOR for w in closed["win"]]
        fig.add_trace(
            go.Scatter(
                x=closed["hold_min"],
                y=closed["pnl"],
                mode="markers",
                marker=dict(color=colors, size=8),
                name="Trades",
                hovertemplate="Hold %{x:.1f} min<br>PnL $%{y:.1f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_color="#777")
    fig.update_layout(
        template="plotly_white",
        title="Hold time vs PnL",
        xaxis_title="Hold time (minutes)",
        yaxis_title="PnL (USD)",
        height=450,
    )
    if out_path:
        _save_html(fig, out_path)
    return fig


def plot_rolling_winrate_html(
    closed: pd.DataFrame,
    window: int = 20,
    out_path: Path | str | None = None,
) -> Any:
    """Interactive rolling win rate."""
    if not _PLOTLY:
        raise ImportError("plotly is required")
    roll = rolling_win_rate(closed, window=window)
    fig = go.Figure()
    if roll.empty:
        fig.add_annotation(
            text="No closed trades", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=roll.index,
                y=roll.values,
                mode="lines",
                name=f"{window}-trade win rate",
                line=dict(color="#1565c0", width=2),
            )
        )
        fig.add_hline(y=50, line_color="#777", line_dash="dot", annotation_text="50%")
    fig.update_layout(
        template="plotly_white",
        title=f"Rolling {window}-trade win rate",
        xaxis_title="Trade number",
        yaxis_title="Win rate (%)",
        yaxis_range=[0, 100],
        height=400,
    )
    if out_path:
        _save_html(fig, out_path)
    return fig


def plot_tod_heatmap_html(closed: pd.DataFrame, out_path: Path | str | None = None) -> Any:
    """Interactive weekday × hour PnL heatmap."""
    if not _PLOTLY:
        raise ImportError("plotly is required")
    matrix, weekdays, hours = weekday_hour_pnl(closed)
    fig = go.Figure()
    if closed.empty or len(hours) == 0:
        fig.add_annotation(
            text="No closed trades", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    else:
        text = [[f"{v:.0f}" if np.isfinite(v) else "" for v in row] for row in matrix]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[f"{h:02d}" for h in hours],
                y=weekdays,
                colorscale="RdYlGn",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                colorbar=dict(title="Sum PnL (USD)"),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="Weekday × hour PnL",
        xaxis_title="Hour of entry",
        height=420,
    )
    if out_path:
        _save_html(fig, out_path)
    return fig


def write_trade_diagnostics(
    split_dir: Path,
    trades: pd.DataFrame,
    bars: pd.DataFrame | None,
    *,
    dpi: int = 150,
    save_plots: bool = True,
    save_html: bool = True,
    lots: float = 1.0,
    contract_size: float = 1.0,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
) -> None:
    """Write hold-time, win-rate, heatmap, and MAE/MFE charts into *split_dir*."""
    closed = closed_trades_table(trades)
    if save_plots:
        plot_hold_time_pnl(closed, out_path=split_dir / "hold_time_pnl.png", dpi=dpi)
        plot_rolling_winrate(closed, out_path=split_dir / "rolling_winrate.png", dpi=dpi)
        plot_tod_heatmap(closed, out_path=split_dir / "tod_heatmap.png", dpi=dpi)
        if bars is not None and not bars.empty:
            mae = mae_mfe_table(
                bars,
                trades,
                lots=lots,
                contract_size=contract_size,
                max_loss_per_trade_usd=max_loss_per_trade_usd,
                take_profit_per_trade_usd=take_profit_per_trade_usd,
            )
            plot_mae_mfe(mae, out_path=split_dir / "mae_mfe.png", dpi=dpi)
    if save_html:
        try:
            plot_hold_time_pnl_html(closed, out_path=split_dir / "hold_time_pnl.html")
            plot_rolling_winrate_html(closed, out_path=split_dir / "rolling_winrate.html")
            plot_tod_heatmap_html(closed, out_path=split_dir / "tod_heatmap.html")
            if bars is not None and not bars.empty:
                mae = mae_mfe_table(
                    bars,
                    trades,
                    lots=lots,
                    contract_size=contract_size,
                    max_loss_per_trade_usd=max_loss_per_trade_usd,
                    take_profit_per_trade_usd=take_profit_per_trade_usd,
                )
                plot_mae_mfe_html(mae, out_path=split_dir / "mae_mfe.html")
        except ImportError:
            pass
