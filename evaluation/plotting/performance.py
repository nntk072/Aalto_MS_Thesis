from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

from .common import (
    EQUITY_COLOR,
    LONG_COLOR,
    SHORT_COLOR,
    _apply_style,
    _save,
)
from .order_window import _extend_trade_window as _extend_trade_window
from .order_window import _extract_window as _extract_window
from .order_window import _structure_anchors as _structure_anchors
from .plot_series import (
    apply_mpl_date_axis,
    daily_pnl,
)


def plot_daily_pnl(
    equity: pd.Series,
    daily_loss_limit: float | None = None,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Per-session P&L bars with the FTMO daily-loss dollar cap."""
    _apply_style()
    pnl = daily_pnl(equity)
    fig, ax = plt.subplots(figsize=(14, 4))
    if pnl.empty:
        ax.text(0.5, 0.5, "No daily P&L", ha="center", va="center", transform=ax.transAxes)
    else:
        colors = [LONG_COLOR if v >= 0 else SHORT_COLOR for v in pnl.to_numpy()]
        ax.bar(pnl.index, pnl.to_numpy(), color=colors, width=0.8, align="center")
        ax.axhline(0, color="#777777", linewidth=0.7)
        if daily_loss_limit is not None:
            ax.axhline(
                -float(daily_loss_limit),
                color="#ff9800",
                linewidth=0.9,
                linestyle=":",
                label=f"Daily loss cap (−${daily_loss_limit:,.0f})",
            )
            ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    apply_mpl_date_axis(ax, tz=pd.DatetimeIndex(equity.index).tz)
    fig.autofmt_xdate()
    ax.set_title("Daily P&L", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L (USD)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_trade_pnl_hist(
    trades: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Histogram of per-trade PnL with win / loss colour split."""
    _apply_style()
    if "pnl" not in trades.columns or trades["pnl"].dropna().empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No trade PnL data", ha="center", va="center", transform=ax.transAxes)
        if out_path:
            _save(fig, out_path, dpi)
        return fig

    pnl = trades["pnl"].dropna()
    wins = pnl[pnl >= 0]
    losses = pnl[pnl < 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = min(60, max(10, len(pnl) // 10))
    if not wins.empty:
        ax.hist(
            wins.to_numpy(), bins=bins, color=LONG_COLOR, alpha=0.8, label=f"Wins ({len(wins)})"
        )
    if not losses.empty:
        ax.hist(
            losses.to_numpy(),
            bins=bins,
            color=SHORT_COLOR,
            alpha=0.8,
            label=f"Losses ({len(losses)})",
        )
    ax.axvline(0, color="#777", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Trade PnL Distribution", fontweight="bold")
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_direction_summary(
    trades: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Plot completed-trade counts and PnL distributions by direction."""
    _apply_style()
    pairs = _pair_trades(trades)
    rows = [
        {
            "direction": "Long" if int(open_row.get("direction", 0)) == 1 else "Short",
            "pnl": float(close_row.get("pnl", 0.0) or 0.0),
        }
        for open_row, close_row in pairs
    ]
    closed = pd.DataFrame(rows)
    fig, (ax_count, ax_pnl) = plt.subplots(1, 2, figsize=(11, 4))
    if closed.empty:
        for ax in (ax_count, ax_pnl):
            ax.text(0.5, 0.5, "No completed trades", ha="center", va="center")
            ax.set_axis_off()
    else:
        order = ["Long", "Short"]
        counts = closed["direction"].value_counts().reindex(order, fill_value=0)
        sums = closed.groupby("direction")["pnl"].sum().reindex(order, fill_value=0.0)
        colors = [LONG_COLOR, SHORT_COLOR]
        ax_count.bar(order, counts.to_numpy(), color=colors)
        ax_count.set_title("Completed Orders by Direction", fontweight="bold")
        ax_count.set_ylabel("Trade count")
        for i, value in enumerate(counts):
            ax_count.text(i, float(value), str(int(value)), ha="center", va="bottom")
        for label, color in zip(order, colors, strict=True):
            values = closed.loc[closed["direction"] == label, "pnl"]
            if not values.empty:
                ax_pnl.hist(
                    values.to_numpy(),
                    bins=max(5, min(30, len(values))),
                    alpha=0.65,
                    color=color,
                    label=f"{label} (n={len(values)})",
                )
        ax_pnl.axvline(0.0, color="#555", linewidth=0.8)
        ax_pnl.set_title("PnL Distribution by Direction", fontweight="bold")
        ax_pnl.set_xlabel("PnL (USD)")
        ax_pnl.set_ylabel("Count")
        ax_pnl.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_pnl.legend(fontsize=8)
        ax_pnl.text(
            0.98,
            0.98,
            "\n".join(f"{label}: ${sums[label]:,.2f}" for label in order),
            transform=ax_pnl.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999"},
        )
    for ax in (ax_count, ax_pnl):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_returns_dist(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Bar-return distribution with normal overlay."""
    _apply_style()
    rets = equity.pct_change().dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = min(80, max(20, len(rets) // 50))
    ax.hist(
        rets.to_numpy() * 100,
        bins=bins,
        color=EQUITY_COLOR,
        alpha=0.75,
        density=True,
        label="Returns",
    )

    # Normal overlay
    mu, sigma = rets.mean() * 100, rets.std() * 100
    if sigma > 0:
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 300)
        try:
            from scipy.stats import norm

            ax.plot(x, norm.pdf(x, mu, sigma), color="#333", linewidth=1.0, label="Normal fit")
        except ImportError:
            pass
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_monthly_returns_heatmap(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Calendar heatmap of monthly returns."""
    _apply_style()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5,
            0.5,
            "Insufficient data for monthly heatmap",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        if out_path:
            _save(fig, out_path, dpi)
        return fig

    years = sorted(pd.DatetimeIndex(monthly.index).year.unique())
    months = list(range(1, 13))
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    matrix = np.full((len(years), 12), np.nan)
    for dt, val in monthly.items():
        dt_ts = cast(pd.Timestamp, dt)
        yi = years.index(dt_ts.year)
        mi = dt_ts.month - 1
        matrix[yi, mi] = val

    vabs = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(months)), max(3, len(years))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])

    for yi in range(len(years)):
        for mi in range(12):
            v = matrix[yi, mi]
            if not np.isnan(v):
                ax.text(
                    mi,
                    yi,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#000" if abs(v) < vabs * 0.6 else "#fff",
                )

    plt.colorbar(im, ax=ax, fraction=0.03, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap", fontweight="bold")
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig
