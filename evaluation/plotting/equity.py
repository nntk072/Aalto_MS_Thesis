from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

from .common import (
    BREACH_COLOR,
    EQUITY_COLOR,
    LONG_COLOR,
    PEAK_COLOR,
    SHORT_COLOR,
    _apply_style,
    _save,
)
from .order_window import _extend_trade_window as _extend_trade_window
from .order_window import _extract_window as _extract_window
from .order_window import _structure_anchors as _structure_anchors
from .orders import _per_trade_equity
from .plot_series import (
    apply_mpl_date_axis,
    daily_drawdown_usd,
    daily_loss_limit_series,
    drawdown_ylim,
    max_drawdown_pct,
)


def plot_equity_curve(
    equity: pd.Series,
    breaches: list[str] | None = None,
    breach_events: list[dict[str, Any]] | None = None,
    initial_balance: float = 100_000.0,
    daily_loss_limit: float | None = None,
    max_loss_limit: float | None = None,
    profit_target: float | None = None,
    trades: pd.DataFrame | None = None,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Equity over time with peak line, FTMO threshold lines, and breach markers.

    Uses ``breach_events`` (list of dicts with 'time' key) for accurate vertical
    lines.  Falls back to legacy ``breaches`` count only when no events available.

    When ``trades`` is given the x-axis is the trade order (each trade's close
    time) so the chart reads as equity per order; otherwise it is a date axis.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    per_trade = False
    times: Any = None
    x: Any = None
    peak: Any = None
    if trades is not None and not trades.empty and "time" in trades.columns:
        pairs = _pair_trades(trades)
        if pairs:
            times, eq_vals = _per_trade_equity(pairs, equity)
            per_trade = True
            x = np.arange(len(pairs), dtype=float)
            ax.plot(x, eq_vals, color=EQUITY_COLOR, label="Equity")
            peak = np.maximum.accumulate(eq_vals)
            ax.plot(
                x,
                peak,
                color=PEAK_COLOR,
                alpha=0.5,
                linewidth=0.8,
                label="Peak equity",
            )
            step = max(1, len(x) // 12)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([t.strftime("%m/%d %H:%M") for t in times[::step]], rotation=45)
            ax.set_xlabel("Trade order")
    if not per_trade:
        ax.plot(equity.index, np.asarray(equity.values), color=EQUITY_COLOR, label="Equity")
        peak = equity.cummax()
        ax.plot(
            equity.index,
            np.asarray(peak.values),
            color=PEAK_COLOR,
            alpha=0.5,
            linewidth=0.8,
            label="Peak equity",
        )
        ax.set_xlabel("Date")

    if daily_loss_limit is not None:
        if per_trade:
            eq_trade = pd.Series(eq_vals, index=times)
            limit_s = daily_loss_limit_series(eq_trade, daily_loss_limit)
            ax.plot(
                x,
                np.asarray(limit_s.values),
                color="#ff9800",
                linewidth=1.1,
                linestyle=":",
                drawstyle="steps-post",
                label=f"Daily loss limit (${daily_loss_limit:,.0f} from day open)",
            )
        else:
            limit_s = daily_loss_limit_series(equity, daily_loss_limit)
            ax.plot(
                limit_s.index,
                np.asarray(limit_s.values),
                color="#ff9800",
                linewidth=1.1,
                linestyle=":",
                drawstyle="steps-post",
                label=f"Daily loss limit (${daily_loss_limit:,.0f} from day open)",
            )
    if max_loss_limit is not None:
        ax.axhline(
            initial_balance - max_loss_limit,
            color=BREACH_COLOR,
            linewidth=0.9,
            linestyle=":",
            label=f"Max loss limit (${max_loss_limit:,.0f})",
        )
    if profit_target is not None:
        ax.axhline(
            initial_balance + profit_target,
            color=LONG_COLOR,
            linewidth=1.0,
            linestyle="--",
            label=f"Profit target (${initial_balance + profit_target:,.0f})",
        )

    # Payout resets are economic withdrawals, so mark them explicitly on the
    # operating-equity curve instead of letting the reset look like an unexplained loss.
    if trades is not None and not trades.empty and "type" in trades.columns:
        payout_rows = trades[trades["type"].astype(str).eq("payout_reset")]
        if not payout_rows.empty and "time" in payout_rows.columns:
            payout_times = pd.to_datetime(payout_rows["time"], errors="coerce").dropna()
            for i, payout_time in enumerate(payout_times):
                date_number = mdates.date2num(pd.Timestamp(payout_time).to_pydatetime())
                x_position = float(date_number)
                if per_trade and times is not None and len(times):
                    time_index = pd.DatetimeIndex(times)
                    time_ns = np.asarray(time_index).astype(np.int64, copy=False)
                    position = int(np.searchsorted(time_ns, pd.Timestamp(payout_time).value))
                    x_position = float(min(max(position, 0), len(time_index) - 1))
                ax.axvline(
                    x_position,
                    color=LONG_COLOR,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.7,
                    label="Payout reset" if i == 0 else None,
                )

    # User preference: do not draw breach vertical lines on equity chart.

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    if not per_trade:
        apply_mpl_date_axis(ax, tz=pd.DatetimeIndex(equity.index).tz)
        fig.autofmt_xdate()
    ax.set_title("Equity Curve", fontweight="bold")
    ax.set_ylabel("Balance (USD)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _draw_dd_panel(
    ax: Any,
    series: pd.Series,
    *,
    title: str,
    floor: float,
    color: str,
    ylabel: str,
    y_formatter: Any,
    constraint: float | None = None,
    constraint_label: str | None = None,
    y_locator: Any | None = None,
) -> None:
    ax.fill_between(series.index, series.to_numpy(), 0, color=color, alpha=0.55, label=title)
    ax.plot(series.index, series.to_numpy(), color=color, linewidth=0.8)
    if constraint is not None:
        ax.axhline(
            constraint,
            color="#ff9800",
            linewidth=0.9,
            linestyle=":",
            label=constraint_label,
        )
    ax.yaxis.set_major_formatter(y_formatter)
    if y_locator is not None:
        ax.yaxis.set_major_locator(y_locator)
    apply_mpl_date_axis(ax, tz=pd.DatetimeIndex(series.index).tz)
    ax.set_ylim(*drawdown_ylim(series, floor))
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True)


def plot_drawdown(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
    daily_loss_limit: float | None = 5_000.0,
) -> Figure:
    """Two-panel chart: peak-to-trough DD (%) and daily DD (USD).

    Daily constraint is the FTMO **$5,000** cap (independent of equity).
    Trailing 7% DD is a training policy only — it is not drawn.
    """
    _apply_style()
    max_dd = max_drawdown_pct(equity)
    day_dd = daily_drawdown_usd(equity)
    tz = pd.DatetimeIndex(equity.index).tz
    daily_floor = -float(daily_loss_limit) if daily_loss_limit else 0.0
    max_floor = float(np.nanmin(np.asarray(max_dd, dtype=float))) if len(max_dd) else -1.0
    if np.isnan(max_floor):
        max_floor = -1.0

    fig, (ax_max, ax_day) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    _draw_dd_panel(
        ax_max,
        max_dd,
        title="Maximum drawdown",
        floor=min(max_floor, -1.0),
        color=SHORT_COLOR,
        ylabel="Drawdown (%)",
        y_formatter=mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"),
        y_locator=mticker.MultipleLocator(5.0),
    )
    _draw_dd_panel(
        ax_day,
        day_dd,
        title="Daily drawdown",
        floor=daily_floor
        if daily_loss_limit
        else (float(np.nanmin(day_dd.to_numpy())) if len(day_dd) else 0.0),
        color="#ff9800",
        ylabel="Drawdown (USD)",
        y_formatter=mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"),
        constraint=(-float(daily_loss_limit) if daily_loss_limit else None),
        constraint_label=(
            f"Daily loss cap (−${daily_loss_limit:,.0f})" if daily_loss_limit else None
        ),
    )
    ax_day.set_xlabel("Date")
    apply_mpl_date_axis(ax_day, tz=tz)
    fig.autofmt_xdate()
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig
