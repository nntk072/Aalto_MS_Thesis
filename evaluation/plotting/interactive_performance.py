from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .interactive_common import (
    _BREACH_COLOR,
    _EQUITY_COLOR,
    _LONG_COLOR,
    _PEAK_COLOR,
    _SHORT_COLOR,
    _TEMPLATE,
    _check,
    _save,
    go,
)
from .plot_series import (
    daily_drawdown_usd,
    daily_loss_limit_series,
    daily_pnl,
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
) -> go.Figure:
    _check()
    from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

    per_trade = False
    times: Any = None
    x: Any = None
    peak: Any = None
    if trades is not None and not trades.empty and "time" in trades.columns:
        pairs = _pair_trades(trades)
        if pairs:
            from .plots import _per_trade_equity

            times, eq_vals = _per_trade_equity(pairs, equity)
            x = np.arange(len(pairs))
            per_trade = True

    if per_trade:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=eq_vals,
                mode="lines",
                name="Equity",
                line=dict(color=_EQUITY_COLOR, width=1.5),
            )
        )
        peak = np.maximum.accumulate(eq_vals)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=peak,
                mode="lines",
                name="Peak Equity",
                line=dict(color=_PEAK_COLOR, width=0.8, dash="dot"),
                opacity=0.6,
            )
        )
        step = max(1, len(x) // 12)
        tick_vals = x[::step].tolist()
        tick_text = [t.strftime("%m/%d %H:%M") for t in times[::step]]
        x_tick_format = None
        x_title = "Trade order"
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Equity",
                line=dict(color=_EQUITY_COLOR, width=1.5),
            )
        )
        peak = equity.cummax()
        fig.add_trace(
            go.Scatter(
                x=peak.index,
                y=peak.values,
                mode="lines",
                name="Peak Equity",
                line=dict(color=_PEAK_COLOR, width=0.8, dash="dot"),
                opacity=0.6,
            )
        )
        tick_vals = None
        tick_text = None
        x_tick_format = "%Y-%m-%d"
        x_title = "Date"

    if daily_loss_limit is not None:
        if per_trade:
            eq_trade = pd.Series(eq_vals, index=times)
            limit_s = daily_loss_limit_series(eq_trade, daily_loss_limit)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=limit_s.values,
                    mode="lines",
                    name=f"Daily loss limit ${daily_loss_limit:,.0f} (from day open)",
                    line=dict(color="#ff9800", width=1.2, dash="dot"),
                    line_shape="hv",
                )
            )
        else:
            limit_s = daily_loss_limit_series(equity, daily_loss_limit)
            fig.add_trace(
                go.Scatter(
                    x=limit_s.index,
                    y=limit_s.values,
                    mode="lines",
                    name=f"Daily loss limit ${daily_loss_limit:,.0f} (from day open)",
                    line=dict(color="#ff9800", width=1.2, dash="dot"),
                    line_shape="hv",
                )
            )
    if max_loss_limit is not None:
        y_lim = initial_balance - max_loss_limit
        fig.add_hline(
            y=y_lim,
            line_color=_BREACH_COLOR,
            line_dash="dot",
            annotation_text=f"Max loss limit ${max_loss_limit:,.0f}",
            annotation_position="bottom right",
        )
    if profit_target is not None:
        fig.add_hline(
            y=initial_balance + profit_target,
            line_color=_LONG_COLOR,
            line_dash="dash",
            annotation_text=f"Profit target ${initial_balance + profit_target:,.0f}",
            annotation_position="top right",
        )

    if trades is not None and not trades.empty and "type" in trades.columns:
        payout_rows = trades[trades["type"].astype(str).eq("payout_reset")]
        if not payout_rows.empty and "time" in payout_rows.columns:
            payout_times = pd.to_datetime(payout_rows["time"], errors="coerce").dropna()
            for payout_time in payout_times:
                x_position: int | pd.Timestamp = payout_time
                if per_trade and times is not None and len(times):
                    time_index = pd.DatetimeIndex(times)
                    time_ns = np.asarray(time_index).astype(np.int64, copy=False)
                    position = int(np.searchsorted(time_ns, pd.Timestamp(payout_time).value))
                    x_position = min(max(position, 0), len(time_index) - 1)
                fig.add_vline(
                    x=x_position,
                    line_color=_LONG_COLOR,
                    line_dash="dash",
                    line_width=1,
                    opacity=0.7,
                )

    # User preference: do not draw breach vertical lines on equity chart.

    fig.update_layout(
        template=_TEMPLATE,
        title="Equity Curve",
        xaxis_title=x_title,
        yaxis_title="Balance (USD)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        xaxis_tickformat=x_tick_format,
        xaxis_tickvals=tick_vals,
        xaxis_ticktext=tick_text,
        hovermode="x unified",
        height=450,
    )

    if out_path:
        _save(fig, out_path)
    return fig


def plot_drawdown(
    equity: pd.Series,
    out_path: Path | str | None = None,
    daily_loss_limit: float | None = 5_000.0,
) -> go.Figure:
    _check()
    from plotly.subplots import make_subplots

    max_dd = max_drawdown_pct(equity)
    day_dd = daily_drawdown_usd(equity)
    max_floor = float(np.nanmin(np.asarray(max_dd, dtype=float))) if len(max_dd) else -1.0
    if np.isnan(max_floor):
        max_floor = -1.0
    max_lo, _ = drawdown_ylim(max_dd, min(max_floor, -1.0))
    daily_floor = -float(daily_loss_limit) if daily_loss_limit else 0.0
    day_lo, _ = drawdown_ylim(day_dd, daily_floor)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Maximum drawdown", "Daily drawdown"),
    )
    fig.add_trace(
        go.Scatter(
            x=max_dd.index,
            y=max_dd.values,
            mode="lines",
            name="Max drawdown",
            fill="tozeroy",
            line=dict(color=_SHORT_COLOR, width=0.8),
            fillcolor="rgba(239,83,80,0.25)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day_dd.index,
            y=day_dd.values,
            mode="lines",
            name="Daily drawdown",
            fill="tozeroy",
            line=dict(color="#ff9800", width=0.8),
            fillcolor="rgba(255,152,0,0.25)",
        ),
        row=2,
        col=1,
    )
    # Daily FTMO cap is $5,000. Trailing 7% is training-only — not plotted.
    if daily_loss_limit:
        fig.add_hline(
            y=-float(daily_loss_limit),
            line_color="#ff9800",
            line_dash="dot",
            annotation_text=f"Daily loss cap −${daily_loss_limit:,.0f}",
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", range=[max_lo, 0], row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (USD)", tickprefix="$", range=[day_lo, 0], row=2, col=1)
    fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d", row=2, col=1)
    fig.update_xaxes(tickformat="%Y-%m-%d", row=1, col=1)
    fig.update_layout(
        template=_TEMPLATE,
        title="Drawdown",
        hovermode="x unified",
        height=520,
    )

    if out_path:
        _save(fig, out_path)
    return fig


def plot_daily_pnl(
    equity: pd.Series,
    daily_loss_limit: float | None = None,
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    pnl = daily_pnl(equity)
    fig = go.Figure()
    if pnl.empty:
        fig.add_annotation(
            text="No daily P&L", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    else:
        colors = [_LONG_COLOR if v >= 0 else _SHORT_COLOR for v in pnl.to_numpy()]
        fig.add_trace(go.Bar(x=pnl.index, y=pnl.values, marker_color=colors, name="Daily P&L"))
        if daily_loss_limit is not None:
            fig.add_hline(
                y=-float(daily_loss_limit),
                line_color="#ff9800",
                line_dash="dot",
                annotation_text=f"Daily loss cap −${daily_loss_limit:,.0f}",
            )
    fig.update_layout(
        template=_TEMPLATE,
        title="Daily P&L",
        xaxis_title="Date",
        yaxis_title="P&L (USD)",
        yaxis_tickprefix="$",
        xaxis_tickformat="%Y-%m-%d",
        height=400,
    )
    if out_path:
        _save(fig, out_path)
    return fig


def plot_trade_pnl_hist(
    trades: pd.DataFrame,
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    if "pnl" not in trades.columns or trades["pnl"].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No trade PnL data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        if out_path:
            _save(fig, out_path)
        return fig

    pnl = trades["pnl"].dropna()
    wins = pnl[pnl >= 0]
    losses = pnl[pnl < 0]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(
            go.Histogram(
                x=wins.values,
                name=f"Wins ({len(wins)})",
                marker_color=_LONG_COLOR,
                opacity=0.8,
                nbinsx=50,
            )
        )
    if not losses.empty:
        fig.add_trace(
            go.Histogram(
                x=losses.values,
                name=f"Losses ({len(losses)})",
                marker_color=_SHORT_COLOR,
                opacity=0.8,
                nbinsx=50,
            )
        )

    fig.add_vline(x=0, line_color="#fff", line_width=0.8)
    fig.update_layout(
        template=_TEMPLATE,
        title="Trade PnL Distribution",
        xaxis_title="PnL (USD)",
        yaxis_title="Count",
        barmode="overlay",
        hovermode="x",
        height=400,
    )

    if out_path:
        _save(fig, out_path)
    return fig


def plot_returns_dist(
    equity: pd.Series,
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    rets = equity.pct_change().dropna() * 100

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=rets.values,
            name="Returns",
            marker_color=_EQUITY_COLOR,
            opacity=0.75,
            nbinsx=80,
            histnorm="probability density",
        )
    )
    fig.add_vline(x=0, line_color="#ff9800", line_width=0.8)
    fig.update_layout(
        template=_TEMPLATE,
        title="Bar-Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Density",
        height=400,
    )

    if out_path:
        _save(fig, out_path)
    return fig


def plot_monthly_returns_heatmap(
    equity: pd.Series,
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        if out_path:
            _save(fig, out_path)
        return fig

    years = sorted(pd.DatetimeIndex(monthly.index).year.unique())
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
        matrix[years.index(dt_ts.year), dt_ts.month - 1] = val

    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in matrix]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=month_labels,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Return (%)"),
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        height=max(300, len(years) * 40 + 150),
    )

    if out_path:
        _save(fig, out_path)
    return fig
