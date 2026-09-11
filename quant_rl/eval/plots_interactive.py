"""Interactive Plotly charts for backtest/eval visualization.

All functions save a self-contained HTML file and return the Figure object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from .plot_series import (
    daily_drawdown_pct,
    daily_loss_limit_series,
    daily_pnl,
    drawdown_ylim,
    max_drawdown_pct,
)

log = logging.getLogger(__name__)

_TEMPLATE = "plotly_white"
_LONG_COLOR = "#26a69a"
_SHORT_COLOR = "#ef5350"
_CLOSE_COLOR = "#ffd54f"
_EQUITY_COLOR = "#42a5f5"
_PEAK_COLOR = "#90caf9"
_BREACH_COLOR = "#ff1744"


def _check() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive charts: pip install plotly")


def _save(fig: go.Figure, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# 1. Equity curve (interactive)
# ---------------------------------------------------------------------------
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
    from .plots import _pair_trades

    per_trade = False
    times: Any = None
    x: Any = None
    peak: Any = None
    if trades is not None and not trades.empty and "time" in trades.columns:
        pairs = _pair_trades(trades)
        if pairs:
            times = pd.to_datetime([c["time"] for _, c in pairs])
            eq_at = equity.reindex(times).ffill().bfill()
            eq_vals = np.asarray(eq_at.to_numpy(dtype=float))
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
        limit_s = daily_loss_limit_series(equity, daily_loss_limit)
        if per_trade:
            limit_s = limit_s.reindex(times).ffill().bfill()
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


# ---------------------------------------------------------------------------
# 2. Drawdown (interactive)
# ---------------------------------------------------------------------------
def plot_drawdown(
    equity: pd.Series,
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    from plotly.subplots import make_subplots

    max_dd = max_drawdown_pct(equity)
    day_dd = daily_drawdown_pct(equity)
    max_lo, _ = drawdown_ylim(max_dd, -10.0)
    day_lo, _ = drawdown_ylim(day_dd, -5.0)

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
    fig.add_hline(y=-10.0, line_color="#777", line_dash="dot", row=1, col=1)
    fig.add_hline(y=-5.0, line_color="#777", line_dash="dot", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", range=[max_lo, 0], row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", range=[day_lo, 0], row=2, col=1)
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


# ---------------------------------------------------------------------------
# 3. Price + candlestick + order markers (interactive)
# ---------------------------------------------------------------------------
def plot_price_with_orders(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    max_points: int = 3000,
    candle_tf: str = "15min",
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    ohlcv = bars[["open", "high", "low", "close"]].copy()
    ohlcv = (
        ohlcv.resample(candle_tf)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna()
    )
    if len(ohlcv) > max_points:
        ohlcv = ohlcv.iloc[-max_points:]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="Price",
            increasing_line_color=_LONG_COLOR,
            decreasing_line_color=_SHORT_COLOR,
        )
    )

    if trades is not None and not trades.empty and "time" in trades.columns:
        opens = trades[trades["type"] == "open"]
        normal_closes = trades[trades["type"].isin(["close", "eod_close"])]
        forced_closes = trades[trades["type"].isin(["forced_close", "stop_close"])]

        if "direction" in opens.columns:
            long_opens = opens[opens["direction"] == 1]
            short_opens = opens[opens["direction"] == -1]
        else:
            long_opens = short_opens = pd.DataFrame()

        min_ts = ohlcv.index.min()
        max_ts = ohlcv.index.max()

        def _bin_to_candle(times: pd.Series) -> pd.Index:
            """Floor times to candle_tf and skip markers outside the plotted window."""
            binned = pd.DatetimeIndex(times).floor(candle_tf)
            result = []
            for ts in binned:
                if ts < min_ts or ts > max_ts:
                    continue
                if ts in ohlcv.index:
                    result.append(ts)
                else:
                    loc = int(ohlcv.index.searchsorted(ts, side="left"))
                    if loc >= len(ohlcv):
                        continue
                    candidate = pd.Timestamp(ohlcv.index[loc])
                    if candidate < min_ts or candidate > max_ts:
                        continue
                    result.append(candidate)
            return pd.DatetimeIndex(result)

        def _snap_df(
            df: pd.DataFrame, prefer_trade_price: bool = False
        ) -> tuple[pd.Index, np.ndarray[Any, Any]]:
            """Return aligned (x, y) arrays for markers within plotted candle range."""
            if df.empty:
                return pd.DatetimeIndex([]), np.array([])

            snapped_times: list[pd.Timestamp] = []
            y_vals: list[float] = []

            for _, row in df.iterrows():
                ts = pd.Timestamp(row["time"])
                snapped = _bin_to_candle(pd.Series([ts]))
                if len(snapped) == 0:
                    continue
                s_ts = snapped[0]
                snapped_times.append(s_ts)

                if prefer_trade_price and "price" in df.columns and pd.notna(row.get("price")):
                    y_vals.append(float(row["price"]))
                else:
                    y_vals.append(float(cast(Any, ohlcv["close"].at[s_ts])))

            return pd.DatetimeIndex(snapped_times), np.asarray(y_vals, dtype=float)

        if not long_opens.empty:
            snapped_x, snapped_y = _snap_df(long_opens, prefer_trade_price=True)
            if len(snapped_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=snapped_x,
                        y=snapped_y,
                        mode="markers",
                        name="Long open",
                        marker=dict(symbol="triangle-up", size=12, color=_LONG_COLOR, opacity=0.9),
                    )
                )
        if not short_opens.empty:
            snapped_x, snapped_y = _snap_df(short_opens, prefer_trade_price=True)
            if len(snapped_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=snapped_x,
                        y=snapped_y,
                        mode="markers",
                        name="Short open",
                        marker=dict(
                            symbol="triangle-down", size=12, color=_SHORT_COLOR, opacity=0.9
                        ),
                    )
                )
        if not normal_closes.empty:
            snapped_x, snapped_y = _snap_df(normal_closes, prefer_trade_price=False)
            if len(snapped_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=snapped_x,
                        y=snapped_y,
                        mode="markers",
                        name="Close",
                        marker=dict(symbol="x", size=9, color=_CLOSE_COLOR, opacity=0.85),
                    )
                )
        if not forced_closes.empty:
            snapped_x, snapped_y = _snap_df(forced_closes, prefer_trade_price=False)
            if len(snapped_x) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=snapped_x,
                        y=snapped_y,
                        mode="markers",
                        name="Forced/Stop close",
                        marker=dict(
                            symbol="x-open",
                            size=14,
                            color=_BREACH_COLOR,
                            line=dict(width=2, color=_BREACH_COLOR),
                            opacity=1.0,
                        ),
                    )
                )

    fig.update_layout(
        template=_TEMPLATE,
        title="Price + Orders",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=550,
    )

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Trade PnL histogram (interactive)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. Bar-return distribution (interactive)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 6. Monthly returns heatmap (interactive)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 7. Strategy comparison (interactive)
# ---------------------------------------------------------------------------
def plot_baseline_comparison(
    equity_dict: dict[str, pd.Series],
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    colors = [_EQUITY_COLOR, _LONG_COLOR, _SHORT_COLOR, _CLOSE_COLOR, "#ce93d8", "#80cbc4"]

    fig = go.Figure()
    for (name, eq), col in zip(equity_dict.items(), colors):
        normed = (eq / eq.iloc[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=normed.index,
                y=normed.values,
                mode="lines",
                name=name,
                line=dict(color=col, width=1.5),
            )
        )

    fig.add_hline(y=0, line_color="#555", line_dash="dot")
    fig.update_layout(
        template=_TEMPLATE,
        title="Strategy Comparison — Normalized Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        height=450,
    )

    if out_path:
        _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Per-trade M1 candlestick charts (one HTML per trade)
# ---------------------------------------------------------------------------


def plot_per_trade_orders(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    orders_dir: Path | str,
    context_bars: int = 60,
    max_charts: int = 200,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    lots: float = 1.0,
    contract_size: float = 1.0,
    show_mae_mfe: bool = True,
    show_sl_tp: bool = True,
    secondary_bars: pd.DataFrame | None = None,
) -> None:
    """Generate one M1 candlestick HTML per trade in *orders_dir* with MT5-style overlays.

    Filenames match the PNG counterpart produced by ``plots.plot_per_trade_orders``::

        trade_NNNN_YYYYMMDD_HHMMopen_HHMMclose_{L|S}_{p|m}PnL.html

    Parameters
    ----------
    bars : M1 price bars (DatetimeIndex, open/high/low/close columns).
    trades : Trade log with type/direction/price/time/pnl columns.
    orders_dir : Destination folder; created if absent.
    context_bars : M1 bars to show before entry and after exit.
    max_charts : Cap on number of charts; trades sampled evenly when over limit.
    max_loss_per_trade_usd : Maximum loss limit for SL calculation.
    take_profit_per_trade_usd : Take profit limit for TP calculation.
    lots : Position size in lots.
    contract_size : Contract size.
    show_mae_mfe : Whether to plot MAE/MFE lines.
    show_sl_tp : Whether to plot SL/TP lines.
    secondary_bars : Optional US500 (or correlated) bars for SMT lines.
    """
    _check()
    from plotly.subplots import make_subplots

    from .chart_indicators import compute_chart_overlays_full, slice_overlays
    from .chart_overlays import (
        VWAP_COLOR,
        build_overlay_events,
        draw_macd_rsi_plotly,
        draw_overlays_plotly,
    )
    from .plots import _extract_window, _pair_trades, _trade_filename
    from .trade_metrics import compute_trade_metrics

    orders_dir = Path(orders_dir)
    orders_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pair_trades(trades)
    if not pairs:
        return

    total = len(pairs)
    if total > max_charts:
        indices = np.linspace(0, total - 1, max_charts, dtype=int).tolist()
        pairs = [pairs[i] for i in indices]
        log.info("Per-trade HTML: sampling %d/%d trades → %s", max_charts, total, orders_dir)
    else:
        log.info("Per-trade HTML: %d charts → %s", total, orders_dir)

    # Compute EMA50/MACD/RSI/VWAP once on the full history so indicators are
    # warmed up and match the strategy's real signals, then slice per trade.
    full_overlays = compute_chart_overlays_full(bars)
    overlay_events = build_overlay_events(bars, secondary=secondary_bars)

    for seq_i, (open_row, close_row) in enumerate(pairs):
        t_open = pd.Timestamp(open_row["time"])
        t_close = pd.Timestamp(close_row["time"])
        window = _extract_window(bars, t_open, t_close, context_bars)
        if len(window) < 3:
            continue

        direction = int(open_row["direction"]) if pd.notna(open_row.get("direction")) else 0
        pnl = float(close_row["pnl"]) if pd.notna(close_row.get("pnl")) else 0.0
        close_type = str(close_row["type"])

        # Compute MAE/MFE/SL/TP metrics
        metrics = compute_trade_metrics(
            bars,
            open_row,
            close_row,
            max_loss_per_trade_usd=max_loss_per_trade_usd,
            take_profit_per_trade_usd=take_profit_per_trade_usd,
            lots=lots,
            contract_size=contract_size,
        )

        # Compute chart overlays (EMA50, MACD, RSI, VWAP), sliced from full-history calc
        overlays = slice_overlays(full_overlays, window.index)

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        )

        # Top panel: candlestick
        fig.add_trace(
            go.Candlestick(
                x=window.index,
                open=window["open"],
                high=window["high"],
                low=window["low"],
                close=window["close"],
                name="Price",
                increasing_line_color=_LONG_COLOR,
                decreasing_line_color=_SHORT_COLOR,
            ),
            row=1,
            col=1,
        )

        # Top panel: EMA50 overlay
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=overlays["ema50"],
                name="EMA50",
                line=dict(color="#0066cc", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=overlays["vwap"],
                name="VWAP",
                line=dict(color=VWAP_COLOR, width=1.8, dash="dashdot"),
            ),
            row=1,
            col=1,
        )
        events = overlay_events.clip(pd.Timestamp(window.index[0]), pd.Timestamp(window.index[-1]))
        draw_overlays_plotly(fig, events, row=1, col=1)

        # Entry marker: green arrow (triangle-up for long, triangle-down for short)
        i_o = int(window.index.get_indexer(pd.Index([t_open]), method="nearest")[0])
        if 0 <= i_o < len(window):
            ep = (
                float(open_row["price"])
                if pd.notna(open_row.get("price"))
                else float(window["close"].iloc[i_o])
            )
            fig.add_trace(
                go.Scatter(
                    x=[window.index[i_o]],
                    y=[ep],
                    mode="markers",
                    name="Entry",
                    showlegend=True,
                    marker=dict(
                        symbol="triangle-up" if direction == 1 else "triangle-down",
                        size=14,
                        color="#00cc00",  # bright green
                    ),
                ),
                row=1,
                col=1,
            )

        # Exit marker: red arrow (triangle-down for long, triangle-up for short)
        i_c = int(window.index.get_indexer(pd.Index([t_close]), method="nearest")[0])
        if 0 <= i_c < len(window):
            ep2 = (
                float(close_row["price"])
                if pd.notna(close_row.get("price"))
                else float(window["close"].iloc[i_c])
            )
            fig.add_trace(
                go.Scatter(
                    x=[window.index[i_c]],
                    y=[ep2],
                    mode="markers",
                    name="Exit",
                    showlegend=True,
                    marker=dict(
                        symbol="triangle-down" if direction == 1 else "triangle-up",
                        size=14,
                        color="#ff0000",  # bright red
                    ),
                ),
                row=1,
                col=1,
            )

        # Add MAE/MFE horizontal lines
        if show_mae_mfe:
            # MAE line (red dashed)
            fig.add_hline(
                y=metrics.mae_price,
                line_color="#ff6666",
                line_dash="dash",
                annotation_text="MAE",
                annotation_position="right",
                row=1,
                col=1,
            )
            # MFE line (green dashed)
            fig.add_hline(
                y=metrics.mfe_price,
                line_color="#66ff66",
                line_dash="dash",
                annotation_text="MFE",
                annotation_position="right",
                row=1,
                col=1,
            )

        # Add SL/TP horizontal lines (if configured)
        if show_sl_tp:
            if metrics.sl_price is not None:
                fig.add_hline(
                    y=metrics.sl_price,
                    line_color="#ff0000",
                    line_dash="dot",
                    annotation_text="SL",
                    annotation_position="right",
                    row=1,
                    col=1,
                )
            if metrics.tp_price is not None:
                fig.add_hline(
                    y=metrics.tp_price,
                    line_color="#00cc00",
                    line_dash="dot",
                    annotation_text="TP",
                    annotation_position="right",
                    row=1,
                    col=1,
                )

        # Bottom panel: MACD (left) + RSI (right)
        draw_macd_rsi_plotly(fig, window, overlays, row=2)

        close_reason = close_type if close_type != "close" else "normal"
        dir_label = "Long" if direction == 1 else "Short"

        # Calculate trade info
        duration_mins = int((t_close - t_open).total_seconds() / 60)
        duration_secs = int((t_close - t_open).total_seconds() % 60)
        lots_val = float(open_row.get("lots", 1.0)) if pd.notna(open_row.get("lots")) else 1.0
        volume = lots_val  # Volume in lots

        # Reconciliation: recompute PnL from the entry/exit prices shown on
        # this chart and compare to the logged PnL. A mismatch flags a
        # pairing bug or a stale/fallback exit price rather than a "weird"
        # strategy result.
        pnl_calc = (metrics.exit_price - metrics.entry_price) * direction * lots_val * contract_size
        close_reason_detail = (
            str(close_row.get("reason")) if pd.notna(close_row.get("reason")) else close_type
        )

        # Create extended title with trade info
        title = (
            f"{dir_label} | Open {t_open.strftime('%Y-%m-%d %H:%M')} "
            f"→ Close {t_close.strftime('%H:%M')} | "
            f"PnL: {pnl:+.2f} | {close_reason}<br>"
            f"<sub>Direction: {'Buy' if direction == 1 else 'Sell'} | "
            f"Open: {metrics.entry_price:.2f} | Close: {metrics.exit_price:.2f} | "
            f"Volume: {volume:.2f} | Duration: {duration_mins}m{duration_secs}s<br>"
            f"PnL (logged): {pnl:+.2f} | PnL (calc): {pnl_calc:+.2f} | "
            f"Reason: {close_reason_detail}</sub>"
        )

        fig.update_layout(
            template=_TEMPLATE,
            title=title,
            xaxis2_title="Time (M1)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=680,
        )

        fname = _trade_filename(seq_i + 1, open_row, close_row, "html")
        try:
            _save(fig, orders_dir / fname)
        except Exception as exc:
            log.debug("Skipping per-trade HTML %d: %s", seq_i + 1, exc)


# ---------------------------------------------------------------------------
# 9. PO3 / IFVG signals (interactive)
# ---------------------------------------------------------------------------


def plot_fvg_signals_interactive(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    candle_tf: str = "5min",
    max_points: int = 3000,
    show_entries: bool = True,
    out_path: Path | str | None = None,
) -> go.Figure:
    """Interactive Plotly chart of PO3/IFVG zone rectangles + entry markers.

    Parameters
    ----------
    bars : pd.DataFrame
        M1 OHLC bars (DatetimeIndex) used to compute ``signals``.
    signals : pd.DataFrame
        Output of :func:`~quant_rl.features.po3_config.detect_po3_entries`
        (or any of its sub-detectors).  Shares the index of ``bars``.
    window : tuple[Timestamp, Timestamp], optional
        Slice of the data to display (inclusive).
    candle_tf : str
        Pandas resample rule for the candles, e.g. ``'1min'``, ``'5min'``.
    max_points : int
        Cap on candle count.
    show_entries : bool
        Draw entry_long/entry_short markers.
    out_path : Path | str, optional
        If given, a self-contained HTML file is written here.

    Returns
    -------
    go.Figure
        The interactive figure.
    """
    _check()
    from quant_rl.eval.po3_plots import _resample_ohlc
    from quant_rl.features.po3_config import build_fvg_zones

    if window is not None:
        start, end = window
        bars = bars.loc[start:end]
    if not len(bars):
        raise ValueError("No bars to plot after applying window.")

    zones = build_fvg_zones(bars, signals)
    ohlcv = _resample_ohlc(bars, candle_tf)
    if len(ohlcv) > max_points:
        ohlcv = ohlcv.iloc[-max_points:]
    if ohlcv.empty:
        raise ValueError("No candles to plot after resampling.")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["open"],
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            name="Price",
            increasing_line_color=_LONG_COLOR,
            decreasing_line_color=_SHORT_COLOR,
        )
    )

    # Zone rectangles.
    zone_colors = {
        ("htf_fvg", "bullish"): "rgba(27,122,61,0.25)",
        ("htf_fvg", "bearish"): "rgba(179,49,49,0.25)",
        ("ltf_ifvg", "bullish"): "rgba(129,212,168,0.35)",
        ("ltf_ifvg", "bearish"): "rgba(239,154,154,0.35)",
    }
    for z in zones:
        if z.end_ts < ohlcv.index.min() or z.start_ts > ohlcv.index.max():
            continue
        fig.add_shape(
            type="rect",
            x0=z.start_ts,
            x1=z.end_ts,
            y0=z.zone_low,
            y1=z.zone_high,
            fillcolor=zone_colors[(z.kind, z.side)],
            line=dict(width=0),
            layer="below",
        )

    # Entry markers.
    if show_entries:
        for col, marker, color, dy in (
            ("entry_long", "triangle-up", _LONG_COLOR, 1.004),
            ("entry_short", "triangle-down", _SHORT_COLOR, 0.996),
        ):
            if col not in signals.columns:
                continue
            hits = signals[col] == 1
            if not hits.any():
                continue
            xs = []
            ys = []
            for ts in signals.index[hits]:
                if ts < ohlcv.index.min() or ts > ohlcv.index.max():
                    continue
                loc = int(ohlcv.index.searchsorted(ts, side="left"))
                if loc >= len(ohlcv):
                    continue
                base = ohlcv["high"].iloc[loc] if dy > 1 else ohlcv["low"].iloc[loc]
                xs.append(ohlcv.index[loc])
                ys.append(base * dy)
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(symbol=marker, size=11, color=color),
                        name=col,
                    )
                )

    fig.update_layout(
        template=_TEMPLATE,
        title="PO3 (AMD) + HTF/LTF IFVG Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        height=650,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(rangeslider_visible=False)

    if out_path:
        _save(fig, out_path)
    return fig
