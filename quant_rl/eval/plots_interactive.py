"""Interactive Plotly charts for backtest/eval visualization.

All functions save a self-contained HTML file and return the Figure object.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

log = logging.getLogger(__name__)

_TEMPLATE = "plotly_white"
_LONG_COLOR  = "#26a69a"
_SHORT_COLOR = "#ef5350"
_CLOSE_COLOR = "#ffd54f"
_EQUITY_COLOR = "#42a5f5"
_PEAK_COLOR   = "#90caf9"
_BREACH_COLOR = "#ff1744"


def _check() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive charts: pip install plotly")


def _save(fig: "go.Figure", path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# 1. Equity curve (interactive)
# ---------------------------------------------------------------------------
def plot_equity_curve(
    equity: pd.Series,
    breaches: list[str] | None = None,
    breach_events: list[dict] | None = None,
    initial_balance: float = 100_000.0,
    daily_loss_limit: float | None = None,
    max_loss_limit: float | None = None,
    out_path: Path | str | None = None,
) -> "go.Figure":
    _check()
    peak = equity.cummax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color=_EQUITY_COLOR, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=peak.index, y=peak.values,
        mode="lines", name="Peak Equity",
        line=dict(color=_PEAK_COLOR, width=0.8, dash="dot"),
        opacity=0.6,
    ))

    if daily_loss_limit is not None:
        y_lim = initial_balance - daily_loss_limit
        fig.add_hline(y=y_lim, line_color="#ff9800", line_dash="dot",
                      annotation_text=f"Daily loss limit ${daily_loss_limit:,.0f}",
                      annotation_position="bottom right")
    if max_loss_limit is not None:
        y_lim = initial_balance - max_loss_limit
        fig.add_hline(y=y_lim, line_color=_BREACH_COLOR, line_dash="dot",
                      annotation_text=f"Max loss limit ${max_loss_limit:,.0f}",
                      annotation_position="bottom right")

    # User preference: do not draw breach vertical lines on equity chart.

    fig.update_layout(
        template=_TEMPLATE,
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Balance (USD)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
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
) -> "go.Figure":
    _check()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        line=dict(color=_SHORT_COLOR, width=0.8),
        fillcolor="rgba(239,83,80,0.25)",
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        height=300,
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
) -> "go.Figure":
    _check()
    ohlcv = bars[["open", "high", "low", "close"]].copy()
    ohlcv = ohlcv.resample(candle_tf).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    if len(ohlcv) > max_points:
        ohlcv = ohlcv.iloc[-max_points:]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["open"], high=ohlcv["high"],
        low=ohlcv["low"],   close=ohlcv["close"],
        name="Price",
        increasing_line_color=_LONG_COLOR,
        decreasing_line_color=_SHORT_COLOR,
    ))

    if trades is not None and not trades.empty and "time" in trades.columns:
        opens  = trades[trades["type"] == "open"]
        normal_closes = trades[trades["type"].isin(["close", "eod_close"])]
        forced_closes = trades[trades["type"].isin(["forced_close", "stop_close"])]

        if "direction" in opens.columns:
            long_opens  = opens[opens["direction"] ==  1]
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
                    loc = ohlcv.index.searchsorted(ts, side="left")
                    if loc >= len(ohlcv):
                        continue
                    candidate = ohlcv.index[loc]
                    if candidate < min_ts or candidate > max_ts:
                        continue
                    result.append(candidate)
            return pd.DatetimeIndex(result)

        def _snap_df(df: pd.DataFrame, prefer_trade_price: bool = False) -> tuple[pd.Index, np.ndarray]:
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
                    y_vals.append(float(ohlcv.at[s_ts, "close"]))

            return pd.DatetimeIndex(snapped_times), np.asarray(y_vals, dtype=float)

        if not long_opens.empty:
            snapped_x, snapped_y = _snap_df(long_opens, prefer_trade_price=True)
            if len(snapped_x) > 0:
                fig.add_trace(go.Scatter(
                    x=snapped_x, y=snapped_y,
                    mode="markers", name="Long open",
                    marker=dict(symbol="triangle-up", size=12, color=_LONG_COLOR, opacity=0.9),
                ))
        if not short_opens.empty:
            snapped_x, snapped_y = _snap_df(short_opens, prefer_trade_price=True)
            if len(snapped_x) > 0:
                fig.add_trace(go.Scatter(
                    x=snapped_x, y=snapped_y,
                    mode="markers", name="Short open",
                    marker=dict(symbol="triangle-down", size=12, color=_SHORT_COLOR, opacity=0.9),
                ))
        if not normal_closes.empty:
            snapped_x, snapped_y = _snap_df(normal_closes, prefer_trade_price=False)
            if len(snapped_x) > 0:
                fig.add_trace(go.Scatter(
                    x=snapped_x, y=snapped_y,
                    mode="markers", name="Close",
                    marker=dict(symbol="x", size=9, color=_CLOSE_COLOR, opacity=0.85),
                ))
        if not forced_closes.empty:
            snapped_x, snapped_y = _snap_df(forced_closes, prefer_trade_price=False)
            if len(snapped_x) > 0:
                fig.add_trace(go.Scatter(
                    x=snapped_x, y=snapped_y,
                    mode="markers", name="Forced/Stop close",
                    marker=dict(symbol="x-open", size=14, color=_BREACH_COLOR,
                                line=dict(width=2, color=_BREACH_COLOR), opacity=1.0),
                ))

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
) -> "go.Figure":
    _check()
    if "pnl" not in trades.columns or trades["pnl"].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(text="No trade PnL data", xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)
        if out_path:
            _save(fig, out_path)
        return fig

    pnl = trades["pnl"].dropna()
    wins   = pnl[pnl >= 0]
    losses = pnl[pnl < 0]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins.values, name=f"Wins ({len(wins)})",
            marker_color=_LONG_COLOR, opacity=0.8, nbinsx=50,
        ))
    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses.values, name=f"Losses ({len(losses)})",
            marker_color=_SHORT_COLOR, opacity=0.8, nbinsx=50,
        ))

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
) -> "go.Figure":
    _check()
    rets = equity.pct_change().dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets.values, name="Returns",
        marker_color=_EQUITY_COLOR, opacity=0.75, nbinsx=80,
        histnorm="probability density",
    ))
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
) -> "go.Figure":
    _check()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)
        if out_path:
            _save(fig, out_path)
        return fig

    years  = sorted(monthly.index.year.unique())
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    matrix = np.full((len(years), 12), np.nan)
    for dt, val in monthly.items():
        matrix[years.index(dt.year), dt.month - 1] = val

    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=month_labels,
        y=[str(y) for y in years],
        colorscale="RdYlGn",
        zmid=0,
        text=text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Return (%)"),
    ))
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
) -> "go.Figure":
    _check()
    colors = [_EQUITY_COLOR, _LONG_COLOR, _SHORT_COLOR, _CLOSE_COLOR, "#ce93d8", "#80cbc4"]

    fig = go.Figure()
    for (name, eq), col in zip(equity_dict.items(), colors):
        normed = (eq / eq.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=normed.index, y=normed.values,
            mode="lines", name=name,
            line=dict(color=col, width=1.5),
        ))

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
    """
    _check()
    from .plots import _pair_trades, _extract_window, _trade_filename
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

    for seq_i, (open_row, close_row) in enumerate(pairs):
        t_open  = pd.Timestamp(open_row["time"])
        t_close = pd.Timestamp(close_row["time"])
        window  = _extract_window(bars, t_open, t_close, context_bars)
        if len(window) < 3:
            continue

        direction  = int(open_row["direction"]) if pd.notna(open_row.get("direction")) else 0
        pnl        = float(close_row["pnl"])    if pd.notna(close_row.get("pnl"))       else 0.0
        close_type = str(close_row["type"])

        # Compute MAE/MFE/SL/TP metrics
        metrics = compute_trade_metrics(
            bars, open_row, close_row,
            max_loss_per_trade_usd=max_loss_per_trade_usd,
            take_profit_per_trade_usd=take_profit_per_trade_usd,
            lots=lots,
            contract_size=contract_size,
        )

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=window.index,
            open=window["open"], high=window["high"],
            low=window["low"],   close=window["close"],
            name="Price",
            increasing_line_color=_LONG_COLOR,
            decreasing_line_color=_SHORT_COLOR,
        ))

        # Entry marker: green arrow (triangle-up for long, triangle-down for short)
        i_o = window.index.get_indexer([t_open], method="nearest")[0]
        if 0 <= i_o < len(window):
            ep = float(open_row["price"]) if pd.notna(open_row.get("price")) else float(window["close"].iloc[i_o])
            fig.add_trace(go.Scatter(
                x=[window.index[i_o]], y=[ep],
                mode="markers", name="Entry",
                marker=dict(
                    symbol="triangle-up" if direction == 1 else "triangle-down",
                    size=14,
                    color="#00cc00",  # bright green
                ),
            ))

        # Exit marker: red arrow (triangle-down for long, triangle-up for short)
        i_c = window.index.get_indexer([t_close], method="nearest")[0]
        if 0 <= i_c < len(window):
            ep2 = float(window["close"].iloc[i_c])
            fig.add_trace(go.Scatter(
                x=[window.index[i_c]], y=[ep2],
                mode="markers", name="Exit",
                marker=dict(
                    symbol="triangle-down" if direction == 1 else "triangle-up",
                    size=14,
                    color="#ff0000",  # bright red
                ),
            ))

        # Add MAE/MFE horizontal lines
        if show_mae_mfe:
            # MAE line (red dashed)
            fig.add_hline(y=metrics.mae_price, line_color="#ff6666", line_dash="dash",
                         annotation_text="MAE", annotation_position="right")
            # MFE line (green dashed)
            fig.add_hline(y=metrics.mfe_price, line_color="#66ff66", line_dash="dash",
                         annotation_text="MFE", annotation_position="right")

        # Add SL/TP horizontal lines (if configured)
        if show_sl_tp:
            if metrics.sl_price is not None:
                fig.add_hline(y=metrics.sl_price, line_color="#ff0000", line_dash="dot",
                             annotation_text="SL", annotation_position="right")
            if metrics.tp_price is not None:
                fig.add_hline(y=metrics.tp_price, line_color="#00cc00", line_dash="dot",
                             annotation_text="TP", annotation_position="right")

        close_reason = close_type if close_type != "close" else "normal"
        dir_label = "Long" if direction == 1 else "Short"
        
        # Calculate trade info
        duration_mins = int((t_close - t_open).total_seconds() / 60)
        duration_secs = int((t_close - t_open).total_seconds() % 60)
        lots_val = float(open_row.get("lots", 1.0)) if pd.notna(open_row.get("lots")) else 1.0
        volume = lots_val  # Volume in lots
        
        # Create extended title with trade info
        title = (
            f"{dir_label} | Open {t_open.strftime('%Y-%m-%d %H:%M')} "
            f"→ Close {t_close.strftime('%H:%M')} | "
            f"PnL: {pnl:+.2f} | {close_reason}<br>"
            f"<sub>Direction: {'Buy' if direction == 1 else 'Sell'} | "
            f"Open: {metrics.entry_price:.2f} | Close: {metrics.exit_price:.2f} | "
            f"Volume: {volume:.2f} | Duration: {duration_mins}m{duration_secs}s</sub>"
        )
        
        fig.update_layout(
            template=_TEMPLATE,
            title=title,
            xaxis_title="Time (M1)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=550,
        )

        fname = _trade_filename(seq_i + 1, open_row, close_row, "html")
        try:
            _save(fig, orders_dir / fname)
        except Exception as exc:
            log.debug("Skipping per-trade HTML %d: %s", seq_i + 1, exc)

