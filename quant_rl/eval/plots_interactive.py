"""Interactive Plotly charts for backtest/eval visualization.

All functions save a self-contained HTML file and return the Figure object.
"""
from __future__ import annotations

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


_TEMPLATE = "plotly_dark"
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

    # One accurate vline per real breach event
    if breach_events:
        for ev in breach_events:
            ts = ev.get("time")
            reason = ev.get("reason", "breach")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            # Use add_shape instead of add_vline to avoid plotly annotation bug
            fig.add_shape(
                type="line",
                x0=ts_str, x1=ts_str, xref="x",
                y0=0, y1=1, yref="paper",
                line=dict(color=_BREACH_COLOR, width=1.0, dash="dash"),
                opacity=0.7,
            )
            fig.add_annotation(
                x=ts_str, xref="x", yref="paper",
                y=0.99, text=reason,
                showarrow=False, font=dict(size=9, color=_BREACH_COLOR),
                align="right",
            )

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

        def _bin_to_candle(times: pd.Series) -> pd.Index:
            """Floor times to candle_tf and snap to nearest available candle."""
            binned = pd.DatetimeIndex(times).floor(candle_tf)
            result = []
            for ts in binned:
                if ts in ohlcv.index:
                    result.append(ts)
                else:
                    loc = ohlcv.index.searchsorted(ts, side="left")
                    loc = min(loc, len(ohlcv) - 1)
                    result.append(ohlcv.index[loc])
            return pd.DatetimeIndex(result)

        def _price_col(df: pd.DataFrame, col: str = "price") -> np.ndarray:
            if col in df.columns:
                return df[col].values
            snapped = _bin_to_candle(df["time"])
            return ohlcv["close"].reindex(snapped, method="nearest", tolerance=pd.Timedelta(candle_tf)).fillna(
                method="ffill"
            ).values

        if not long_opens.empty:
            snapped_x = _bin_to_candle(long_opens["time"])
            fig.add_trace(go.Scatter(
                x=snapped_x, y=_price_col(long_opens),
                mode="markers", name="Long open",
                marker=dict(symbol="triangle-up", size=12, color=_LONG_COLOR, opacity=0.9),
            ))
        if not short_opens.empty:
            snapped_x = _bin_to_candle(short_opens["time"])
            fig.add_trace(go.Scatter(
                x=snapped_x, y=_price_col(short_opens),
                mode="markers", name="Short open",
                marker=dict(symbol="triangle-down", size=12, color=_SHORT_COLOR, opacity=0.9),
            ))
        if not normal_closes.empty:
            snapped_x = _bin_to_candle(normal_closes["time"])
            close_prices = ohlcv["close"].reindex(snapped_x, method="nearest", tolerance=pd.Timedelta(candle_tf)).fillna(method="ffill")
            fig.add_trace(go.Scatter(
                x=snapped_x, y=close_prices.values,
                mode="markers", name="Close",
                marker=dict(symbol="x", size=9, color=_CLOSE_COLOR, opacity=0.85),
            ))
        if not forced_closes.empty:
            snapped_x = _bin_to_candle(forced_closes["time"])
            forced_prices = ohlcv["close"].reindex(snapped_x, method="nearest", tolerance=pd.Timedelta(candle_tf)).fillna(method="ffill")
            fig.add_trace(go.Scatter(
                x=snapped_x, y=forced_prices.values,
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
