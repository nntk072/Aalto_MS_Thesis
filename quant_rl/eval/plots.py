"""Static matplotlib charts for backtest/eval visualization.

All functions save a PNG and return the Figure object.
Uses the non-interactive Agg backend so it works headless in CI.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "grid.color": "#e6e6e6",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "font.size": 10,
}

LONG_COLOR  = "#26a69a"   # teal
SHORT_COLOR = "#ef5350"   # red
CLOSE_COLOR = "#ffd54f"   # amber
EQUITY_COLOR = "#42a5f5"  # blue
PEAK_COLOR   = "#90caf9"  # lighter blue
BREACH_COLOR = "#ff1744"  # bright red


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: Path | str, dpi: int = 150) -> plt.Figure:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 1. Equity curve
# ---------------------------------------------------------------------------
def plot_equity_curve(
    equity: pd.Series,
    breaches: list[str] | None = None,
    breach_events: list[dict] | None = None,
    initial_balance: float = 100_000.0,
    daily_loss_limit: float | None = None,
    max_loss_limit: float | None = None,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Equity over time with peak line, FTMO threshold lines, and breach markers.

    Uses ``breach_events`` (list of dicts with 'time' key) for accurate vertical
    lines.  Falls back to legacy ``breaches`` count only when no events available.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    peak = equity.cummax()
    ax.plot(equity.index, equity.values, color=EQUITY_COLOR, label="Equity")
    ax.plot(equity.index, peak.values, color=PEAK_COLOR, alpha=0.5, linewidth=0.8, label="Peak equity")

    if daily_loss_limit is not None:
        ax.axhline(initial_balance - daily_loss_limit, color="#ff9800", linewidth=0.9,
                   linestyle=":", label=f"Daily loss limit (${daily_loss_limit:,.0f})")
    if max_loss_limit is not None:
        ax.axhline(initial_balance - max_loss_limit, color=BREACH_COLOR, linewidth=0.9,
                   linestyle=":", label=f"Max loss limit (${max_loss_limit:,.0f})")

    # User preference: do not draw breach vertical lines on equity chart.

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    ax.set_title("Equity Curve", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance (USD)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 2. Drawdown
# ---------------------------------------------------------------------------
def plot_drawdown(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Underwater drawdown area chart."""
    _apply_style()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(dd.index, dd.values * 100, 0, color=SHORT_COLOR, alpha=0.6, label="Drawdown")
    ax.plot(dd.index, dd.values * 100, color=SHORT_COLOR, linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    ax.set_title("Drawdown", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 3. Price + candlestick + order markers
# ---------------------------------------------------------------------------
def plot_price_with_orders(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None,
    max_points: int = 3000,
    candle_tf: str = "15min",
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Candlestick chart with buy/sell/close markers.

    Resamples `bars` (M1) to `candle_tf` (e.g. "15min") and caps at
    `max_points` candles so the PNG stays readable.
    """
    try:
        import mplfinance as mpf
    except ImportError:
        return _plot_price_line_fallback(bars, trades, max_points, out_path, dpi)

    _apply_style()

    # Resample to target timeframe
    ohlcv = bars[["open", "high", "low", "close"]].copy()
    if "volume" in bars.columns:
        ohlcv["volume"] = bars["volume"]
    ohlcv = ohlcv.resample(candle_tf).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        **({} if "volume" not in ohlcv.columns else {"volume": "sum"}),
    }).dropna()

    if len(ohlcv) > max_points:
        ohlcv = ohlcv.iloc[-max_points:]

    # Build addplots for orders — candle-interval binning via floor to candle_tf
    addplots = []
    if trades is not None and not trades.empty and "time" in trades.columns:
        opens  = trades[trades["type"] == "open"].copy()
        closes = trades[trades["type"].isin(["close", "forced_close", "eod_close", "stop_close"])].copy()
        forced = trades[trades["type"].isin(["forced_close", "stop_close"])].copy()

        min_ts = ohlcv.index.min()
        max_ts = ohlcv.index.max()

        def _bin_time(ts: pd.Timestamp) -> pd.Timestamp | None:
            """Floor timestamp to candle bin and drop markers outside the plotted window."""
            if ts < min_ts or ts > max_ts:
                return None
            binned = ts.floor(candle_tf)
            if binned in ohlcv.index:
                return binned
            # fallback: nearest right candle within range
            loc = ohlcv.index.searchsorted(ts, side="left")
            if loc >= len(ohlcv):
                return None
            candidate = ohlcv.index[loc]
            if candidate < min_ts or candidate > max_ts:
                return None
            return candidate

        def _make_series(subset: pd.DataFrame, direction: int) -> pd.Series | None:
            mask = (subset["direction"] == direction) if "direction" in subset.columns else pd.Series(False, index=subset.index)
            rows = subset[mask] if "direction" in subset.columns else pd.DataFrame()
            if rows.empty:
                return None
            s = pd.Series(np.nan, index=ohlcv.index)
            for _, row in rows.iterrows():
                candle_ts = _bin_time(row["time"])
                if candle_ts is None:
                    continue
                loc = ohlcv.index.get_loc(candle_ts)
                s.iloc[loc] = ohlcv["low"].iloc[loc] * 0.9995 if direction == 1 else ohlcv["high"].iloc[loc] * 1.0005
            return s

        long_s  = _make_series(opens, 1)
        short_s = _make_series(opens, -1)

        close_s = pd.Series(np.nan, index=ohlcv.index)
        forced_s = pd.Series(np.nan, index=ohlcv.index)
        for _, row in closes.iterrows():
            candle_ts = _bin_time(row["time"])
            if candle_ts is None:
                continue
            loc = ohlcv.index.get_loc(candle_ts)
            if row["type"] in ("forced_close", "stop_close"):
                forced_s.iloc[loc] = ohlcv["close"].iloc[loc]
            else:
                close_s.iloc[loc] = ohlcv["close"].iloc[loc]

        if long_s is not None and long_s.notna().any():
            addplots.append(mpf.make_addplot(long_s, type="scatter", markersize=80,
                                              marker="^", color=LONG_COLOR))
        if short_s is not None and short_s.notna().any():
            addplots.append(mpf.make_addplot(short_s, type="scatter", markersize=80,
                                              marker="v", color=SHORT_COLOR))
        if close_s.notna().any():
            addplots.append(mpf.make_addplot(close_s, type="scatter", markersize=50,
                                              marker="x", color=CLOSE_COLOR))
        if forced_s.notna().any():
            addplots.append(mpf.make_addplot(forced_s, type="scatter", markersize=80,
                                              marker="X", color=BREACH_COLOR))

    mc = mpf.make_marketcolors(up=LONG_COLOR, down=SHORT_COLOR, edge="inherit",
                                wick="inherit", volume="in")
    s  = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc,
                             facecolor="#ffffff", figcolor="#ffffff",
                             gridcolor="#e6e6e6", gridstyle="--")

    save_kwargs: dict[str, Any] = {}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_kwargs["savefig"] = dict(fname=str(out_path), dpi=dpi, bbox_inches="tight")

    plot_kwargs: dict[str, Any] = {
        "type": "candle",
        "style": s,
        "title": "Price + Orders",
        "warn_too_much_data": len(ohlcv) + 1,
        "returnfig": True,
        **save_kwargs,
    }
    if addplots:
        plot_kwargs["addplot"] = addplots

    fig, _ = mpf.plot(ohlcv, **plot_kwargs)
    plt.close(fig)
    return fig


def _plot_price_line_fallback(bars, trades, max_points, out_path, dpi) -> plt.Figure:
    """Fallback to a simple line chart when mplfinance is not installed."""
    _apply_style()
    close = bars["close"].copy()
    if len(close) > max_points:
        close = close.iloc[-max_points:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(close.index, close.values, color=EQUITY_COLOR, linewidth=0.8, label="Close")

    if trades is not None and not trades.empty and "time" in trades.columns:
        opens  = trades[trades["type"] == "open"]
        long_opens  = opens[opens.get("direction", pd.Series()) == 1] if "direction" in opens.columns else pd.DataFrame()
        short_opens = opens[opens.get("direction", pd.Series()) == -1] if "direction" in opens.columns else pd.DataFrame()
        closes = trades[trades["type"].isin(["close", "forced_close", "eod_close"])]

        if not long_opens.empty:
            ax.scatter(long_opens["time"], long_opens["price"] if "price" in long_opens.columns else
                       close.reindex(long_opens["time"], method="nearest").values,
                       marker="^", color=LONG_COLOR, s=40, label="Long", zorder=5)
        if not short_opens.empty:
            ax.scatter(short_opens["time"], short_opens["price"] if "price" in short_opens.columns else
                       close.reindex(short_opens["time"], method="nearest").values,
                       marker="v", color=SHORT_COLOR, s=40, label="Short", zorder=5)
        if not closes.empty and "equity" in closes.columns:
            close_prices = close.reindex(closes["time"], method="nearest")
            ax.scatter(closes["time"], close_prices.values,
                       marker="x", color=CLOSE_COLOR, s=30, label="Close", zorder=5)

    ax.set_title("Price + Orders", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 4. Trade PnL histogram
# ---------------------------------------------------------------------------
def plot_trade_pnl_hist(
    trades: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Histogram of per-trade PnL with win / loss colour split."""
    _apply_style()
    if "pnl" not in trades.columns or trades["pnl"].dropna().empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No trade PnL data", ha="center", va="center", transform=ax.transAxes)
        if out_path:
            _save(fig, out_path, dpi)
        return fig

    pnl = trades["pnl"].dropna()
    wins   = pnl[pnl >= 0]
    losses = pnl[pnl < 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = min(60, max(10, len(pnl) // 10))
    if not wins.empty:
        ax.hist(wins.values,   bins=bins, color=LONG_COLOR,  alpha=0.8, label=f"Wins ({len(wins)})")
    if not losses.empty:
        ax.hist(losses.values, bins=bins, color=SHORT_COLOR, alpha=0.8, label=f"Losses ({len(losses)})")
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


# ---------------------------------------------------------------------------
# 5. Bar-return distribution
# ---------------------------------------------------------------------------
def plot_returns_dist(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar-return distribution with normal overlay."""
    _apply_style()
    rets = equity.pct_change().dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = min(80, max(20, len(rets) // 50))
    ax.hist(rets.values * 100, bins=bins, color=EQUITY_COLOR, alpha=0.75, density=True, label="Returns")

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


# ---------------------------------------------------------------------------
# 6. Monthly returns heatmap
# ---------------------------------------------------------------------------
def plot_monthly_returns_heatmap(
    equity: pd.Series,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Calendar heatmap of monthly returns."""
    _apply_style()
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                ha="center", va="center", transform=ax.transAxes)
        if out_path:
            _save(fig, out_path, dpi)
        return fig

    years  = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    matrix = np.full((len(years), 12), np.nan)
    for dt, val in monthly.items():
        yi = years.index(dt.year)
        mi = dt.month - 1
        matrix[yi, mi] = val

    vabs = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(months)), max(3, len(years))))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for yi in range(len(years)):
        for mi in range(12):
            v = matrix[yi, mi]
            if not np.isnan(v):
                ax.text(mi, yi, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color="#000" if abs(v) < vabs * 0.6 else "#fff")

    plt.colorbar(im, ax=ax, fraction=0.03, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap", fontweight="bold")
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig


# ---------------------------------------------------------------------------
# 7. Baseline comparison
# ---------------------------------------------------------------------------
def plot_baseline_comparison(
    equity_dict: dict[str, pd.Series],
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Overlay normalized equity curves for multiple strategies."""
    _apply_style()
    colors = [EQUITY_COLOR, LONG_COLOR, SHORT_COLOR, CLOSE_COLOR, "#ce93d8", "#80cbc4"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for (name, eq), col in zip(equity_dict.items(), colors):
        normed = eq / eq.iloc[0]
        ax.plot(normed.index, normed.values, label=name, color=col)

    ax.axhline(1.0, color="#555", linewidth=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{(x-1)*100:.0f}%"))
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    ax.set_title("Strategy Comparison — Normalized Returns", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig
