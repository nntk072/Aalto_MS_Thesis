from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

from .common import (
    _NOTE_FONT_PT,
    BREACH_COLOR,
    CLOSE_COLOR,
    EQUITY_COLOR,
    LONG_COLOR,
    SHORT_COLOR,
    _apply_style,
    _save,
)
from .order_window import _extend_trade_window as _extend_trade_window
from .order_window import _extract_window as _extract_window
from .order_window import _structure_anchors as _structure_anchors

log = logging.getLogger(__name__)


def plot_price_with_orders(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None,
    max_points: int = 3000,
    candle_tf: str = "15min",
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
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
    ohlcv = (
        ohlcv.resample(candle_tf)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                **({} if "volume" not in ohlcv.columns else {"volume": "sum"}),
            }
        )
        .dropna()
    )

    if len(ohlcv) > max_points:
        ohlcv = ohlcv.iloc[-max_points:]

    # Build addplots for orders — candle-interval binning via floor to candle_tf
    addplots = []
    if trades is not None and not trades.empty and "time" in trades.columns:
        opens = trades[trades["type"] == "open"].copy()
        closes = trades[
            trades["type"].isin(["close", "forced_close", "eod_close", "stop_close", "tp_close"])
        ].copy()

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
            loc = int(ohlcv.index.searchsorted(ts, side="left"))
            if loc >= len(ohlcv):
                return None
            candidate = ohlcv.index[loc]
            if candidate < min_ts or candidate > max_ts:
                return None
            return pd.Timestamp(candidate)

        def _make_series(subset: pd.DataFrame, direction: int) -> pd.Series | None:
            mask = (
                (subset["direction"] == direction)
                if "direction" in subset.columns
                else pd.Series(False, index=subset.index)
            )
            rows = subset[mask] if "direction" in subset.columns else pd.DataFrame()
            if rows.empty:
                return None
            s = pd.Series(np.nan, index=ohlcv.index)
            for _, row in rows.iterrows():
                candle_ts = _bin_time(row["time"])
                if candle_ts is None:
                    continue
                loc = cast(int, ohlcv.index.get_loc(candle_ts))
                s.iloc[loc] = (
                    ohlcv["low"].iloc[loc] * 0.9995
                    if direction == 1
                    else ohlcv["high"].iloc[loc] * 1.0005
                )
            return s

        long_s = _make_series(opens, 1)
        short_s = _make_series(opens, -1)

        close_s = pd.Series(np.nan, index=ohlcv.index)
        forced_s = pd.Series(np.nan, index=ohlcv.index)
        for _, row in closes.iterrows():
            candle_ts = _bin_time(row["time"])
            if candle_ts is None:
                continue
            loc = cast(int, ohlcv.index.get_loc(candle_ts))
            if row["type"] in ("forced_close", "stop_close"):
                forced_s.iloc[loc] = ohlcv["close"].iloc[loc]
            else:
                close_s.iloc[loc] = ohlcv["close"].iloc[loc]

        if long_s is not None and long_s.notna().any():
            addplots.append(
                mpf.make_addplot(
                    long_s, type="scatter", markersize=80, marker="^", color=LONG_COLOR
                )
            )
        if short_s is not None and short_s.notna().any():
            addplots.append(
                mpf.make_addplot(
                    short_s, type="scatter", markersize=80, marker="v", color=SHORT_COLOR
                )
            )
        if close_s.notna().any():
            addplots.append(
                mpf.make_addplot(
                    close_s, type="scatter", markersize=50, marker="x", color=CLOSE_COLOR
                )
            )
        if forced_s.notna().any():
            addplots.append(
                mpf.make_addplot(
                    forced_s, type="scatter", markersize=80, marker="X", color=BREACH_COLOR
                )
            )

    mc = mpf.make_marketcolors(
        up=LONG_COLOR, down=SHORT_COLOR, edge="inherit", wick="inherit", volume="in"
    )
    s = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        marketcolors=mc,
        facecolor="#ffffff",
        figcolor="#ffffff",
        gridcolor="#e6e6e6",
        gridstyle="--",
    )

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

    fig_any, _ = mpf.plot(ohlcv, **plot_kwargs)
    fig = cast(Figure, fig_any)
    plt.close(fig)
    return fig


def _plot_price_line_fallback(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None,
    max_points: int,
    out_path: Path | str | None,
    dpi: int,
) -> Figure:
    """Fallback to a simple line chart when mplfinance is not installed."""
    _apply_style()
    close = bars["close"].copy()
    if len(close) > max_points:
        close = close.iloc[-max_points:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(close.index, close.to_numpy(), color=EQUITY_COLOR, linewidth=0.8, label="Close")

    if trades is not None and not trades.empty and "time" in trades.columns:
        opens = trades[trades["type"] == "open"]
        long_opens = (
            opens[opens.get("direction", pd.Series()) == 1]
            if "direction" in opens.columns
            else pd.DataFrame()
        )
        short_opens = (
            opens[opens.get("direction", pd.Series()) == -1]
            if "direction" in opens.columns
            else pd.DataFrame()
        )
        closes = trades[
            trades["type"].isin(["close", "forced_close", "eod_close", "stop_close", "tp_close"])
        ]

        if not long_opens.empty:
            ax.scatter(
                long_opens["time"],
                long_opens["price"]
                if "price" in long_opens.columns
                else close.reindex(long_opens["time"], method="nearest").to_numpy(),
                marker="^",
                color=LONG_COLOR,
                s=40,
                label="Long",
                zorder=5,
            )
        if not short_opens.empty:
            ax.scatter(
                short_opens["time"],
                short_opens["price"]
                if "price" in short_opens.columns
                else close.reindex(short_opens["time"], method="nearest").to_numpy(),
                marker="v",
                color=SHORT_COLOR,
                s=40,
                label="Short",
                zorder=5,
            )
        if not closes.empty and "equity" in closes.columns:
            close_prices = close.reindex(closes["time"], method="nearest")
            ax.scatter(
                closes["time"],
                close_prices.to_numpy(),
                marker="x",
                color=CLOSE_COLOR,
                s=30,
                label="Close",
                zorder=5,
            )

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


def _per_trade_equity(
    pairs: list[tuple[pd.Series, pd.Series]],
    equity: pd.Series,
) -> tuple[pd.DatetimeIndex, np.ndarray[Any, Any]]:
    """Equity after each completed trade.

    Prefer the close-row ``equity`` from the trade log (authoritative). Only
    fall back to reindexing the continuous equity series when that column is
    missing — reindex is unsafe when equity timestamps were misaligned.
    """
    # Trades can combine timestamps emitted by different market-data paths,
    # some naive and some timezone-aware. Normalize them before constructing
    # the index so report export remains robust across both paths.
    times = pd.to_datetime([c["time"] for _, c in pairs], utc=True)
    vals = np.empty(len(pairs), dtype=float)
    need_fallback = False
    for i, (_, close_row) in enumerate(pairs):
        if "equity" in close_row.index and pd.notna(close_row["equity"]):
            vals[i] = float(close_row["equity"])
        else:
            vals[i] = np.nan
            need_fallback = True
    if need_fallback:
        filled = equity.reindex(times).ffill().bfill()
        miss = np.isnan(vals)
        vals[miss] = np.asarray(filled.to_numpy(dtype=float))[miss]
    return pd.DatetimeIndex(times), vals


def _candle_body_days(index: pd.DatetimeIndex) -> float:
    """Body width of one bar, in matplotlib day units."""
    if len(index) < 2:
        step = pd.Timedelta("1min")
    else:
        deltas = index[1:] - index[:-1]
        positive = deltas[deltas > pd.Timedelta(0)]
        step = pd.Timedelta(pd.Series(positive).median()) if len(positive) else pd.Timedelta("1min")
        if step <= pd.Timedelta(0):
            step = pd.Timedelta("1min")
    return float(step / pd.Timedelta("1D"))


def _axis_date_format(window: pd.DataFrame) -> str:
    """``%H:%M`` for a single calendar day; include month/day otherwise."""
    idx = pd.DatetimeIndex(window.index)
    if len(idx) == 0:
        return "%H:%M"
    days = idx.normalize().unique()
    return "%H:%M" if len(days) == 1 else "%m/%d %H:%M"


def _close_time_label(t_open: pd.Timestamp, t_close: pd.Timestamp) -> str:
    """Close time for chart titles; include the date when it differs from open."""
    if pd.Timestamp(t_open).normalize() == pd.Timestamp(t_close).normalize():
        return pd.Timestamp(t_close).strftime("%H:%M")
    return pd.Timestamp(t_close).strftime("%Y-%m-%d %H:%M")


def _trade_filename(idx: int, open_row: pd.Series, close_row: pd.Series, ext: str) -> str:
    """Build a descriptive filename for a per-trade chart.

    Format: ``trade_NNNN_YYYYMMDD_HHMMopen_HHMMclose_{L|S}_{p|m}PnL.{ext}``
    """
    t_open = pd.Timestamp(open_row["time"])
    t_close = pd.Timestamp(close_row["time"])
    date = t_open.strftime("%Y%m%d")
    t_o = t_open.strftime("%H%M")
    t_c = t_close.strftime("%H%M")
    side = "L" if open_row.get("direction", 0) == 1 else "S"
    pnl = float(close_row["pnl"]) if pd.notna(close_row.get("pnl")) else 0.0
    sign = "p" if pnl >= 0 else "m"
    return f"trade_{idx:04d}_{date}_{t_o}_{t_c}_{side}_{sign}{abs(pnl):.1f}.{ext}"


def plot_per_trade_orders(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    orders_dir: Path | str,
    context_bars: int = 60,
    max_charts: int = 200,
    dpi: int = 150,
    max_loss_per_trade_usd: float | None = None,
    take_profit_per_trade_usd: float | None = None,
    lots: float = 1.0,
    contract_size: float = 1.0,
    show_mae_mfe: bool = True,
    show_sl_tp: bool = True,
    show_vwap: bool = False,
    secondary_bars: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> None:
    """Generate 2-panel PNG per trade with datetime axes, EMA50, MACD, and trade info.

    Filenames encode the key trade metadata::

        trade_NNNN_YYYYMMDD_HHMMopen_HHMMclose_{L|S}_{p|m}PnL.png

    Parameters
    ----------
    bars : M1 price bars (DatetimeIndex, open/high/low/close columns).
    trades : Trade log with type/direction/price/time/pnl columns.
    orders_dir : Destination folder; created if absent.
    context_bars : M1 bars to show before entry and after exit (~1 min each).
    max_charts : Cap on number of charts; trades are sampled evenly when
        the total exceeds this limit.
    dpi : PNG resolution.
    max_loss_per_trade_usd : Maximum loss limit for SL calculation.
    take_profit_per_trade_usd : Take profit limit for TP calculation.
    lots : Position size in lots.
    contract_size : Contract size.
    show_mae_mfe : Whether to plot MAE/MFE lines.
    show_sl_tp : Whether to plot SL/TP lines.
    show_vwap : Draw session VWAP. Off until the feed has real tick volume.
    secondary_bars : Optional US500 (or correlated) bars for SMT lines.
    """

    from evaluation.reporting.trade_note import build_trade_note

    from .chart_indicators import break_overlay_gaps, compute_chart_overlays_full
    from .chart_overlays import (
        VWAP_COLOR,
        build_overlay_events,
        draw_macd_rsi_mpl,
        draw_overlays_mpl,
    )
    from .order_chart import draw_order_levels_mpl
    from .order_window import _session_end_positions, prepare_order_chart
    from .overlay_events import confirmed_pivots

    orders_dir = Path(orders_dir)
    orders_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pair_trades(trades)
    if not pairs:
        return

    total = len(pairs)
    if total > max_charts:
        indices = np.linspace(0, total - 1, max_charts, dtype=int).tolist()
        pairs = [pairs[i] for i in indices]
        log.info("Per-trade PNG: sampling %d/%d trades → %s", max_charts, total, orders_dir)
    else:
        log.info("Per-trade PNG: %d charts → %s", total, orders_dir)

    # Compute EMA50/MACD/RSI/VWAP once on the full history so indicators are
    # warmed up and match the strategy's real signals, then slice per trade.
    full_overlays = compute_chart_overlays_full(bars)
    overlay_events = build_overlay_events(bars, secondary=secondary_bars)
    confirmed_events = confirmed_pivots(overlay_events, bars)
    session_end_positions = _session_end_positions(pd.DatetimeIndex(bars.index))

    for seq_i, (open_row, close_row) in enumerate(pairs):
        prepared = prepare_order_chart(
            bars,
            open_row,
            close_row,
            overlay_events=overlay_events,
            full_overlays=full_overlays,
            context_bars=context_bars,
            max_loss_per_trade_usd=max_loss_per_trade_usd,
            take_profit_per_trade_usd=take_profit_per_trade_usd,
            lots=lots,
            contract_size=contract_size,
            show_mae_mfe=show_mae_mfe,
            show_sl_tp=show_sl_tp,
            features=features,
            confirmed_events=confirmed_events,
            session_end_positions=session_end_positions,
        )
        if prepared is None:
            continue
        window = prepared.window
        metrics = prepared.metrics
        overlays = prepared.overlays
        levels = prepared.levels
        events = prepared.events
        t_open = prepared.t_open
        t_close = prepared.t_close
        overlays = break_overlay_gaps(overlays)

        direction = int(open_row["direction"]) if pd.notna(open_row.get("direction")) else 0
        pnl = float(close_row["pnl"]) if pd.notna(close_row.get("pnl")) else 0.0
        close_type = str(close_row["type"])

        # Create 2-panel figure: price + dual-axis MACD/RSI
        fig, (ax_price, ax_osc) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            gridspec_kw={"height_ratios": [2, 1]},
            sharex=True,
        )

        # --- Top panel: Candlesticks + EMA50 + Entry/Exit + MAE/MFE/SL/TP ---
        colors_up = [
            LONG_COLOR if o <= c else SHORT_COLOR for o, c in zip(window["open"], window["close"])
        ]

        body_width = _candle_body_days(pd.DatetimeIndex(window.index))
        for i, (idx, row) in enumerate(window.iterrows()):
            hl_color = colors_up[i]
            # High-low line
            ax_price.plot(
                [idx, idx],
                [row["low"], row["high"]],
                color=hl_color,
                linewidth=0.5,
            )
            # Open-close body
            body_height = abs(row["close"] - row["open"])
            body_bottom = min(row["open"], row["close"])
            x0 = mdates.date2num(cast(pd.Timestamp, idx)) - body_width / 2
            rect = Rectangle(
                (x0, body_bottom),
                body_width,
                body_height,
                facecolor=hl_color,
                edgecolor=hl_color,
                linewidth=0.8,
            )
            ax_price.add_patch(rect)

        # HTF/LTF imbalance boxes, same geometry as the PO3 sample figure.
        from evaluation.plotting.po3_plots import _draw_zones
        from features.po3.zones import build_fvg_zones_for_plot

        _draw_zones(ax_price, build_fvg_zones_for_plot(window), window)

        # EMA50
        ema = overlays["ema50"]
        ax_price.plot(
            ema.index,
            ema.to_numpy(),
            color="#0066cc",
            linewidth=2,
            label="EMA50",
            zorder=3,
        )
        if show_vwap:
            vwap = overlays["vwap"]
            ax_price.plot(
                vwap.index,
                vwap.to_numpy(),
                color=VWAP_COLOR,
                linewidth=1.7,
                linestyle="-.",
                label="VWAP",
                zorder=3,
            )
        draw_overlays_mpl(ax_price, events)
        draw_order_levels_mpl(ax_price, levels)
        # Structure lines must not stretch a one-minute trade across later months.
        pad = float(pd.Timedelta("30s") / pd.Timedelta("1D"))
        ax_price.set_xlim(
            mdates.date2num(pd.Timestamp(window.index[0])) - pad,
            mdates.date2num(pd.Timestamp(window.index[-1])) + pad,
        )

        # Formatting
        ax_price.set_ylabel("Price")
        ax_price.legend(loc="upper left", fontsize=8)
        ax_price.grid(True, alpha=0.3)

        # Format x-axis in the bars' own timezone. Use month/day when the
        # window spans multiple calendar days so labels cannot collide.
        axis_tz = pd.DatetimeIndex(window.index).tz
        date_fmt = _axis_date_format(window)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=axis_tz))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12))
        plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- Bottom panel: MACD (left) + RSI (right) ---
        draw_macd_rsi_mpl(ax_osc, window, overlays)
        ax_osc.set_xlabel("Time (M1)")
        ax_osc.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt, tz=axis_tz))
        ax_osc.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=12))
        plt.setp(ax_osc.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # --- Trade note below both subplots ---
        duration_mins = int((t_close - t_open).total_seconds() / 60)
        duration_secs = int((t_close - t_open).total_seconds() % 60)
        lots_val = float(open_row.get("lots", 1.0)) if pd.notna(open_row.get("lots")) else 1.0

        # Reconciliation: recompute PnL from the entry/exit prices shown on
        # this chart and compare to the logged PnL. A mismatch flags a
        # pairing bug or a stale/fallback exit price rather than a "weird"
        # strategy result.
        pnl_calc = (metrics.exit_price - metrics.entry_price) * direction * lots_val * contract_size
        close_reason_detail = (
            str(close_row.get("reason")) if pd.notna(close_row.get("reason")) else close_type
        )

        note_lines = build_trade_note(
            open_row,
            close_row,
            entry_price=metrics.entry_price,
            exit_price=metrics.exit_price,
            direction=direction,
            pnl=pnl,
            pnl_calc=pnl_calc,
            duration_mins=duration_mins,
            duration_secs=duration_secs,
            lots=lots_val,
            close_reason_detail=close_reason_detail,
            extra_notes=levels.notes,
        )
        # The caption sits under the two subplots so it never covers candles.
        # Its block height is derived from the real line count and the figure is
        # grown to fit: with a fixed note_top/0.028 step, 6+ note lines spilled
        # to y<0, note_bottom clamped to 0, and tight_layout reserved no room --
        # so bbox_inches="tight" pulled the note onto the rotated x tick labels.
        line_in = _NOTE_FONT_PT * 1.35 / 72.0
        n_note_lines = len(note_lines)
        note_h_in = n_note_lines * line_in + 0.22
        fig_h_in = max(9.0, note_h_in + 7.0)
        fig.set_size_inches(fig.get_figwidth(), fig_h_in)

        note_top = note_h_in / fig_h_in
        for i, line in enumerate(note_lines):
            fig.text(
                0.012,
                note_top - i * line_in / fig_h_in,
                line,
                fontsize=_NOTE_FONT_PT,
                family="monospace",
                ha="left",
                va="top",
            )
        # Reserve the note band (plus a gap) and a little left padding so the
        # 45-degree tick labels cannot reach the figure edge.
        note_bottom = min(0.9, note_top + 0.006)

        # --- Title ---
        dir_label = "Long" if direction == 1 else "Short"
        close_reason = close_type if close_type != "close" else "normal"
        title = (
            f"{dir_label} | Open {t_open.strftime('%Y-%m-%d %H:%M')} "
            f"→ Close {_close_time_label(t_open, t_close)} | "
            f"PnL: {pnl:+.2f} | {close_reason}"
        )
        fig.suptitle(title, fontsize=11, fontweight="bold")

        # Save
        fname = _trade_filename(seq_i + 1, open_row, close_row, "png")
        try:
            fig.tight_layout(rect=(0.008, note_bottom, 0.995, 0.96))
            fig.savefig(str(orders_dir / fname), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log.debug("Skipping per-trade PNG %d: %s", seq_i + 1, exc)
