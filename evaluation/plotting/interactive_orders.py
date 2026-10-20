from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .interactive_common import (
    _BREACH_COLOR,
    _CLOSE_COLOR,
    _LONG_COLOR,
    _SHORT_COLOR,
    _TEMPLATE,
    _check,
    _save,
    go,
    log,
)


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
        normal_closes = trades[trades["type"].isin(["close", "eod_close", "tp_close"])]
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
    show_vwap: bool = False,
    secondary_bars: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
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
    show_vwap : Draw session VWAP. Off until the feed has real tick volume.
    secondary_bars : Optional US500 (or correlated) bars for SMT lines.
    """
    _check()
    from plotly.subplots import make_subplots

    from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

    from .chart_indicators import break_overlay_gaps, compute_chart_overlays_full
    from .chart_overlays import (
        VWAP_COLOR,
        build_overlay_events,
        draw_macd_rsi_plotly,
        draw_overlays_plotly,
    )
    from .order_chart import draw_order_levels_plotly
    from .order_window import _session_end_positions, prepare_order_chart
    from .overlay_events import confirmed_pivots
    from .plots import _trade_filename

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
        ema = overlays["ema50"]
        fig.add_trace(
            go.Scatter(
                x=ema.index,
                y=ema.to_numpy(),
                name="EMA50",
                line=dict(color="#0066cc", width=2),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        if show_vwap:
            vwap = overlays["vwap"]
            fig.add_trace(
                go.Scatter(
                    x=vwap.index,
                    y=vwap.to_numpy(),
                    name="VWAP",
                    connectgaps=False,
                    line=dict(color=VWAP_COLOR, width=1.8, dash="dashdot"),
                ),
                row=1,
                col=1,
            )
        draw_overlays_plotly(fig, events, row=1, col=1)
        draw_order_levels_plotly(fig, levels)

        # Bottom panel: MACD (left) + RSI (right)
        draw_macd_rsi_plotly(fig, window, overlays, row=2)

        close_reason = close_type if close_type != "close" else "normal"
        dir_label = "Long" if direction == 1 else "Short"

        # Calculate trade info
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

        from evaluation.reporting.trade_note import build_trade_note

        from .plots import _close_time_label

        # Same caption as the PNG renderer, shown under the two subplots.
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

        # Title keeps the headline; the detail lives in the note block.
        title = (
            f"{dir_label} | Open {t_open.strftime('%Y-%m-%d %H:%M')} "
            f"→ Close {_close_time_label(t_open, t_close)} | "
            f"PnL: {pnl:+.2f} | {close_reason}"
        )

        # Hide overnight / weekend gaps when a multi-day window slips through.
        days = pd.DatetimeIndex(window.index).normalize().unique()
        clock = pd.DatetimeIndex(window.index)
        pre_ny = ((clock.hour < 16) | ((clock.hour == 16) & (clock.minute < 30))).any()
        rangebreaks: list[dict[str, Any]] = [dict(bounds=["sat", "mon"])]
        if not pre_ny:
            rangebreaks.append(dict(bounds=[23.01, 16.5], pattern="hour"))
        tickformat = "%H:%M" if len(days) == 1 else "%m/%d %H:%M"

        fig.update_xaxes(
            range=[window.index[0], window.index[-1]],
        )
        fig.update_layout(
            template=_TEMPLATE,
            title=title,
            xaxis2_title="Time (M1)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            height=680 + 22 * len(note_lines),
            margin=dict(b=40 + 18 * len(note_lines)),
            xaxis=dict(rangebreaks=rangebreaks, tickformat=tickformat),
            xaxis2=dict(rangebreaks=rangebreaks, tickformat=tickformat),
        )
        # Caption under both subplots, matching the PNG renderer.
        for i, line in enumerate(note_lines):
            fig.add_annotation(
                x=0,
                y=-0.12 - 0.045 * i,
                xref="paper",
                yref="paper",
                text=line,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(family="Courier New, monospace", size=11),
            )
        fig.update_yaxes(range=list(levels.ylim), row=1, col=1)

        fname = _trade_filename(seq_i + 1, open_row, close_row, "html")
        try:
            _save(fig, orders_dir / fname)
        except Exception as exc:
            log.debug("Skipping per-trade HTML %d: %s", seq_i + 1, exc)
