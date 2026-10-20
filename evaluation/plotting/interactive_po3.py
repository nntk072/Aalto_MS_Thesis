from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interactive_common import (
    _LONG_COLOR,
    _SHORT_COLOR,
    _TEMPLATE,
    _check,
    _save,
    go,
)


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
        Output of :func:`~features.po3.config.detect_po3_entries`
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
    from evaluation.plotting.po3_plots import _resample_ohlc
    from features.po3.zones import build_fvg_zones_for_plot

    if not len(bars):
        raise ValueError("No bars to plot after applying window.")

    zones = build_fvg_zones_for_plot(bars)
    plot_bars = bars
    if window is not None:
        start, end = window
        plot_bars = bars.loc[start:end]
    if not len(plot_bars):
        raise ValueError("No bars to plot after applying window.")

    ohlcv = _resample_ohlc(plot_bars, candle_tf)
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
            line=dict(width=0.6, color=zone_colors[(z.kind, z.side)]),
            layer="below",
        )

    # Entry markers at candle close.
    if show_entries:
        for col, marker, color in (
            ("entry_long", "triangle-up", _LONG_COLOR),
            ("entry_short", "triangle-down", _SHORT_COLOR),
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
                if loc > 0 and ohlcv.index[loc] > ts:
                    loc = loc - 1
                xs.append(ohlcv.index[loc])
                ys.append(float(ohlcv["close"].iloc[loc]))
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
