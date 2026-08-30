"""PO3 / IFVG signal charts.

Renders candlesticks with HTF FVG / LTF IFVG zone rectangles and entry markers,
so the signal columns produced by :mod:`quant_rl.features.po3_config` can be
visually inspected next to price.

Uses the non-interactive Agg backend so it works headless in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from quant_rl.eval.plots import LONG_COLOR, SHORT_COLOR, _apply_style, _save
from quant_rl.features.po3_config import FVGZone, build_fvg_zones

log = logging.getLogger(__name__)

# Zone colors (matched to the repo's LONG/SHORT theme).
HTF_FVG_BULL_COLOR = "#1b7a3d"  # darker green  (M15 imbalance)
HTF_FVG_BEAR_COLOR = "#b33131"  # darker red    (M15 imbalance)
LTF_IFVG_BULL_COLOR = "#81d4a8"  # light green  (confirmed M5 IFVG)
LTF_IFVG_BEAR_COLOR = "#ef9a9a"  # light red    (confirmed M5 IFVG)

ENTRY_LONG_COLOR = LONG_COLOR
ENTRY_SHORT_COLOR = SHORT_COLOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resample_ohlc(bars: pd.DataFrame, candle_tf: str) -> pd.DataFrame:
    """Resample M1 bars to a coarser candle rule (e.g. '5min')."""
    ohlcv = bars[["open", "high", "low", "close"]].copy()
    agg_spec: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in bars.columns:
        ohlcv["volume"] = bars["volume"]
        agg_spec["volume"] = "sum"
    return ohlcv.resample(candle_tf).agg(agg_spec).dropna(subset=["open"])  # type: ignore[arg-type]


def _draw_candles(ax: Axes, ohlcv: pd.DataFrame) -> None:
    """Draw simple OHLC candlesticks on ``ax`` using date-num x positions."""
    xs = mdates.date2num(ohlcv.index)  # type: ignore[no-untyped-call]
    for i, (_, row) in enumerate(ohlcv.iterrows()):
        x = float(xs[i])
        color = LONG_COLOR if row["close"] >= row["open"] else SHORT_COLOR
        # Wick
        ax.plot([x, x], [row["low"], row["high"]], color=color, linewidth=1.0)
        # Body
        y_low = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"])
        if height < 1e-12:
            ax.plot([x - 0.35, x + 0.35], [row["close"], row["close"]], color=color, linewidth=1.0)
        else:
            ax.add_patch(
                Rectangle(
                    (x - 0.35, y_low),
                    0.7,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5,
                )
            )


def _zone_color(zone: FVGZone) -> str:
    if zone.kind == "htf_fvg":
        return HTF_FVG_BULL_COLOR if zone.side == "bullish" else HTF_FVG_BEAR_COLOR
    return LTF_IFVG_BULL_COLOR if zone.side == "bullish" else LTF_IFVG_BEAR_COLOR


def _draw_zones(ax: Axes, zones: list[FVGZone], ohlcv: pd.DataFrame) -> None:
    """Overlay translucent zone rectangles for every zone overlapping the window."""
    if not zones:
        return
    x_min = float(mdates.date2num(ohlcv.index.min()))  # type: ignore[no-untyped-call]
    x_max = float(mdates.date2num(ohlcv.index.max()))  # type: ignore[no-untyped-call]
    for zone in zones:
        if zone.end_ts < ohlcv.index.min() or zone.start_ts > ohlcv.index.max():
            continue
        x0 = max(float(mdates.date2num(zone.start_ts)), x_min)  # type: ignore[no-untyped-call]
        x1 = min(float(mdates.date2num(zone.end_ts)), x_max)  # type: ignore[no-untyped-call]
        if x1 <= x0:
            continue
        # Filled (invalidated) zones are faded more than active ones.
        alpha = 0.28 if zone.invalidated else 0.40
        ax.add_patch(
            Rectangle(
                (x0, zone.zone_low),
                x1 - x0,
                zone.zone_high - zone.zone_low,
                facecolor=_zone_color(zone),
                edgecolor=_zone_color(zone),
                alpha=alpha,
                linewidth=0.0,
            )
        )


def _draw_entries(ax: Axes, signals: pd.DataFrame, ohlcv: pd.DataFrame) -> None:
    """Scatter entry markers at the top/bottom of the matching candles."""
    for col, marker, color, dy in (
        ("entry_long", "^", ENTRY_LONG_COLOR, 1.004),
        ("entry_short", "v", ENTRY_SHORT_COLOR, 0.996),
    ):
        if col not in signals.columns:
            continue
        hits = signals[col] == 1
        if not hits.any():
            continue
        xs: list[float] = []
        ys: list[float] = []
        for ts in signals.index[hits]:
            if ts < ohlcv.index.min() or ts > ohlcv.index.max():
                continue
            loc = ohlcv.index.searchsorted(ts, side="left")
            if loc >= len(ohlcv):
                continue
            x = float(mdates.date2num(ohlcv.index[loc]))  # type: ignore[no-untyped-call]
            base = ohlcv["high"].iloc[loc] if dy > 1 else ohlcv["low"].iloc[loc]
            xs.append(x)
            ys.append(base * dy)
        if xs:
            ax.scatter(xs, ys, marker=marker, s=55, color=color, zorder=5, label=col)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_fvg_signals(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    candle_tf: str = "5min",
    max_points: int = 3000,
    show_entries: bool = True,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Render PO3/IFVG signals on a candlestick chart.

    Parameters
    ----------
    bars : pd.DataFrame
        M1 OHLC bars (DatetimeIndex) used to compute ``signals``.
    signals : pd.DataFrame
        Output of :func:`~quant_rl.features.po3_config.detect_po3_entries`
        (or any of its sub-detectors).  Shares the index of ``bars``.
    window : tuple[Timestamp, Timestamp], optional
        Slice of the data to display (inclusive).  Defaults to the full range.
    candle_tf : str
        Pandas resample rule for the candles, e.g. ``'1min'``, ``'5min'``, ``'15min'``.
    max_points : int
        Cap on candle count; keeps the PNG readable.
    show_entries : bool
        Draw entry_long/entry_short markers when those columns are present.
    out_path : Path | str, optional
        If given, the PNG is written here (parents created).
    dpi : int
        PNG resolution.

    Returns
    -------
    plt.Figure
        The rendered figure (axes already closed).
    """
    _apply_style()

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

    fig, ax = plt.subplots(figsize=(14, 7))
    _draw_candles(ax, ohlcv)
    _draw_zones(ax, zones, ohlcv)

    if show_entries:
        _draw_entries(ax, signals, ohlcv)

    # Legend with representative proxies.
    handles: list[Any] = [
        Rectangle((0, 0), 1, 1, facecolor=HTF_FVG_BULL_COLOR, alpha=0.4, label="HTF bullish FVG"),
        Rectangle((0, 0), 1, 1, facecolor=HTF_FVG_BEAR_COLOR, alpha=0.4, label="HTF bearish FVG"),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=LTF_IFVG_BULL_COLOR,
            alpha=0.4,
            label="LTF IFVG confirmed (bull)",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=LTF_IFVG_BEAR_COLOR,
            alpha=0.4,
            label="LTF IFVG confirmed (bear)",
        ),
    ]
    if show_entries:
        if "entry_long" in signals.columns:
            handles.append(
                Line2D([], [], marker="^", color=ENTRY_LONG_COLOR, ls="None", label="Long entry")
            )
        if "entry_short" in signals.columns:
            handles.append(
                Line2D([], [], marker="v", color=ENTRY_SHORT_COLOR, ls="None", label="Short entry")
            )
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)

    ax.set_title("PO3 (AMD) + HTF/LTF IFVG Signals")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())  # type: ignore[no-untyped-call]
    )
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig
