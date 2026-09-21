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
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from quant_rl.eval.plots import LONG_COLOR, SHORT_COLOR, _apply_style, _save
from quant_rl.features.po3_config import FVGZone, build_fvg_zones_for_plot

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
    """Draw simple OHLC candlesticks on ``ax`` using date-num x positions.

    Body width is a fraction of the median bar spacing in matplotlib date
    units (days). A fixed width of 0.7 days is ~200× a 5-minute bar and
    paints the chart as solid blobs.
    """
    xs = np.asarray(mdates.date2num(ohlcv.index), dtype=float)  # type: ignore[no-untyped-call,type-var]
    if len(xs) >= 2:
        spacing = float(np.median(np.diff(xs)))
    else:
        spacing = 1.0 / (24.0 * 60.0)  # 1 minute fallback
    half = 0.3 * max(spacing, 1e-9)
    width = 2.0 * half
    for i, (_, row) in enumerate(ohlcv.iterrows()):
        x = float(xs[i])
        color = LONG_COLOR if row["close"] >= row["open"] else SHORT_COLOR
        # Wick
        ax.plot([x, x], [row["low"], row["high"]], color=color, linewidth=1.0)
        # Body
        y_low = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"])
        if height < 1e-12:
            ax.plot([x - half, x + half], [row["close"], row["close"]], color=color, linewidth=1.0)
        else:
            ax.add_patch(
                Rectangle(
                    (x - half, y_low),
                    width,
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


def _candle_half_width(ohlcv: pd.DataFrame) -> float:
    """Half candle body width in matplotlib date units (matches ``_draw_candles``)."""
    xs = np.asarray(mdates.date2num(ohlcv.index), dtype=float)  # type: ignore[no-untyped-call,type-var]
    if len(xs) >= 2:
        spacing = float(np.median(np.diff(xs)))
    else:
        spacing = 1.0 / (24.0 * 60.0)
    return 0.3 * max(spacing, 1e-9)


def _snap_ts_to_candle_x(ts: pd.Timestamp, ohlcv: pd.DataFrame, *, side: str) -> float:
    """Map a zone timestamp to the left (``side='left'``) or right candle edge."""
    idx = ohlcv.index
    loc = int(idx.searchsorted(ts, side="left"))
    if loc >= len(idx):
        loc = len(idx) - 1
    elif loc > 0 and abs(idx[loc] - ts) > abs(idx[loc - 1] - ts):
        loc = loc - 1
    half = _candle_half_width(ohlcv)
    x = float(mdates.date2num(idx[loc]))  # type: ignore[no-untyped-call]
    return x - half if side == "left" else x + half


def _draw_zones(ax: Axes, zones: list[FVGZone], ohlcv: pd.DataFrame) -> None:
    """Overlay translucent zone rectangles snapped to displayed candle edges."""
    if not zones:
        return
    x_min = float(mdates.date2num(ohlcv.index.min()))  # type: ignore[no-untyped-call]
    x_max = float(mdates.date2num(ohlcv.index.max()))  # type: ignore[no-untyped-call]
    half = _candle_half_width(ohlcv)
    x_min -= half
    x_max += half
    for zone in zones:
        if zone.end_ts < ohlcv.index.min() or zone.start_ts > ohlcv.index.max():
            continue
        x0 = max(_snap_ts_to_candle_x(zone.start_ts, ohlcv, side="left"), x_min)
        x1 = min(_snap_ts_to_candle_x(zone.end_ts, ohlcv, side="right"), x_max)
        if x1 <= x0:
            continue
        # Filled (invalidated) zones are faded more than active ones.
        alpha = 0.28 if zone.invalidated else 0.40
        color = _zone_color(zone)
        ax.add_patch(
            Rectangle(
                (x0, zone.zone_low),
                x1 - x0,
                zone.zone_high - zone.zone_low,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.6,
            )
        )


def _draw_entries(ax: Axes, signals: pd.DataFrame, ohlcv: pd.DataFrame) -> None:
    """Scatter entry markers at the containing candle's close."""
    for col, marker, color in (
        ("entry_long", "^", ENTRY_LONG_COLOR),
        ("entry_short", "v", ENTRY_SHORT_COLOR),
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
            # Prefer the candle that contains ts when resampling coarsens bars.
            if loc > 0 and ohlcv.index[loc] > ts:
                loc = loc - 1
            x = float(mdates.date2num(ohlcv.index[loc]))  # type: ignore[no-untyped-call]
            ys.append(float(ohlcv["close"].iloc[loc]))
            xs.append(x)
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
    *,
    htf: str = "M15",
    ltf: str = "M5",
    zone_kinds: tuple[str, ...] | None = None,
    max_zones: int | None = None,
) -> Figure:
    """Render PO3/IFVG signals on a candlestick chart.

    Zone geometry uses native HTF/LTF detection via
    :func:`~quant_rl.features.po3_config.build_fvg_zones_for_plot` (plot-only;
    agent causality unchanged).  Pass lookback in ``bars`` and set ``window``
    to the NY session so pre-session formations still define box starts.

    Parameters
    ----------
    bars : pd.DataFrame
        M1 OHLC bars (DatetimeIndex). May include lookback before ``window``.
    signals : pd.DataFrame
        Entry columns (``entry_long`` / ``entry_short``) for markers; zones
        are built from OHLC, not from these signal columns.
    window : tuple[Timestamp, Timestamp], optional
        Slice of the data to display (inclusive).  Zones still use full ``bars``.
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
    htf, ltf : str
        Native timeframes for plot-only FVG / IFVG zone detection.
    """
    _apply_style()

    if not len(bars):
        raise ValueError("No bars to plot after applying window.")

    # Zones from full spine (incl. lookback); candles/markers from display window.
    zones = build_fvg_zones_for_plot(bars, htf=htf, ltf=ltf)
    if zone_kinds is not None:
        zones = [z for z in zones if z.kind in zone_kinds]
    if window is not None:
        w0, w1 = window
        zones = [z for z in zones if z.end_ts >= w0 and z.start_ts <= w1]
    if max_zones is not None and len(zones) > max_zones:
        zones = sorted(zones, key=lambda z: z.end_ts - z.start_ts)[:max_zones]

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
    # Format in the bars' timezone (broker Etc/GMT-3), not forced UTC.
    tz = getattr(ohlcv.index, "tz", None)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator(), tz=tz)  # type: ignore[no-untyped-call]
    )
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_annotated_ny_session(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    day: pd.Timestamp,
    *,
    rr_ratio: float = 2.0,
    candle_tf: str = "5min",
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """NY-window pedagogy chart: Asian range, IFVG, entry, structural SL, RR TP.

    Does not draw scaled 3-TP. Zones are clipped to the NY window.
    """
    from quant_rl.data.session import ny_session_mask
    from quant_rl.eval.data_plots import _slice_ny_day

    plot_bars, lookback = _slice_ny_day(bars, day, lookback_bars=200)
    if plot_bars.empty:
        raise ValueError(f"No NY bars on {day}")
    day_feat = features.reindex(plot_bars.index)
    signals = pd.DataFrame(index=lookback.index)
    for col in ("entry_long", "entry_short"):
        if col in features.columns:
            signals[col] = features[col].reindex(lookback.index).fillna(0)
    fig = plot_fvg_signals(
        lookback,
        signals,
        window=(plot_bars.index[0], plot_bars.index[-1]),
        candle_tf=candle_tf,
        max_points=400,
        show_entries=True,
        dpi=dpi,
        zone_kinds=("ltf_ifvg",),
        max_zones=6,
    )
    ax = fig.axes[0]
    # Asian range + first entry SL/TP (RR, not 3-scale).
    for col, style in (("asian_high", "--"), ("asian_low", "--")):
        if col in features.columns:
            val = features[col].reindex(plot_bars.index).dropna()
            if not val.empty and np.isfinite(val.iloc[-1]):
                ax.axhline(float(val.iloc[-1]), color="#6a1b9a", linestyle=style, linewidth=0.9)
    ny_mask = ny_session_mask(pd.DatetimeIndex(plot_bars.index)).to_numpy()
    entry_idx = None
    side = 0
    for col, sgn in (("entry_long", 1), ("entry_short", -1)):
        if col not in day_feat.columns:
            continue
        hits = day_feat[col].fillna(0).astype(float).gt(0) & ny_mask
        if hits.any():
            entry_idx = plot_bars.index[hits.to_numpy()][0]
            side = sgn
            break
    if entry_idx is not None and side != 0:
        row = features.loc[entry_idx] if entry_idx in features.index else None
        px = float(plot_bars.loc[entry_idx, "close"])
        sl_col = "po3_manipulation_low" if side == 1 else "po3_manipulation_high"
        sl = float(row[sl_col]) if row is not None and sl_col in features.columns else np.nan
        if np.isfinite(sl) and sl != px:
            risk = abs(px - sl)
            tp = px + side * rr_ratio * risk
            ax.axhline(sl, color=SHORT_COLOR, linewidth=1.0, label="SL (manip)")
            ax.axhline(tp, color=LONG_COLOR, linewidth=1.0, label="TP (RR, not 3-scale)")
    ax.set_title(f"NY PO3/IFVG — {pd.Timestamp(day).date()} (gate diagnostic; RR TP)")
    if out_path:
        _save(fig, out_path, dpi)
    return fig
