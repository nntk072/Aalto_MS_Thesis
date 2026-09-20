"""Data and feature EDA charts for thesis figure packs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quant_rl.data.session import ny_session_mask
from quant_rl.eval.plots import EQUITY_COLOR, LONG_COLOR, SHORT_COLOR, _apply_style, _save

log = logging.getLogger(__name__)

# Preferred continuous columns for observation histograms (z-scored in pipeline).
_HIST_CONTINUOUS = (
    "rsi",
    "atr",
    "ema_9_dist",
    "macd",
    "ret_1",
    "adx",
    "bb_pct_b",
    "realized_vol",
    "atr_norm_price",
)
# Binary / event-style flags shown as rate bars rather than 40-bin hists.
_HIST_FLAGS = (
    "entry_long",
    "entry_short",
    "price_in_ifvg_bull",
    "price_in_ifvg_bear",
    "fvg_in_bull",
    "fvg_in_bear",
    "po3_distribution",
    "po3_manipulation_active",
)

_EVENT_SIGNAL_NAMES = (
    "entry_long",
    "entry_short",
    "fvg_in_bull",
    "fvg_in_bear",
    "htf_fvg_bullish",
    "htf_fvg_bearish",
)
_STATE_SIGNAL_NAMES = (
    "price_in_ifvg_bull",
    "price_in_ifvg_bear",
)


def plot_coverage_calendar(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Daily bar counts with NY vs full-day breakdown."""
    _apply_style()
    idx = pd.DatetimeIndex(bars.index)
    days = idx.normalize()
    full = days.value_counts().sort_index()
    ny_mask = ny_session_mask(idx)
    ny_counts = (
        days[ny_mask.to_numpy()].value_counts().sort_index().reindex(full.index).fillna(0)
        if ny_mask.any()
        else pd.Series(0.0, index=full.index)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(
        full.index,
        np.asarray(full.to_numpy(), dtype=float),
        width=0.8,
        color="#90caf9",
        label="All bars",
    )
    ax.bar(
        ny_counts.index,
        np.asarray(ny_counts.to_numpy(), dtype=float),
        width=0.8,
        color=EQUITY_COLOR,
        label="NY (16:30–23:00)",
    )
    ax.set_title("Daily Bar Coverage", fontweight="bold")
    ax.set_ylabel("Bars per day")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    ax.grid(True, axis="y")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_log_return_dist(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """M1 log-return histogram with normal overlay (fat-tail check)."""
    _apply_style()
    close = bars["close"].astype(float)
    log_rets = np.log(close / close.shift(1))
    rets = cast(pd.Series, log_rets).replace([np.inf, -np.inf], np.nan).dropna()
    rets = rets[np.isfinite(rets)]

    fig, ax = plt.subplots(figsize=(10, 4))
    if rets.empty:
        ax.text(0.5, 0.5, "No returns", ha="center", va="center", transform=ax.transAxes)
    else:
        vals = rets.to_numpy() * 100.0
        bins = min(80, max(20, len(vals) // 50))
        ax.hist(vals, bins=bins, density=True, color=EQUITY_COLOR, alpha=0.75, label="Log returns")
        mu, sigma = float(vals.mean()), float(vals.std(ddof=1) or 1e-9)
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 300)
        dens = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        ax.plot(x, dens, color=SHORT_COLOR, linewidth=1.5, label="Normal fit")
        ax.legend(fontsize=8)
    ax.set_xlabel("Log return (%)")
    ax.set_ylabel("Density")
    ax.set_title("M1 Log-Return Distribution", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _volume_series(df: pd.DataFrame) -> pd.Series | None:
    """Return a volume-like series: ``volume``, else ``tickvol``, else ``vol``."""
    for col in ("volume", "tickvol", "vol"):
        if col in df.columns:
            return df[col].astype(float)
    return None


def _minute_to_hhmm(minute: float) -> str:
    m = int(minute) % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def plot_session_profile(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Mean range and volume by minute-of-day inside the NY window."""
    _apply_style()
    idx = pd.DatetimeIndex(bars.index)
    mask = ny_session_mask(idx)
    ny = bars.loc[mask.to_numpy()].copy()
    vol_s = _volume_series(ny) if not ny.empty else None
    n_axes = 2 if vol_s is not None else 1
    fig, axes_arr = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes), sharex=True)
    axes = np.atleast_1d(axes_arr)

    if ny.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No NY bars", ha="center", va="center", transform=ax.transAxes)
    else:
        minute = idx[mask.to_numpy()].hour * 60 + idx[mask.to_numpy()].minute
        ny = ny.assign(_minute=minute)
        rng = (ny["high"] - ny["low"]).groupby(ny["_minute"]).mean()
        axes[0].plot(rng.index, rng.values, color=LONG_COLOR, linewidth=1.3)
        axes[0].set_ylabel("Mean range")
        axes[0].set_title("NY Session Profile (16:30–23:00)", fontweight="bold")
        axes[0].grid(True)
        if vol_s is not None:
            vol = vol_s.groupby(ny["_minute"]).mean()
            axes[1].plot(vol.index, vol.values, color=EQUITY_COLOR, linewidth=1.3)
            vol_label = (
                "tickvol" if "tickvol" in ny.columns and "volume" not in ny.columns else "volume"
            )
            axes[1].set_ylabel(f"Mean {vol_label}")
            axes[1].grid(True)
        axes[-1].set_xlabel("Time of day")
        for ax in axes:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _p: _minute_to_hhmm(x)))
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_split_overlay(
    bars: pd.DataFrame,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Close price with train/test split shading."""
    _apply_style()
    close = bars["close"].astype(float)
    # Downsample for readability on long histories.
    step = max(1, len(close) // 5000)
    plot_close = close.iloc[::step]
    te = pd.Timestamp(train_end)
    ts = pd.Timestamp(test_start)
    idx = pd.DatetimeIndex(plot_close.index)
    if idx.tz is not None:
        if te.tzinfo is None:
            te = te.tz_localize(idx.tz)
        if ts.tzinfo is None:
            ts = ts.tz_localize(idx.tz)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        plot_close.index,
        np.asarray(plot_close.to_numpy(), dtype=float),
        color="#333333",
        linewidth=0.8,
        label="Close",
    )
    ax.axvspan(
        cast(Any, plot_close.index.min()),
        cast(Any, te),
        color=LONG_COLOR,
        alpha=0.12,
        label="Train",
    )
    ax.axvspan(
        cast(Any, ts),
        cast(Any, plot_close.index.max()),
        color=SHORT_COLOR,
        alpha=0.12,
        label="Test",
    )
    ax.axvline(cast(Any, te), color=LONG_COLOR, linestyle="--", linewidth=1.0)
    ax.axvline(cast(Any, ts), color=SHORT_COLOR, linestyle="--", linewidth=1.0)
    ax.set_title("Train / Test Split on Price", fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _select_hist_columns(numeric: pd.DataFrame, max_cols: int) -> list[str]:
    """Curated continuous + flag columns present in ``numeric``."""
    cols: list[str] = []
    for name in _HIST_CONTINUOUS:
        if name in numeric.columns and numeric[name].notna().any():
            cols.append(name)
    for name in _HIST_FLAGS:
        if name in numeric.columns and numeric[name].notna().any():
            cols.append(name)
    if not cols:
        cols = [c for c in numeric.columns if numeric[c].notna().any()][:max_cols]
    return cols[:max_cols]


def plot_feature_histograms(
    features: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
    max_cols: int = 12,
) -> Figure:
    """Grid of histograms for curated observation feature columns (NY bars)."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    ny_mask = ny_session_mask(idx)
    frame = features.loc[ny_mask.to_numpy()] if ny_mask.any() else features
    numeric = frame.select_dtypes(include=[np.number])
    cols = _select_hist_columns(numeric, max_cols)
    n = max(1, len(cols))
    nrows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    flag_set = set(_HIST_FLAGS)
    for i, ax in enumerate(axes_flat):
        if i >= len(cols):
            ax.set_axis_off()
            continue
        col = cols[i]
        vals = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(col, fontsize=9)
            continue
        arr = vals.to_numpy(dtype=float)
        n_obs = len(arr)
        pct_zero = float(np.mean(np.isclose(arr, 0.0))) * 100.0
        med = float(np.median(arr))
        if col in flag_set or set(np.unique(arr)).issubset({0.0, 1.0}):
            rate = float(arr.mean())
            ax.bar([0, 1], [1.0 - rate, rate], color=[SHORT_COLOR, LONG_COLOR], alpha=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_ylabel("Fraction")
        else:
            lo, hi = np.percentile(arr, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(arr.min()), float(arr.max())
            clipped = arr[(arr >= lo) & (arr <= hi)]
            if clipped.size == 0:
                clipped = arr
            ax.hist(clipped, bins=40, color=EQUITY_COLOR, alpha=0.8)
        ax.set_title(f"{col}\nn={n_obs} 0%={pct_zero:.1f} med={med:.3g}", fontsize=8)
        ax.grid(True, axis="y")
    fig.suptitle(
        "Feature Distributions (NY bars, z-scored observations)",
        fontweight="bold",
    )
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _signal_columns(features: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    """Match exact names plus ``{TF}_``-prefixed variants present in features."""
    found: list[str] = []
    for name in names:
        if name in features.columns:
            found.append(name)
        for col in features.columns:
            if col.endswith(f"_{name}") and col not in found:
                found.append(col)
    return found


def plot_signal_counts(
    features: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Daily counts of PO3/FVG signals: events vs state flags in two panels."""
    _apply_style()
    event_cols = _signal_columns(features, _EVENT_SIGNAL_NAMES)
    state_cols = _signal_columns(features, _STATE_SIGNAL_NAMES)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    days = pd.DatetimeIndex(features.index).normalize()

    def _draw(ax: Any, cols: list[str], title: str) -> None:
        if not cols:
            ax.text(
                0.5,
                0.5,
                "No matching signal columns",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            for col in cols:
                daily = features[col].fillna(0).astype(float).gt(0).groupby(days).sum()
                ax.plot(
                    daily.index,
                    np.asarray(daily.to_numpy(), dtype=float),
                    label=col,
                    linewidth=1.2,
                )
            ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        ax.grid(True)

    _draw(axes[0], event_cols, "Daily Signal Counts — events (entries / FVG flags)")
    _draw(axes[1], state_cols, "Daily Signal Counts — state (minutes inside IFVG)")
    axes[1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_signals_for_day(
    features: pd.DataFrame,
    day: pd.Timestamp,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """NY-session timeline of event/state flags for one calendar day."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    day_norm = pd.Timestamp(day).normalize()
    if idx.tz is not None and day_norm.tzinfo is None:
        day_norm = day_norm.tz_localize(idx.tz)
    day_mask = idx.normalize() == day_norm
    ny_mask = ny_session_mask(idx)
    mask = day_mask & ny_mask.to_numpy()
    day_feat = features.loc[mask]
    event_cols = _signal_columns(features, _EVENT_SIGNAL_NAMES)
    state_cols = _signal_columns(features, _STATE_SIGNAL_NAMES)
    cols = event_cols + state_cols
    fig, ax = plt.subplots(figsize=(12, max(3, 0.4 * max(1, len(cols)))))
    if day_feat.empty or not cols:
        ax.text(0.5, 0.5, "No NY signals", ha="center", va="center", transform=ax.transAxes)
    else:
        times = pd.DatetimeIndex(day_feat.index)
        for i, col in enumerate(cols):
            hits = day_feat[col].fillna(0).astype(float).gt(0)
            if not hits.any():
                continue
            ys = np.full(hits.sum(), i, dtype=float)
            ax.scatter(times[hits.to_numpy()], ys, s=12, label=col, alpha=0.8)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # type: ignore[no-untyped-call]
    ax.set_title(f"NY Signals — {day_norm.date()}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def write_data_eda(
    out_dir: Path | str,
    bars: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    train_end: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
    dpi: int = 150,
    po3_sample: bool = True,
) -> Path:
    """Write the thesis data/EDA figure pack under ``out_dir``.

    Args:
        out_dir: Destination folder (created if needed).
        bars: Primary M1 OHLCV bars.
        features: Optional feature frame aligned to ``bars``.
        train_end: Inclusive train boundary for split overlay.
        test_start: Inclusive test boundary for split overlay.
        dpi: PNG resolution.
        po3_sample: When True and signal columns exist, write PO3 charts.

    Returns:
        Path to the EDA directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if bars.empty:
        log.warning("write_data_eda: empty bars — skipping")
        return out

    plot_coverage_calendar(bars, out_path=out / "coverage_calendar.png", dpi=dpi)
    plot_log_return_dist(bars, out_path=out / "log_return_dist.png", dpi=dpi)
    plot_session_profile(bars, out_path=out / "session_profile.png", dpi=dpi)

    if train_end is not None and test_start is not None:
        plot_split_overlay(
            bars, train_end, test_start, out_path=out / "train_test_split.png", dpi=dpi
        )

    if features is not None and not features.empty:
        plot_feature_histograms(features, out_path=out / "feature_histograms.png", dpi=dpi)
        plot_signal_counts(features, out_path=out / "signal_counts.png", dpi=dpi)
        _write_signals_by_day(features, out / "signals_by_day", dpi=dpi)
        if po3_sample:
            _write_po3_charts(bars, features, out, dpi=dpi)

    log.info("Data EDA charts written to %s", out)
    return out


def _ny_session_days(bars: pd.DataFrame) -> list[pd.Timestamp]:
    """Unique calendar days that have at least one NY-session bar."""
    idx = pd.DatetimeIndex(bars.index)
    mask = ny_session_mask(idx)
    if not mask.any():
        return []
    days = pd.DatetimeIndex(idx[mask.to_numpy()]).normalize().unique().sort_values()
    return list(days)


def _write_signals_by_day(features: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """One NY-session signal timeline PNG per calendar day."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for day in _ny_session_days(features):
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        try:
            plot_signals_for_day(features, day, out_path=out_dir / f"{day_str}.png", dpi=dpi)
        except Exception as exc:
            log.warning("signals_by_day %s skipped: %s", day_str, exc)


def _slice_ny_day(
    bars: pd.DataFrame,
    day: pd.Timestamp,
    *,
    lookback_bars: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (plot_bars for NY window, lookback_bars for signal detection)."""
    idx = pd.DatetimeIndex(bars.index)
    day_norm = pd.Timestamp(day).normalize()
    if idx.tz is not None and day_norm.tzinfo is None:
        day_norm = day_norm.tz_localize(idx.tz)
    day_mask = idx.normalize() == day_norm
    ny_mask = ny_session_mask(idx).to_numpy()
    plot_bars = bars.loc[day_mask & ny_mask]
    if plot_bars.empty:
        return plot_bars, plot_bars
    start_iloc = int(bars.index.get_indexer(pd.Index([plot_bars.index[0]]))[0])
    lb_start = max(0, start_iloc - lookback_bars)
    lookback = bars.iloc[lb_start : start_iloc + len(plot_bars)]
    return plot_bars, lookback


def _slice_calendar_day(
    bars: pd.DataFrame,
    day: pd.Timestamp,
    *,
    lookback_bars: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full calendar-day bars, lookback+day spine for zone geometry).

    Plotting shows every M1 bar on the broker calendar day so off-session FVGs
    remain visible. Session filtering for trading stays in the env/features.
    """
    idx = pd.DatetimeIndex(bars.index)
    day_norm = pd.Timestamp(day).normalize()
    if idx.tz is not None and day_norm.tzinfo is None:
        day_norm = day_norm.tz_localize(idx.tz)
    plot_bars = bars.loc[idx.normalize() == day_norm]
    if plot_bars.empty:
        return plot_bars, plot_bars
    start_iloc = int(bars.index.get_indexer(pd.Index([plot_bars.index[0]]))[0])
    lb_start = max(0, start_iloc - lookback_bars)
    end_iloc = start_iloc + len(plot_bars)
    lookback = bars.iloc[lb_start:end_iloc]
    return plot_bars, lookback


def _write_po3_charts(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    out: Path,
    dpi: int,
) -> None:
    """Overview sample + one PO3/FVG PNG per calendar day (full-day candles)."""
    signal_cols = [
        c
        for c in features.columns
        if any(k in c for k in ("fvg", "ifvg", "entry_long", "entry_short", "htf_fvg"))
    ]
    if not signal_cols:
        return
    try:
        from quant_rl.eval.po3_plots import plot_fvg_signals
        from quant_rl.features.po3_config import detect_po3_entries
    except Exception as exc:
        log.warning("PO3 charts skipped (import): %s", exc)
        return

    # Still key off NY session days (thesis trading days), but draw the full day.
    days = _ny_session_days(bars)
    if not days:
        return

    try:
        # One full-spine detect so per-day charts share zone low/high columns.
        signals = detect_po3_entries(bars, htf="M15", primary_tf="M5")
        for col in ("entry_long", "entry_short"):
            if col in features.columns:
                signals[col] = features[col].reindex(signals.index).fillna(0)
    except Exception as exc:
        log.warning("PO3 detect_po3_entries skipped: %s", exc)
        return

    by_day = out / "po3_fvg_by_day"
    by_day.mkdir(parents=True, exist_ok=True)

    def _plot_day(day: pd.Timestamp, path: Path) -> None:
        plot_bars, lookback = _slice_calendar_day(bars, day)
        if plot_bars.empty:
            return
        day_signals = signals.reindex(lookback.index).fillna(0)
        # Full broker day (~24h). 5min → ≤288 candles; allow headroom.
        plot_fvg_signals(
            lookback,
            day_signals,
            window=(plot_bars.index[0], plot_bars.index[-1]),
            candle_tf="5min",
            max_points=400,
            out_path=path,
            dpi=dpi,
        )

    # Overview = most recent trading day (full calendar day).
    try:
        _plot_day(days[-1], out / "po3_fvg_sample.png")
    except Exception as exc:
        log.warning("PO3 sample chart skipped: %s", exc)

    for day in days:
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        try:
            _plot_day(day, by_day / f"{day_str}.png")
        except Exception as exc:
            log.warning("PO3 day chart %s skipped: %s", day_str, exc)
