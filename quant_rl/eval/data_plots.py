"""Data and feature EDA charts for thesis figure packs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quant_rl.data.session import ny_session_mask
from quant_rl.eval.plots import EQUITY_COLOR, LONG_COLOR, SHORT_COLOR, _apply_style, _save

log = logging.getLogger(__name__)


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
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
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
        if "volume" in ny.columns:
            vol = ny["volume"].groupby(ny["_minute"]).mean()
            axes[1].plot(vol.index, vol.values, color=EQUITY_COLOR, linewidth=1.3)
            axes[1].set_ylabel("Mean volume")
        else:
            axes[1].text(
                0.5, 0.5, "No volume column", ha="center", va="center", transform=axes[1].transAxes
            )
        axes[1].set_xlabel("Minute of day")
        axes[1].grid(True)
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


def plot_feature_histograms(
    features: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
    max_cols: int = 12,
) -> Figure:
    """Grid of histograms for observation feature columns."""
    _apply_style()
    numeric = features.select_dtypes(include=[np.number])
    cols = [c for c in numeric.columns if numeric[c].notna().any()][:max_cols]
    n = max(1, len(cols))
    nrows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i >= len(cols):
            ax.set_axis_off()
            continue
        col = cols[i]
        vals = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(vals.to_numpy(), bins=40, color=EQUITY_COLOR, alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.grid(True, axis="y")
    fig.suptitle("Feature Distributions (observation columns)", fontweight="bold")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_signal_counts(
    features: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Daily counts of PO3/FVG/IFVG signal flags when present."""
    _apply_style()
    candidates = [
        c
        for c in (
            "entry_long",
            "entry_short",
            "fvg_in_bull",
            "fvg_in_bear",
            "htf_fvg_bullish",
            "htf_fvg_bearish",
            "price_in_ifvg_bull",
            "price_in_ifvg_bear",
        )
        if c in features.columns
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    if not candidates:
        ax.text(
            0.5,
            0.5,
            "No PO3/FVG signal columns in features",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        days = pd.DatetimeIndex(features.index).normalize()
        for col in candidates:
            daily = features[col].fillna(0).astype(float).gt(0).groupby(days).sum()
            ax.plot(
                daily.index,
                np.asarray(daily.to_numpy(), dtype=float),
                label=col,
                linewidth=1.2,
            )
        ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Daily Signal Counts", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(True)
    fig.autofmt_xdate()
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
        po3_sample: When True and signal columns exist, write a PO3 chart sample.

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
        if po3_sample:
            _maybe_po3_sample(bars, features, out / "po3_fvg_sample.png", dpi=dpi)

    log.info("Data EDA charts written to %s", out)
    return out


def _maybe_po3_sample(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """Write a short PO3/FVG sample chart when signal columns exist."""
    signal_cols = [
        c
        for c in features.columns
        if any(k in c for k in ("fvg", "ifvg", "entry_long", "entry_short", "htf_fvg"))
    ]
    if not signal_cols:
        return
    try:
        from quant_rl.eval.po3_plots import plot_fvg_signals

        # Last ~3 NY sessions (~3 * 390 bars) or 2000 bars, whichever smaller.
        n = min(len(bars), 2000)
        sample_bars = bars.iloc[-n:]
        sample_sig = features.reindex(sample_bars.index)
        plot_fvg_signals(
            sample_bars,
            sample_sig,
            candle_tf="5min",
            max_points=800,
            out_path=out_path,
            dpi=dpi,
        )
    except Exception as exc:
        log.warning("PO3 sample chart skipped: %s", exc)
