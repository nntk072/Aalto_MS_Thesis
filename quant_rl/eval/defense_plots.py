"""Defense-pack EDA: coverage/gaps, spread, 24h activity, tails, pre-RL checks."""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from quant_rl.data.activity import activity_series
from quant_rl.data.session import ny_session_mask
from quant_rl.eval.plots import EQUITY_COLOR, LONG_COLOR, SHORT_COLOR, _apply_style, _save

log = logging.getLogger(__name__)

_TECH_PCA = ("rsi", "atr", "ema_9_dist", "macd", "ret_1", "adx", "bb_pct_b")
_EVENT_FLAGS = (
    ("in_ifvg", ("price_in_ifvg_bull", "price_in_ifvg_bear")),
    ("manipulation", ("po3_manipulation_active",)),
    ("distribution", ("po3_distribution",)),
    ("entry", ("entry_long", "entry_short")),
)


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
        for col in df.columns:
            if str(col).endswith(f"_{name}"):
                return str(col)
    return None


def _flag_union(features: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    acc: pd.Series | None = None
    for name in names:
        col = _first_col(features, (name,))
        if col is None:
            continue
        part = features[col].fillna(0).astype(float).gt(0)
        acc = part if acc is None else (acc | part)
    if acc is None:
        return pd.Series(False, index=features.index)
    return acc


def _ny_mask(index: pd.DatetimeIndex) -> NDArray[np.bool_]:
    return np.asarray(ny_session_mask(index), dtype=bool)


def _normal_qq(ax: Any, vals: NDArray[np.float64]) -> None:
    """QQ against N(0,1) using stdlib inverse CDF (no scipy)."""
    x = np.sort(np.asarray(vals, dtype=float))
    n = int(x.size)
    if n < 2:
        return
    p = (np.arange(1, n + 1) - 0.5) / n
    nd = NormalDist()
    theo = np.array([nd.inv_cdf(float(pi)) for pi in p], dtype=float)
    loc = float(np.mean(x))
    scale = float(np.std(x)) or 1.0
    ax.scatter(theo, (x - loc) / scale, s=8, color=EQUITY_COLOR, alpha=0.7)
    ax.plot(theo, theo, color="#333333", linewidth=0.8)


def plot_coverage_with_gaps(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Daily bar counts plus missing minutes vs a 1440-bar weekday."""
    _apply_style()
    idx = pd.DatetimeIndex(bars.index)
    days = idx.normalize()
    full = days.value_counts().sort_index()
    expected = pd.Series(
        [0.0 if d.dayofweek >= 5 else 1440.0 for d in full.index],
        index=full.index,
    )
    missing = (expected - full.astype(float)).clip(lower=0.0)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax0.bar(full.index, np.asarray(full.to_numpy(), dtype=float), width=0.8, color=EQUITY_COLOR)
    ax0.set_ylabel("Bars / day")
    ax0.set_title("Daily coverage (full M1)", fontweight="bold")
    ax0.grid(True, axis="y")
    ax1.bar(
        missing.index, np.asarray(missing.to_numpy(), dtype=float), width=0.8, color=SHORT_COLOR
    )
    ax1.set_ylabel("Missing minutes")
    ax1.set_title("Gaps vs 1440 weekday minutes (weekends expected 0)", fontweight="bold")
    ax1.grid(True, axis="y")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_spread_regime(
    bars: pd.DataFrame,
    point_size: float = 0.01,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Spread in price units: NY vs off-hours."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    if "spread" not in bars.columns:
        ax.text(0.5, 0.5, "No spread column", ha="center", va="center", transform=ax.transAxes)
    else:
        px = bars["spread"].astype(float) * float(point_size)
        ny = _ny_mask(pd.DatetimeIndex(bars.index))
        for mask, label, color in (
            (ny, "NY 16:30–23:00", EQUITY_COLOR),
            (~ny, "Off-hours", SHORT_COLOR),
        ):
            vals = px.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            ax.hist(vals.to_numpy(), bins=60, density=True, alpha=0.55, label=label, color=color)
        ax.legend(fontsize=8)
    ax.set_xlabel("Spread (price units)")
    ax.set_ylabel("Density")
    ax.set_title("Spread regime: NY vs off-hours", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_activity_24h(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Mean tick-count activity by minute-of-day; NY window shaded."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    act = activity_series(bars)
    idx = pd.DatetimeIndex(bars.index)
    if act is None:
        ax.text(0.5, 0.5, "No tickvol activity", ha="center", va="center", transform=ax.transAxes)
    else:
        minute = idx.hour * 60 + idx.minute
        mean_act = act.groupby(minute).mean()
        ax.plot(
            np.asarray(mean_act.index),
            np.asarray(mean_act.to_numpy(), dtype=float),
            color=EQUITY_COLOR,
            linewidth=1.2,
        )
        ax.axvspan(16 * 60 + 30, 23 * 60, color=LONG_COLOR, alpha=0.12, label="NY")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 24 * 60)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(120))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _p: f"{int(x) // 60:02d}:{int(x) % 60:02d}")
        )
    ax.set_xlabel("Time of day (broker tz)")
    ax.set_ylabel("Mean tick count")
    ax.set_title("24h activity (tickvol; CFD vol ignored)", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_ny_return_tails(
    bars: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """NY M1 log returns: log-y histogram + normal QQ."""
    _apply_style()
    idx = pd.DatetimeIndex(bars.index)
    ny = bars.loc[_ny_mask(idx)]
    close = ny["close"].astype(float)
    ratio = close / close.shift(1)
    rets = pd.Series(np.log(np.asarray(ratio.to_numpy(), dtype=float)), index=ratio.index)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    if rets.empty:
        ax0.text(0.5, 0.5, "No returns", ha="center", va="center", transform=ax0.transAxes)
    else:
        vals = np.asarray(rets.to_numpy(), dtype=float) * 100.0
        ax0.hist(vals, bins=80, color=EQUITY_COLOR, alpha=0.8)
        ax0.set_yscale("log")
        kurt = float(np.asarray(pd.Series(vals).kurtosis(), dtype=float).reshape(-1)[0])
        ax0.set_title(f"NY log returns (excess kurtosis={kurt:.1f})", fontweight="bold")
        _normal_qq(ax1, np.asarray(vals, dtype=np.float64))
        ax1.set_title("Normal QQ", fontweight="bold")
    ax0.set_xlabel("Log return (%)")
    ax0.set_ylabel("Count (log)")
    ax0.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_curated_feature_hists(
    features: pd.DataFrame,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Clipped histograms for a small observation subset (NY bars)."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    frame = features.loc[_ny_mask(idx)] if _ny_mask(idx).any() else features
    names = ("rsi", "atr_norm_price", "ret_1", "price_in_ifvg_bull", "po3_distribution")
    cols = [c for n in names if (c := _first_col(frame, (n,))) is not None]
    n = max(1, len(cols))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes_arr = np.atleast_1d(axes)
    for ax, col in zip(axes_arr, cols, strict=False):
        vals = frame[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        uniq = set(np.unique(np.round(vals, 8)))
        if uniq <= {0.0, 1.0}:
            ax.bar([0, 1], [1.0 - vals.mean(), vals.mean()], color=[SHORT_COLOR, LONG_COLOR])
        else:
            lo, hi = np.percentile(vals, [1, 99])
            clipped = vals[(vals >= lo) & (vals <= hi)] if hi > lo else vals
            ax.hist(clipped, bins=40, color=EQUITY_COLOR, alpha=0.85)
        ax.set_title(col, fontsize=8)
        ax.grid(True, axis="y")
    fig.suptitle("Curated NY observation columns", fontweight="bold")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _split_mask(
    index: pd.DatetimeIndex, train_end: str, test_start: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    te = pd.Timestamp(train_end)
    ts = pd.Timestamp(test_start)
    if index.tz is not None:
        if te.tzinfo is None:
            te = te.tz_localize(index.tz)
        if ts.tzinfo is None:
            ts = ts.tz_localize(index.tz)
    te = te + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return np.asarray(index <= te, dtype=bool), np.asarray(index >= ts, dtype=bool)


def plot_event_rates(
    features: pd.DataFrame,
    train_end: str,
    test_start: str,
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Percent of NY minutes with IFVG / PO3 / entry flags, train vs test."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    ny = _ny_mask(idx)
    train_m, test_m = _split_mask(idx, train_end, test_start)
    labels: list[str] = []
    train_r: list[float] = []
    test_r: list[float] = []
    for lab, names in _EVENT_FLAGS:
        flag = _flag_union(features, names)
        labels.append(lab)
        tr = flag.loc[ny & train_m]
        te = flag.loc[ny & test_m]
        train_r.append(float(tr.mean() * 100.0) if len(tr) else 0.0)
        test_r.append(float(te.mean() * 100.0) if len(te) else 0.0)
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, train_r, width=0.36, color=LONG_COLOR, label="Train")
    ax.bar(x + 0.18, test_r, width=0.36, color=SHORT_COLOR, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of NY minutes")
    ax.set_title("PO3 / IFVG occupancy (sparsity)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_ifvg_occupancy_monthly(
    features: pd.DataFrame,
    test_start: str = "2026-01-01",
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Monthly % of NY minutes inside IFVG, with locked-OOS line."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    flag = _flag_union(features, ("price_in_ifvg_bull", "price_in_ifvg_bear"))
    ny = _ny_mask(idx)
    flag_ny = flag.loc[ny]
    monthly = flag_ny.resample("MS").mean() * 100.0
    fig, ax = plt.subplots(figsize=(12, 4))
    if monthly.empty:
        ax.text(0.5, 0.5, "No IFVG flags", ha="center", va="center", transform=ax.transAxes)
    else:
        xs = pd.DatetimeIndex(monthly.index)
        ax.plot(
            xs,
            np.asarray(monthly.to_numpy(), dtype=float),
            color=EQUITY_COLOR,
            marker="o",
            linewidth=1.3,
        )
        ts = pd.Timestamp(test_start)
        if xs.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(xs.tz)
        ax.axvline(ts, color=SHORT_COLOR, linestyle="--", label="Locked OOS")  # type: ignore[arg-type]
        ax.legend(fontsize=8)
    ax.set_ylabel("% of NY minutes")
    ax.set_title("Monthly IFVG occupancy", fontweight="bold")
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def plot_ifvg_event_study(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    train_end: str,
    horizon: int = 60,
    out_path: Path | str | None = None,
    dpi: int = 150,
    seed: int = 0,
) -> Figure:
    """Purged train-only forward returns after IFVG+distribution vs random NY."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    train_m, _ = _split_mask(idx, train_end, "2099-01-01")
    ny = _ny_mask(idx)
    ifvg = _flag_union(features, ("price_in_ifvg_bull", "price_in_ifvg_bear"))
    dist = _flag_union(features, ("po3_distribution",))
    setup = np.asarray(ifvg) & np.asarray(dist) & ny & train_m
    onset = np.asarray(setup & np.concatenate([[False], ~setup[:-1]]), dtype=bool)
    close = np.asarray(
        bars["close"].reindex(features.index).astype(float).to_numpy(), dtype=np.float64
    )
    event = _mean_path(close, onset, horizon, purge=horizon)
    rng = np.random.default_rng(seed)
    ny_train = np.where(ny & train_m)[0]
    n_evt = int(onset.sum())
    ctrl = np.zeros(len(close), dtype=np.bool_)
    if n_evt > 0 and len(ny_train) > 0:
        picks = rng.choice(ny_train, size=min(n_evt, len(ny_train)), replace=False)
        ctrl[picks] = True
    random_path = _mean_path(close, ctrl, horizon, purge=horizon)
    fig, ax = plt.subplots(figsize=(10, 4))
    h = np.arange(horizon + 1)
    ax.plot(h, event, color=LONG_COLOR, label="IFVG + distribution")
    ax.plot(h, random_path, color=SHORT_COLOR, linestyle="--", label="Random NY")
    ax.axhline(0.0, color="#888888", linewidth=0.7)
    ax.legend(fontsize=8)
    ax.set_xlabel("Bars after onset")
    ax.set_ylabel("Mean close-to-close return")
    ax.set_title("Train event study (purged, NY)", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig


def _mean_path(
    close: NDArray[np.float64], onset: NDArray[np.bool_], horizon: int, *, purge: int
) -> NDArray[np.float64]:
    """Mean relative return path; skip onsets within *purge* bars of a prior pick."""
    acc = np.zeros(horizon + 1, dtype=float)
    n_used = 0
    last = -purge - 1
    hits = np.where(onset)[0]
    for i in hits:
        if i - last <= purge:
            continue
        end = i + horizon
        if end >= len(close) or not np.isfinite(close[i]) or close[i] == 0:
            continue
        path = close[i : end + 1] / close[i] - 1.0
        if path.shape[0] != horizon + 1 or not np.isfinite(path).all():
            continue
        acc += path
        n_used += 1
        last = i
    if n_used == 0:
        return np.zeros(horizon + 1, dtype=float)
    return acc / n_used


def plot_pca_ifvg(
    features: pd.DataFrame,
    train_end: str,
    max_points: int = 8000,
    out_path: Path | str | None = None,
    dpi: int = 150,
    seed: int = 0,
) -> Figure:
    """2-D PCA of technical columns, coloured by in-IFVG (NY train subsample)."""
    _apply_style()
    idx = pd.DatetimeIndex(features.index)
    train_m, _ = _split_mask(idx, train_end, "2099-01-01")
    ny = _ny_mask(idx)
    cols = [c for n in _TECH_PCA if (c := _first_col(features, (n,))) is not None]
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(cols) < 2:
        ax.text(0.5, 0.5, "Need ≥2 technical columns", ha="center", va="center")
        if out_path:
            _save(fig, out_path, dpi)
        return fig
    frame = features.loc[ny & train_m, cols].replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        ax.text(0.5, 0.5, "Empty PCA frame", ha="center", va="center")
        if out_path:
            _save(fig, out_path, dpi)
        return fig
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed)
    mat = frame.to_numpy(dtype=float)
    mat = (mat - mat.mean(axis=0)) / np.clip(mat.std(axis=0), 1e-8, None)
    _u, _s, vt = np.linalg.svd(mat, full_matrices=False)
    pcs = mat @ vt[:2].T
    color_flag = np.asarray(
        _flag_union(features, ("price_in_ifvg_bull", "price_in_ifvg_bear")).reindex(frame.index)
    )
    ax.scatter(
        pcs[~color_flag, 0],
        pcs[~color_flag, 1],
        s=6,
        alpha=0.25,
        color="#90a4ae",
        label="not in IFVG",
    )
    ax.scatter(
        pcs[color_flag, 0],
        pcs[color_flag, 1],
        s=8,
        alpha=0.45,
        color=LONG_COLOR,
        label="in IFVG",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Technical PCA coloured by IFVG", fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path, dpi)
    return fig
