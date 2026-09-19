"""Distribution visualisation: PnL histogram with VaR/CVaR and regime box plots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from quant_rl.evaluation.distributions import (
    DistributionMetrics,
    compute_distribution_metrics,
)
from quant_rl.evaluation.temporal import (
    conditional_pnl_distributions,
    conditional_pnl_groups,
    holding_time_distribution,
    sweep_delay_distribution,
)

from .plots import LONG_COLOR, SHORT_COLOR, _apply_style, _save


def plot_pnl_histogram(
    pnls: Sequence[float],
    path: str | Path | None = None,
    bins: int = 30,
    dpi: int = 150,
    metrics: DistributionMetrics | None = None,
) -> Figure:
    """Render a PnL histogram with optional VaR5 / CVaR5 markers.

    Args:
        pnls: Per-trade PnL observations.
        path: Destination PNG path (optional).
        bins: Number of histogram bins.
        dpi: PNG resolution.
        metrics: Precomputed distribution summary; computed when omitted.

    Returns:
        The matplotlib Figure.
    """
    _apply_style()
    values = np.asarray(pnls, dtype=float)
    values = values[np.isfinite(values)]
    dist = metrics or compute_distribution_metrics(values.tolist())

    fig, ax = plt.subplots(figsize=(10, 4))
    if values.size:
        ax.hist(values, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.85)
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, label="Zero")
        if np.isfinite(dist.var_05):
            ax.axvline(
                dist.var_05,
                color=SHORT_COLOR,
                linestyle="-",
                linewidth=1.4,
                label=f"VaR5 ({dist.var_05:.1f})",
            )
        if np.isfinite(dist.cvar_05):
            ax.axvline(
                dist.cvar_05,
                color="#b71c1c",
                linestyle=":",
                linewidth=1.4,
                label=f"CVaR5 ({dist.cvar_05:.1f})",
            )
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No trade PnL data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Trade PnL (USD)")
    ax.set_ylabel("Frequency")
    title = "PnL Distribution"
    if dist.count:
        title += f"  (n={dist.count}, mean={dist.mean:.1f}, med={dist.median:.1f})"
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y")
    fig.tight_layout()
    if path:
        _save(fig, path, dpi)
    return fig


def plot_pnl_box_by_regime(
    groups: dict[str, Any],
    path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Render a box plot of PnL per strategy regime.

    ``groups`` maps a regime name to a sequence of PnL values or a
    :class:`DistributionMetrics` (raw values preferred).

    Args:
        groups: Mapping of regime name to PnL observations.
        path: Destination PNG path (optional).
        dpi: PNG resolution.

    Returns:
        The matplotlib Figure.
    """
    _apply_style()
    box_data: list[np.ndarray[Any, Any]] = []
    labels: list[str] = []
    for name in ("overall", "long", "short", "london", "asian"):
        if name not in groups:
            continue
        raw = groups[name]
        if isinstance(raw, DistributionMetrics):
            continue  # need raw values; skip metrics-only entries
        values = np.asarray(raw, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        box_data.append(values)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(8, 4))
    if box_data:
        ax.boxplot(box_data, tick_labels=labels)
        ax.axhline(0.0, color=SHORT_COLOR, linestyle="-", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Trade PnL (USD)")
    ax.set_title("PnL by Regime", fontweight="bold")
    ax.grid(True, axis="y")
    fig.tight_layout()
    if path:
        _save(fig, path, dpi)
    return fig


def plot_holding_time_hist(
    holding_s: Sequence[float],
    path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Histogram of per-trade holding times in minutes."""
    _apply_style()
    values = np.asarray(holding_s, dtype=float)
    values = values[np.isfinite(values)]
    minutes = values / 60.0

    fig, ax = plt.subplots(figsize=(10, 4))
    if minutes.size:
        bins = min(40, max(8, int(minutes.size // 3)))
        ax.hist(minutes, bins=bins, color=LONG_COLOR, edgecolor="white", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "No holding-time data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Holding time (minutes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Holding Time Distribution", fontweight="bold")
    ax.grid(True, axis="y")
    fig.tight_layout()
    if path:
        _save(fig, path, dpi)
    return fig


def write_distribution_artifacts(
    split_dir: Path,
    trades: pd.DataFrame,
    *,
    dpi: int = 150,
    save_plots: bool = True,
) -> dict[str, Any]:
    """Write ``pnl_distribution.json`` and related distribution charts.

    Args:
        split_dir: Output directory for one train/test split.
        trades: Trade log DataFrame.
        dpi: PNG resolution.
        save_plots: When False, only write the JSON summary.

    Returns:
        The distribution payload written to JSON.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    groups = conditional_pnl_groups(trades)
    group_metrics = conditional_pnl_distributions(trades)
    overall = group_metrics["overall"]
    hold = holding_time_distribution(trades)
    sweep = sweep_delay_distribution(trades)

    payload: dict[str, Any] = {
        "pnl": {k: asdict(v) for k, v in group_metrics.items()},
        "holding_time_s": asdict(hold),
        "sweep_delay_s": asdict(sweep),
    }
    (split_dir / "pnl_distribution.json").write_text(json.dumps(payload, indent=2))

    if save_plots and not trades.empty:
        from quant_rl.eval.plots import _pair_trades

        plot_pnl_histogram(
            groups["overall"],
            path=split_dir / "pnl_dist_var.png",
            dpi=dpi,
            metrics=overall,
        )
        plot_pnl_box_by_regime(groups, path=split_dir / "pnl_regime_box.png", dpi=dpi)
        pairs_hold = [
            float((pd.Timestamp(c["time"]) - pd.Timestamp(o["time"])).total_seconds())
            for o, c in _pair_trades(trades)
            if pd.notna(o.get("time")) and pd.notna(c.get("time"))
        ]
        plot_holding_time_hist(pairs_hold, path=split_dir / "hold_time_dist.png", dpi=dpi)

    return payload
