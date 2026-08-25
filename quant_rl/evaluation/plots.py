"""Distribution visualisation for the evaluation layer (PLAN 9, WP-E).

Renders under the matplotlib Agg backend so the plots can be produced
headless in CI. Uses the distribution engines from this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402


def plot_pnl_histogram(
    pnls: Sequence[float],
    path: str | Path,
    bins: int = 30,
) -> str | Path:
    """Render and save a PnL histogram.

    Args:
        pnls: Per-trade PnL observations.
        path: Destination PNG path.
        bins: Number of histogram bins.

    Returns:
        The path the figure was written to.
    """
    path = Path(path)
    values = np.asarray(pnls, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))
    if values.size:
        ax.hist(values, bins=bins, color="#4c72b0", edgecolor="white")
        ax.axvline(0.0, color="#c44e52", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Trade PnL")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_pnl_box_by_regime(
    groups: dict[str, Any],
    path: str | Path,
) -> str | Path | None:
    """Render and save a box plot of PnL per strategy regime.

    ``groups`` maps a regime name to a sequence of PnL values (e.g. the
    output of :func:`conditional_pnl_distributions` or raw lists).

    Args:
        groups: Mapping of regime name to its PnL observations.
        path: Destination PNG path.

    Returns:
        The path the figure was written to.
    """
    path = Path(path)
    box_data = []
    labels = []
    for name in ("overall", "long", "short", "london", "asian"):
        if name not in groups:
            continue
        values = np.asarray(groups[name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        box_data.append(values)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(6, 4))
    if box_data:
        ax.boxplot(box_data, tick_labels=labels)
        ax.axhline(0.0, color="#c44e52", linestyle="-", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center")
    ax.set_ylabel("Trade PnL")
    ax.set_title("PnL by Regime")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
