"""Probability calibration diagnostics for the RL policy (Chain D).

A policy whose value head / action logits are interpreted as a
probability of a winning trade should be *calibrated*: among signals it
labels 0.7, roughly 70% should win. This module bins predicted
probabilities against realised outcomes and reports a reliability
diagram (matplotlib), expected calibration error (ECE) and the Brier
score. Run it on zero-latency AND delay-injected backtests (Chain C)
— miscalibration under latency is exactly the failure mode worth seeing.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.axes
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationReport:
    """Binned reliability data + summary scores."""

    bin_centers: NDArray[np.float64]  # mean predicted prob per bin
    bin_win_rates: NDArray[np.float64]  # observed win rate per bin
    bin_counts: NDArray[np.int64]  # samples per bin
    ece: float  # expected calibration error (weighted |gap|)
    brier: float  # Brier score (lower is better, 0 = perfect)

    def summary(self) -> str:
        return (
            f"ECE={self.ece:.4f}  Brier={self.brier:.4f}  "
            f"bins={len(self.bin_centers)}  n={int(self.bin_counts.sum())}"
        )


def calibration_report(
    predicted_probs: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> CalibrationReport:
    """Bin predictions into ``n_bins`` equal-width bins over [0, 1].

    Parameters
    ----------
    predicted_probs:
        Policy-implied probability of a winning trade, in [0, 1].
    outcomes:
        1.0 for winning trades, 0.0 for losers (same order as predictions).
    n_bins:
        Number of equal-width bins. Bins with zero samples are dropped.
    """
    p = np.asarray(predicted_probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.size != y.size:
        raise ValueError("predicted_probs and outcomes must have the same length")
    if p.size == 0:
        return CalibrationReport(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            float("nan"),
            float("nan"),
        )

    p = np.clip(p, 0.0, 1.0)  # type: ignore[assignment]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    centers, win_rates, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        centers.append(float(p[mask].mean()))
        win_rates.append(float(y[mask].mean()))
        counts.append(cnt)

    centers_arr = np.asarray(centers)
    rates_arr = np.asarray(win_rates)
    counts_arr = np.asarray(counts, dtype=np.int64)

    # ECE: sample-weighted mean absolute gap between confidence and accuracy
    gaps = np.abs(rates_arr - centers_arr) if centers_arr.size else np.array([])
    ece = float((gaps * counts_arr).sum() / counts_arr.sum()) if counts_arr.sum() else float("nan")
    brier = float(((p - y) ** 2).mean())

    return CalibrationReport(centers_arr, rates_arr, counts_arr, ece, brier)


def plot_reliability_diagram(
    report: CalibrationReport, ax: matplotlib.axes.Axes | None = None
) -> matplotlib.axes.Axes:
    """Draw the reliability diagram; returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    assert ax is not None

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    if report.bin_centers.size:
        ax.bar(
            report.bin_centers,
            report.bin_win_rates,
            width=0.9 / max(1, report.bin_centers.size),
            alpha=0.6,
            edgecolor="k",
            label="observed win rate",
        )
    ax.set_xlabel("Predicted probability of win")
    ax.set_ylabel("Observed win rate")
    ax.set_title(report.summary())
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return ax
