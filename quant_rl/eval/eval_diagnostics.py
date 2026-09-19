"""Write bootstrap CIs and baseline reliability diagrams into a split dir."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_rl.eval.plots import _pair_trades, _save
from quant_rl.evaluation.bootstrap_ci import metrics_with_ci
from quant_rl.evaluation.calibration import calibration_report, plot_reliability_diagram

log = logging.getLogger(__name__)


def write_bootstrap_cis(
    split_dir: Path,
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    n_boot: int = 500,
    level: float = 0.95,
) -> dict[str, Any]:
    """Compute and persist bootstrap CIs for Sharpe / Sortino / MDD / win-rate."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    eq = np.asarray(equity.astype(float).to_numpy(), dtype=np.float64)
    pairs = _pair_trades(trades) if not trades.empty else []
    pnls = np.asarray(
        [float(c["pnl"]) for _, c in pairs if pd.notna(c.get("pnl"))],
        dtype=np.float64,
    )
    cis = metrics_with_ci(eq, pnls, n_boot=n_boot, level=level)
    payload = {
        name: {
            "estimate": ci.estimate,
            "lower": ci.lower,
            "upper": ci.upper,
            "level": ci.level,
        }
        for name, ci in cis.items()
    }
    (split_dir / "bootstrap_ci.json").write_text(json.dumps(payload, indent=2))
    return payload


def write_calibration_artifacts(
    split_dir: Path,
    trades: pd.DataFrame,
    *,
    dpi: int = 150,
    save_plots: bool = True,
) -> dict[str, Any] | None:
    """Write a regime-rate reliability diagram for closed trades.

    Predicted probability for each trade is the empirical win rate of its
    direction (long vs short) on the same split — a baseline calibration
    check that does not require policy logits.
    """
    pairs = _pair_trades(trades) if not trades.empty else []
    if len(pairs) < 2:
        return None

    records: list[tuple[int, float]] = []
    for open_row, close_row in pairs:
        if pd.isna(close_row.get("pnl")):
            continue
        direction = int(open_row.get("direction", 0) or 0)
        if direction not in (1, -1):
            continue
        records.append((direction, 1.0 if float(close_row["pnl"]) > 0 else 0.0))
    if len(records) < 2:
        return None

    long_out = [y for d, y in records if d == 1]
    short_out = [y for d, y in records if d == -1]
    long_wr = float(np.mean(long_out)) if long_out else 0.5
    short_wr = float(np.mean(short_out)) if short_out else 0.5

    preds = np.asarray([long_wr if d == 1 else short_wr for d, _ in records], dtype=np.float64)
    outcomes = np.asarray([y for _, y in records], dtype=np.float64)
    report = calibration_report(preds, outcomes, n_bins=min(10, max(3, len(records) // 5)))

    payload = {
        "ece": report.ece,
        "brier": report.brier,
        "n": int(report.bin_counts.sum()),
        "long_win_rate": long_wr,
        "short_win_rate": short_wr,
        "summary": report.summary(),
    }
    split_dir = Path(split_dir)
    (split_dir / "calibration.json").write_text(json.dumps(payload, indent=2))

    if save_plots:
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_reliability_diagram(report, ax=ax)
        _save(fig, split_dir / "reliability.png", dpi)

    return payload
