"""Persist decision-diversity metric results and render their figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.metrics.decision_diversity import (
    DiversityReport,
    DiversityThresholds,
    diversity_report,
)
from evaluation.plotting.diversity import render_decision_diversity_figures


def write_decision_artifacts(
    trades: pd.DataFrame,
    directory: Path,
    *,
    thresholds: DiversityThresholds | None = None,
    save_csv: bool = True,
    save_plots: bool = True,
    dpi: int = 100,
) -> DiversityReport:
    """Write the text report, count tables, and the diagnostic figures."""
    directory.mkdir(parents=True, exist_ok=True)
    report = diversity_report(trades, thresholds)
    if save_csv:
        (directory / "decision_diversity.txt").write_text(report.text)
        _write_shares(directory / "sl_family_counts.csv", report.sl_shares)
        _write_shares(directory / "tp_family_counts.csv", report.tp_shares)
        if not report.joint.empty:
            report.joint.to_csv(directory / "sl_tp_joint.csv")
        for name, table in report.by_conditioner.items():
            if not table.empty:
                table.to_csv(directory / f"sl_by_{name}.csv")
    if save_plots and not report.opens.empty:
        render_decision_diversity_figures(report, directory, dpi)
    return report


def _write_shares(path: Path, shares: pd.Series) -> None:
    if shares.empty:
        pd.DataFrame(columns=["family", "share"]).to_csv(path, index=False)
        return
    shares.rename("share").rename_axis("family").reset_index().to_csv(path, index=False)


__all__ = ["write_decision_artifacts"]
