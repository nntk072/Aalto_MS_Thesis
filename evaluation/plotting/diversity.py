"""Matplotlib charts for decision-diversity metric results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics.decision_diversity import DiversityReport

_RR_PCTS: tuple[int, ...] = (10, 25, 50, 75, 90, 95)


def render_decision_diversity_figures(report: DiversityReport, directory: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _bar(report.sl_shares, directory / "sl_family.png", "SL family", dpi)
    _bar(report.tp_shares, directory / "tp_family.png", "TP family", dpi)
    _rr_hist(report.rr, directory / "planned_rr_hist.png", dpi)
    _heatmap(report.opens, directory / "sl_tp_rank_heatmap.png", dpi)
    _manipulation(
        report.by_conditioner.get("manipulation", pd.DataFrame()),
        directory / "sl_by_manipulation.png",
        dpi,
    )
    _staircase(report.opens, "stop_u", "sl_index", directory / "stop_u_vs_index.png", dpi)
    _staircase(report.opens, "target_u", "tp_index", directory / "target_u_vs_index.png", dpi)
    plt.close("all")


def _bar(shares: pd.Series, path: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    if shares.empty:
        ax.set_title(title)
    else:
        ax.bar(list(shares.index.astype(str)), shares.to_numpy(dtype=float))
        ax.set_ylabel("share")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _rr_hist(rr: pd.Series, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("Planned RR")
    if not rr.empty:
        values = rr.to_numpy(dtype=float)
        ax.hist(values, bins=min(20, max(5, values.size // 2)))
        for pct in _RR_PCTS:
            ax.axvline(float(np.quantile(values, pct / 100.0)), color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _heatmap(opens: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("SL rank x TP rank")
    sl = pd.to_numeric(opens["sl_rank"], errors="coerce")
    tp = pd.to_numeric(opens["tp_rank"], errors="coerce")
    pair = pd.DataFrame({"sl": sl, "tp": tp}).dropna()
    edges = np.linspace(0.0, 1.0, 6)
    if not pair.empty:
        hist, _, _ = np.histogram2d(pair["sl"], pair["tp"], bins=edges)
        ax.imshow(hist.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("SL rank")
    ax.set_ylabel("TP rank")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _manipulation(table: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("SL family by manipulation")
    if not table.empty:
        shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        shares.plot(kind="bar", stacked=True, ax=ax, legend=True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _staircase(opens: pd.DataFrame, x_name: str, y_name: str, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title(f"{x_name} to {y_name}")
    if x_name in opens.columns and y_name in opens.columns:
        x = pd.to_numeric(opens[x_name], errors="coerce")
        y = pd.to_numeric(opens[y_name], errors="coerce")
        keep = x.notna() & y.notna()
        ax.scatter(x[keep], y[keep], s=12)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


__all__ = ["render_decision_diversity_figures"]
