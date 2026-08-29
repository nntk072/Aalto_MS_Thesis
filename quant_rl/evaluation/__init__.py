"""Evaluation utilities shared across baselines and RL agents.

This package is the single canonical evaluation stack for the project:
metrics (``PerformanceMetrics``/``compute_metrics``/``calculate_metrics``),
the episode runner, purged walk-forward splitting, and multi-seed
reporting all live here.
"""

from .metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    LEGACY_M1_PERIODS_PER_YEAR,
    PerformanceMetrics,
    calculate_metrics,
    compute_metrics,
    max_drawdown,
    sweep_delay_breakdown,
)
from .report import (
    aggregate_seeds,
    build_comparison_table,
    build_run_report,
    build_summary_table,
    print_report,
    save_metrics_json,
)
from .runner import run_episode
from .walkforward import WFSplit, purged_walk_forward

__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "LEGACY_M1_PERIODS_PER_YEAR",
    "PerformanceMetrics",
    "WFSplit",
    "aggregate_seeds",
    "build_comparison_table",
    "build_run_report",
    "build_summary_table",
    "calculate_metrics",
    "compute_metrics",
    "max_drawdown",
    "print_report",
    "purged_walk_forward",
    "run_episode",
    "save_metrics_json",
    "sweep_delay_breakdown",
]
