"""Multi-seed aggregation, report tables and run-report building.

Consumes :class:`~quant_rl.evaluation.metrics.PerformanceMetrics` instances
only — this is the canonical report layer after the two evaluation stacks
were unified (W9 eval-metrics plan, §1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import PerformanceMetrics, sweep_delay_breakdown

# Metric rows shown in the summary / comparison tables, in display order.
_TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("Sharpe", "sharpe"),
    ("Sortino", "sortino"),
    ("Calmar", "calmar"),
    ("Max Drawdown", "max_drawdown"),
    ("Total Return", "total_return_pct"),
    ("Profit Factor", "profit_factor"),
    ("Expectancy", "expectancy"),
    ("Win Rate", "win_rate"),
    ("Total Trades", "n_trades"),
    ("Total PnL", "total_pnl"),
    ("Avg Trade PnL", "avg_trade"),
    ("Max Consec Loss", "max_consec_loss"),
    ("Turnover", "turnover"),
    ("Breach Count", "breach_count"),
    ("Breach Rate", "breach_rate"),
)


def aggregate_seeds(results: Sequence[PerformanceMetrics]) -> pd.DataFrame:
    """Aggregate per-seed :class:`PerformanceMetrics` into a summary DataFrame.

    Args:
        results: One ``PerformanceMetrics`` instance per seed/run.

    Returns:
        DataFrame with one row per seed and one column per metric.

    Raises:
        ValueError: If *results* is empty or contains a non-metrics object.
    """
    rows: list[dict[str, Any]] = []
    for seed, m in enumerate(results):
        if not isinstance(m, PerformanceMetrics):
            raise ValueError(
                f"result at index {seed} is {type(m).__name__}; expected PerformanceMetrics"
            )
        row: dict[str, Any] = {"seed": seed}
        for _, attr in _TABLE_ROWS:
            row[attr] = getattr(m, attr)
        rows.append(row)
    if not rows:
        raise ValueError("aggregate_seeds requires at least one result")
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    """Print a describe()-style summary of an aggregated seed DataFrame."""
    print("\n=== Multi-Seed Report ===")
    print(df.describe().round(4).to_string())
    print()


def _format_value(label: str, value: float) -> str:
    if label in ("Total Trades", "Max Consec Loss", "Breach Count"):
        return str(int(value))
    if label in ("Max Drawdown", "Total Return", "Win Rate", "Breach Rate"):
        return f"{value * 100:.2f}%"
    if label == "Turnover":
        return f"{value:.6f}"
    if label == "Total PnL":
        return f"{value:.2f}"
    return f"{value:.4f}"


def build_summary_table(m: PerformanceMetrics) -> str:
    """Return a human-readable text table of a single PerformanceMetrics."""
    lines = [
        "=" * 42,
        f"  {'Metric':<22} {'Value':>12}",
        "-" * 42,
    ]
    for label, attr in _TABLE_ROWS:
        lines.append(f"  {label:<22} {_format_value(label, float(getattr(m, attr))):>12}")
    lines.append("=" * 42)
    return "\n".join(lines)


def build_comparison_table(
    train_m: PerformanceMetrics, test_m: PerformanceMetrics | None = None
) -> str:
    """Return a two-column Train vs Test comparison table."""
    has_test = test_m is not None
    width = 62 if has_test else 42
    sep = "=" * width
    mid = "-" * width

    def row(label: str, train_val: str, test_val: str = "") -> str:
        base = f"  {label:<22} {train_val:>14}"
        return base + (f"  {test_val:>14}" if has_test else "")

    header = f"  {'Metric':<22} {'Train':>14}" + (f"  {'Test':>14}" if has_test else "")

    lines = [sep, header, mid]
    for label, attr in _TABLE_ROWS:
        train_val = _format_value(label, float(getattr(train_m, attr)))
        test_val = _format_value(label, float(getattr(test_m, attr))) if test_m else ""
        lines.append(row(label, train_val, test_val))
    lines.append(sep)
    return "\n".join(lines)


def save_metrics_json(m: PerformanceMetrics, path: Path | str) -> None:
    """Persist a PerformanceMetrics instance to a JSON file."""
    Path(path).write_text(json.dumps(asdict(m), indent=2))


def build_run_report(
    metrics: PerformanceMetrics,
    trade_log: Sequence[dict[str, Any]] | None = None,
    **context: Any,
) -> dict[str, Any]:
    """Build the JSON-serialisable report block for one evaluation run.

    Merges all :class:`PerformanceMetrics` fields with the Sweep Delay
    breakdown of the run's trade log, plus any caller context (e.g. the
    run name or split configuration).  This replaces the hand-rolled
    per-script dict merging that used to be duplicated between
    ``train_rl.py`` and ``compare_encoders.py``.

    Args:
        metrics: Metrics returned by :func:`run_episode`.
        trade_log: The environment's ``trade_log`` list, if available.
        **context: Extra key/value pairs merged into the report.

    Returns:
        A dict safe to pass to ``json.dumps``.
    """
    report: dict[str, Any] = asdict(metrics)
    report["sweep_delay"] = sweep_delay_breakdown(trade_log or [])
    report.update(context)
    return report
