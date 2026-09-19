"""Temporal distributions: sweep delay, holding time, and regime PnL groups."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from .distributions import DistributionMetrics, compute_distribution_metrics

_OPEN_TYPES = {"open"}
_CLOSE_TYPES = {"close", "stop_close", "tp_close", "forced_close", "eod_close"}


def _as_records(trade_log: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    """Normalise a trade log to a list of row dicts."""
    if isinstance(trade_log, pd.DataFrame):
        if trade_log.empty:
            return []
        return cast(list[dict[str, Any]], trade_log.to_dict("records"))
    return list(trade_log)


def _entry_exit_pairs(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair each open with the next close (same state machine as plot pairing)."""
    records = _as_records(trade_log)
    if not records:
        return []
    # Prefer chronological bar order when present (matches engine append order).
    if any("bar" in r and r.get("bar") is not None for r in records):
        records = sorted(
            records,
            key=lambda r: float("inf") if r.get("bar") is None else float(r["bar"]),
        )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pending: dict[str, Any] | None = None
    for row in records:
        row_type = str(row.get("type", ""))
        if row_type in _OPEN_TYPES:
            pending = row
        elif row_type in _CLOSE_TYPES and pending is not None:
            pairs.append((pending, row))
            pending = None
    return pairs


def holding_time_distribution(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> DistributionMetrics:
    """Compute the distribution of per-trade holding times in seconds.

    Args:
        trade_log: Environment trade log with ``time`` and ``type`` keys.

    Returns:
        Distribution metrics over holding seconds; count=0 / NaN fields
        when no completed trades are paired.
    """
    pairs = _entry_exit_pairs(trade_log)
    holding_s: list[float] = []
    for open_row, close_row in pairs:
        if open_row.get("time") is None or close_row.get("time") is None:
            continue
        holding_s.append(
            float(
                (pd.Timestamp(close_row["time"]) - pd.Timestamp(open_row["time"])).total_seconds()
            )
        )
    return compute_distribution_metrics(holding_s)


def sweep_delay_distribution(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> DistributionMetrics:
    """Compute the distribution of entry sweep delays in seconds.

    Args:
        trade_log: Environment trade log with ``sweep_delay_s`` on opens.

    Returns:
        Distribution over finite ``sweep_delay_s`` values.
    """
    delays = [
        float(trade["sweep_delay_s"])
        for trade in _as_records(trade_log)
        if str(trade.get("type", "")) == "open" and trade.get("sweep_delay_s") is not None
    ]
    return compute_distribution_metrics(delays)


def _completed_trade_records(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    """Pair each close with its open, carrying PnL, direction and level."""
    completed: list[dict[str, Any]] = []
    for open_row, close_row in _entry_exit_pairs(trade_log):
        if "pnl" not in close_row or close_row.get("pnl") is None:
            continue
        try:
            pnl = float(close_row["pnl"])
        except (TypeError, ValueError):
            continue
        completed.append(
            {
                "pnl": pnl,
                "direction": open_row.get("direction"),
                "level_type": open_row.get("level_type"),
            }
        )
    return completed


def conditional_pnl_groups(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, list[float]]:
    """Return raw PnL lists grouped by regime for box plots."""
    trades = _completed_trade_records(trade_log)
    return {
        "overall": [t["pnl"] for t in trades],
        "long": [t["pnl"] for t in trades if t.get("direction") == 1],
        "short": [t["pnl"] for t in trades if t.get("direction") == -1],
        "london": [t["pnl"] for t in trades if "london" in str(t.get("level_type", ""))],
        "asian": [t["pnl"] for t in trades if "asian" in str(t.get("level_type", ""))],
    }


def conditional_pnl_distributions(
    trade_log: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, DistributionMetrics]:
    """Compute PnL distributions grouped by strategy regime.

    Groups: ``overall``, ``long``, ``short`` and, where the trade log
    exposes them, ``london`` and ``asian`` (from ``level_type``).

    Args:
        trade_log: Environment trade log.

    Returns:
        Mapping from group name to its PnL distribution metrics.
    """
    groups = conditional_pnl_groups(trade_log)
    return {name: compute_distribution_metrics(vals) for name, vals in groups.items()}
