"""Temporal distributions: sweep delay and holding time from a trade log.

Both reuse the generic :func:`compute_distribution_metrics` engine. One
observation means one qualifying sweep event (delay) or one completed
trade (holding time).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd

from .distributions import DistributionMetrics, compute_distribution_metrics

_OPEN_TYPES = {"open"}
_CLOSE_TYPES = {"close", "stop_close", "tp_close", "forced_close"}


def _entry_exit_pairs(trade_log: list[dict[str, Any]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Pair open entries with their matching close entries (FIFO by time).

    Args:
        trade_log: The environment's trade log.

    Returns:
        List of ``(entry_time, exit_time)`` pairs for completed trades.
    """
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    open_queue: deque[pd.Timestamp] = deque()
    for trade in trade_log:
        trade_type = str(trade.get("type", ""))
        tm = trade.get("time")
        if tm is None:
            continue
        ts = pd.Timestamp(tm)
        if trade_type in _OPEN_TYPES:
            open_queue.append(ts)
        elif trade_type in _CLOSE_TYPES and open_queue:
            entry = open_queue.popleft()
            pairs.append((entry, ts))
    return pairs


def holding_time_distribution(trade_log: list[dict[str, Any]]) -> DistributionMetrics:
    """Compute the distribution of per-trade holding times in seconds.

    Args:
        trade_log: Environment trade log with ``time`` and ``type`` keys.

    Returns:
        Distribution metrics over holding seconds; count=0 / NaN fields
        when no completed trades are paired.
    """
    pairs = _entry_exit_pairs(trade_log)
    holding_s = [float((exit_ts - entry_ts).total_seconds()) for entry_ts, exit_ts in pairs]
    return compute_distribution_metrics(holding_s)


def sweep_delay_distribution(trade_log: list[dict[str, Any]]) -> DistributionMetrics:
    """Compute the distribution of entry sweep delays in seconds.

    Args:
        trade_log: Environment trade log with ``sweep_delay_s`` on opens.

    Returns:
        Distribution over finite ``sweep_delay_s`` values.
    """
    delays = [
        float(trade["sweep_delay_s"])
        for trade in trade_log
        if trade.get("type") == "open" and trade.get("sweep_delay_s") is not None
    ]
    return compute_distribution_metrics(delays)


def _completed_trades(trade_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each close with its open entry, carrying PnL, direction and level.

    Returns a list of opened-trade records: ``pnl``, ``direction`` and
    ``level_type`` from the open entry, matched FIFO with its close.
    """
    completed: list[dict[str, Any]] = []
    open_queue: deque[dict[str, Any]] = deque()
    for trade in trade_log:
        trade_type = str(trade.get("type", ""))
        if trade_type in _OPEN_TYPES:
            open_queue.append(trade)
        elif trade_type in _CLOSE_TYPES and open_queue and "pnl" in trade:
            opening = open_queue.popleft()
            completed.append(
                {
                    "pnl": float(trade["pnl"]),
                    "direction": opening.get("direction"),
                    "level_type": opening.get("level_type"),
                }
            )
    return completed


def conditional_pnl_distributions(
    trade_log: list[dict[str, Any]],
) -> dict[str, DistributionMetrics]:
    """Compute PnL distributions grouped by strategy regime (PLAN 9, WP-D).

    Groups: ``overall``, ``long``, ``short`` and, where the trade log
    exposes them, ``london`` and ``asian`` (from ``level_type``).

    Args:
        trade_log: Environment trade log.

    Returns:
        Mapping from group name to its PnL distribution metrics.
    """
    trades = _completed_trades(trade_log)
    long_pnls = [t["pnl"] for t in trades if t.get("direction") == 1]
    short_pnls = [t["pnl"] for t in trades if t.get("direction") == -1]
    london_pnls = [t["pnl"] for t in trades if "london" in str(t.get("level_type", ""))]
    asian_pnls = [t["pnl"] for t in trades if "asian" in str(t.get("level_type", ""))]

    return {
        "overall": compute_distribution_metrics([t["pnl"] for t in trades]),
        "long": compute_distribution_metrics(long_pnls),
        "short": compute_distribution_metrics(short_pnls),
        "london": compute_distribution_metrics(london_pnls),
        "asian": compute_distribution_metrics(asian_pnls),
    }
