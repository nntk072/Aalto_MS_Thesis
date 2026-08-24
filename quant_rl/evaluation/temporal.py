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
