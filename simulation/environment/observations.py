"""Simulation-specific observation window and timeframe composition."""

from __future__ import annotations

from typing import Any

import numpy as np

from decisioning.observations import pad_observation_window

HTF_BRANCHES: tuple[tuple[str, str, str], ...] = (
    ("m5", "M5_", "seq_m5"),
    ("m15", "M15_", "seq_m15"),
    ("h1", "H1_", "seq_h1"),
)
DEFAULT_HTF_WINDOWS: dict[str, int] = {"m5": 24, "m15": 16, "h1": 12}


def split_obs_columns(
    names: list[str],
) -> tuple[np.ndarray[Any, Any], dict[str, np.ndarray[Any, Any]]]:
    """M1 keeps every column that is not a higher-timeframe prefix."""
    m1: list[int] = []
    grouped: dict[str, list[int]] = {key: [] for key, _, _ in HTF_BRANCHES}
    prefixes = {key: prefix for key, prefix, _ in HTF_BRANCHES}
    for i, name in enumerate(names):
        text = str(name)
        branch = next((key for key, prefix in prefixes.items() if text.startswith(prefix)), None)
        if branch is None:
            m1.append(i)
        else:
            grouped[branch].append(i)
    return (
        np.asarray(m1, dtype=np.int64),
        {key: np.asarray(idx, dtype=np.int64) for key, idx in grouped.items()},
    )


def run_starts(block: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Index of the first row of each run of constant columns. NaN matches NaN."""
    rows = int(block.shape[0]) if block.ndim == 2 else 0
    if rows == 0 or int(block.shape[1]) == 0:
        return np.zeros(0, dtype=np.int64)
    changed = np.ones(rows, dtype=bool)
    if rows > 1:
        prev = block[:-1]
        curr = block[1:]
        both_nan = np.isnan(prev) & np.isnan(curr)
        changed[1:] = np.any((prev != curr) & ~both_nan, axis=1)
    return np.flatnonzero(changed).astype(np.int64)


def closed_window(
    block: np.ndarray[Any, Any],
    starts: np.ndarray[Any, Any],
    end: int,
    length: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Last ``length`` closed runs at or before ``end``, oldest to newest.

    ``end`` is the decision row and is included. The row kept for the run that
    contains it is ``end`` itself, so a candle that first appears later is absent.
    A block with no columns is shape ``(length, 1)`` of zeros and a zero mask.
    """
    if block.ndim != 2 or int(block.shape[1]) == 0:
        empty = np.zeros((0, 1), dtype=np.float32)
        return pad_observation_window(empty, length)
    decision = int(end)
    width = int(block.shape[1])
    if decision < 0 or starts.size == 0:
        return pad_observation_window(np.zeros((0, width), dtype=np.float32), length)
    visible = starts[starts <= decision]
    if visible.size == 0:
        return pad_observation_window(np.zeros((0, width), dtype=np.float32), length)
    rows = np.empty(visible.size, dtype=np.int64)
    rows[:-1] = visible[1:] - 1
    rows[-1] = decision
    chosen = rows[-int(length) :]
    values = np.nan_to_num(np.asarray(block[chosen], dtype=np.float32), nan=0.0)
    return pad_observation_window(values, length)
