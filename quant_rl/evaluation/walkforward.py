"""Purged + embargoed walk-forward cross-validation splits.

Adapted from the EA_SCALPER_XAUUSD oracle walk-forward pattern.
Canonical home of the walk-forward splitter; the copy previously living in
``quant_rl/eval/walkforward.py`` was retired when the two evaluation stacks
were unified (see the W9 eval-metrics implementation plan, §1).

Default ``purged_walk_forward`` yields **non-overlapping** test folds (TI-2).
The pre-fix overlapping stepper is kept as ``purged_walk_forward_legacy``.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WFSplit:
    train_idx: np.ndarray[Any, Any]
    test_idx: np.ndarray[Any, Any]
    fold: int


def purged_walk_forward(
    n: int,
    n_splits: int = 5,
    test_size: float = 0.2,
    purge_bars: int = 60,
    embargo_bars: int = 20,
    label_horizon: int = 0,
) -> Generator[WFSplit, None, None]:
    """Yield expanding walk-forward splits with non-overlapping test folds.

    ``test_size`` is the fraction of ``n`` reserved as the *total* test budget
    across all folds (split into ``n_splits`` contiguous blocks at the end of
    the series). Train is always ``[0, test_start)`` after purge/embargo.

    Parameters
    ----------
    n:
        Total number of samples.
    n_splits:
        Number of folds.
    test_size:
        Fraction of ``n`` used as the combined test budget.
    purge_bars:
        Bars removed from the end of train (feature lookback).
    embargo_bars:
        Extra gap between train end and test start.
    label_horizon:
        Extra purge bars for label/episode horizon (added to ``purge_bars``).
    """
    if n_splits < 1 or n <= 0 or test_size <= 0:
        return

    total_test = max(n_splits, int(n * test_size))
    test_len = total_test // n_splits
    if test_len < 1:
        return

    purge_effective = purge_bars + max(0, int(label_horizon))
    test_region_start = n - n_splits * test_len
    min_train = purge_effective + embargo_bars + 1
    if test_region_start < min_train:
        return

    for fold in range(n_splits):
        test_start = test_region_start + fold * test_len
        test_end = test_start + test_len
        train_end = test_start - embargo_bars
        train_end_purged = max(0, train_end - purge_effective)
        if train_end_purged <= 0:
            continue

        train_idx = np.arange(0, train_end_purged)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) == 0:
            continue
        yield WFSplit(train_idx=train_idx, test_idx=test_idx, fold=fold)


def purged_walk_forward_legacy(
    n: int,
    n_splits: int = 5,
    test_size: float = 0.2,
    purge_bars: int = 60,
    embargo_bars: int = 20,
) -> Generator[WFSplit, None, None]:
    """Pre-fix overlapping stepper (F5). Do not use for new work."""
    warnings.warn(
        "purged_walk_forward_legacy has overlapping test folds (F5); "
        "use purged_walk_forward instead",
        DeprecationWarning,
        stacklevel=2,
    )
    test_len = int(n * test_size)
    step = (n - test_len) // n_splits

    for fold in range(n_splits):
        test_start = step * fold + step
        test_end = min(test_start + test_len, n)

        train_end = test_start - embargo_bars
        train_start = 0
        train_end_purged = max(0, train_end - purge_bars)

        if train_end_purged <= train_start:
            continue

        train_idx = np.arange(train_start, train_end_purged)
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) == 0:
            continue

        yield WFSplit(train_idx=train_idx, test_idx=test_idx, fold=fold)
