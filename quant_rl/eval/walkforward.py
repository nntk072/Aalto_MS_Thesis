"""Purged + embargoed walk-forward cross-validation splits.

Adapted from the EA_SCALPER_XAUUSD oracle walk-forward pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd


@dataclass
class WFSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int


def purged_walk_forward(
    n: int,
    n_splits: int = 5,
    test_size: float = 0.2,
    purge_bars: int = 60,
    embargo_bars: int = 20,
) -> Generator[WFSplit, None, None]:
    """Yield purged + embargoed walk-forward splits.

    Parameters
    ----------
    n:
        Total number of samples.
    n_splits:
        Number of folds.
    test_size:
        Fraction of data per test fold.
    purge_bars:
        Bars removed from end of train to prevent leakage (e.g. feature window).
    embargo_bars:
        Bars removed from start of test (allow market to reset).
    """
    test_len = int(n * test_size)
    step = (n - test_len) // n_splits

    for fold in range(n_splits):
        test_start = step * fold + step
        test_end = min(test_start + test_len, n)

        train_end = test_start - embargo_bars
        train_start = 0

        # Purge: remove the last ``purge_bars`` from training
        train_end_purged = max(0, train_end - purge_bars)

        if train_end_purged <= train_start:
            continue

        train_idx = np.arange(train_start, train_end_purged)
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) == 0:
            continue

        yield WFSplit(train_idx=train_idx, test_idx=test_idx, fold=fold)
