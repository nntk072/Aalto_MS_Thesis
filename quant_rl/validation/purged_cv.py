"""Purged walk-forward cross-validation utilities (Lopez de Prado 2018).

The splitter guarantees that the ``purge_bars`` observations immediately
preceding each test block are excluded from training, preventing label
leakage from overlapping look-ahead windows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CVSplit:
    """One walk-forward fold as integer positions into the dataset.

    Attributes:
        fold: Zero-based fold number.
        train_start: First training position (inclusive).
        train_end: One past the last training position (exclusive).
        test_start: First test position (inclusive).
        test_end: One past the last test position (exclusive).
    """

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        """Training positions for this fold."""
        return range(self.train_start, self.train_end)

    @property
    def test_indices(self) -> range:
        """Test positions for this fold."""
        return range(self.test_start, self.test_end)


class PurgedWalkForward:
    """Expanding-window walk-forward splitter with a purge gap.

    The dataset is divided into ``n_splits + 1`` equal blocks. Fold *i*
    trains on everything before block *i+1* (minus the purge gap) and
    tests on block *i+1*, so every observation is tested exactly once and
    training data always precedes test data chronologically.
    """

    def __init__(
        self,
        n_samples: int,
        n_splits: int = 5,
        purge_bars: int = 8,
    ) -> None:
        """Validate inputs and pre-compute fold boundaries.

        Args:
            n_samples: Total number of rows in the dataset.
            n_splits: Number of test folds.
            purge_bars: Gap excluded from training right before each test
                block; must cover the longest label look-ahead window.

        Raises:
            ValueError: If arguments are too small to form the folds.
        """
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        min_required = (n_splits + 1) * 2
        if n_samples < min_required:
            raise ValueError(
                f"n_samples={n_samples} too small for {n_splits} folds "
                f"(need at least {min_required})"
            )
        if purge_bars < 0:
            raise ValueError("purge_bars must be non-negative")

        self.n_samples = n_samples
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self._block = n_samples // (n_splits + 1)

    def split(self) -> list[CVSplit]:
        """Return all folds in chronological order."""
        folds: list[CVSplit] = []
        for i in range(1, self.n_splits + 1):
            test_start = i * self._block
            test_end = test_start + self._block if i < self.n_splits else self.n_samples
            train_end = max(test_start - self.purge_bars, 1)
            folds.append(
                CVSplit(
                    fold=i - 1,
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
        return folds
