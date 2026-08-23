"""Tests for the purged walk-forward splitter."""

from __future__ import annotations

import pytest

from quant_rl.validation import PurgedWalkForward


@pytest.mark.unit
class TestPurgedWalkForward:
    def test_produces_requested_folds(self) -> None:
        # Act
        folds = PurgedWalkForward(n_samples=1_200, n_splits=5, purge_bars=8).split()

        # Assert
        assert len(folds) == 5
        assert [f.fold for f in folds] == [0, 1, 2, 3, 4]

    def test_purge_gap_prevents_leakage(self) -> None:
        # Arrange
        purge = 8
        folds = PurgedWalkForward(n_samples=1_200, n_splits=5, purge_bars=purge).split()

        # Act / Assert — training ends at least `purge` bars before testing
        for f in folds:
            assert f.test_start - f.train_end >= purge

    def test_train_always_precedes_test(self) -> None:
        # Arrange
        folds = PurgedWalkForward(n_samples=1_000, n_splits=4, purge_bars=5).split()

        # Act / Assert
        for f in folds:
            assert f.train_start < f.train_end <= f.test_start < f.test_end

    def test_expanding_window_grows_monotonically(self) -> None:
        # Arrange
        folds = PurgedWalkForward(n_samples=1_000, n_splits=4, purge_bars=5).split()

        # Act
        train_sizes = [len(f.train_indices) for f in folds]

        # Assert
        assert train_sizes == sorted(train_sizes)
        assert all(size > 0 for size in train_sizes)

    def test_last_fold_covers_dataset_end(self) -> None:
        # Arrange
        n = 903  # not divisible on purpose
        folds = PurgedWalkForward(n_samples=n, n_splits=5, purge_bars=6).split()

        # Assert
        assert folds[-1].test_end == n

    def test_invalid_arguments_raise(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="n_splits"):
            PurgedWalkForward(n_samples=100, n_splits=0)
        with pytest.raises(ValueError, match="too small"):
            PurgedWalkForward(n_samples=10, n_splits=5)
        with pytest.raises(ValueError, match="purge_bars"):
            PurgedWalkForward(n_samples=100, n_splits=2, purge_bars=-1)
