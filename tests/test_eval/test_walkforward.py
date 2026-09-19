"""Unit tests for purged walk-forward cross-validation splits."""

from __future__ import annotations

import pytest

from quant_rl.evaluation.walkforward import (
    WFSplit,
    purged_walk_forward,
    purged_walk_forward_legacy,
)


def test_purged_walk_forward_yields_expected_splits() -> None:
    """n=1000, n_splits=5 yields 5 splits with non-overlapping indices."""
    splits = list(purged_walk_forward(n=1000, n_splits=5))
    assert len(splits) == 5

    for split in splits:
        assert isinstance(split, WFSplit)
        assert split.fold >= 0
        assert len(split.train_idx) > 0
        assert len(split.test_idx) > 0


def test_no_leakage_between_train_and_test() -> None:
    """train_idx.max() < test_idx.min() - purge_bars (no leakage)."""
    splits = list(purged_walk_forward(n=1000, n_splits=5))
    for split in splits:
        assert split.train_idx.max() < split.test_idx.min() - 60


def test_test_spans_are_chronological() -> None:
    """test_idx spans are in chronological order across folds."""
    splits = list(purged_walk_forward(n=1000, n_splits=5))
    for i in range(len(splits) - 1):
        assert splits[i].test_idx.min() < splits[i + 1].test_idx.min()


def test_test_folds_do_not_overlap() -> None:
    """Adjacent test folds are disjoint (TI-2 / F5 fix)."""
    splits = list(purged_walk_forward(n=1000, n_splits=5))
    for i in range(len(splits) - 1):
        assert splits[i].test_idx.max() < splits[i + 1].test_idx.min()


def test_label_horizon_widens_purge() -> None:
    """Larger label_horizon shortens the train window."""
    base = list(purged_walk_forward(n=1000, n_splits=3, label_horizon=0))
    wide = list(purged_walk_forward(n=1000, n_splits=3, label_horizon=30))
    assert len(base) == len(wide) == 3
    for a, b in zip(base, wide, strict=True):
        assert len(b.train_idx) <= len(a.train_idx)


def test_test_length_matches_expected() -> None:
    """Each fold gets an equal share of the total test budget."""
    n, n_splits, test_size = 1000, 5, 0.2
    splits = list(purged_walk_forward(n=n, n_splits=n_splits, test_size=test_size))
    expected_test_len = max(n_splits, int(n * test_size)) // n_splits
    for split in splits:
        assert len(split.test_idx) == expected_test_len


def test_empty_result_when_insufficient_data() -> None:
    """Too-small n yields no splits."""
    splits = list(purged_walk_forward(n=50, n_splits=5))
    assert len(splits) == 0


def test_legacy_overlap_documented() -> None:
    """Legacy stepper still overlaps (F5 regression fixture)."""
    with pytest.warns(DeprecationWarning):
        splits = list(purged_walk_forward_legacy(n=1000, n_splits=5))
    assert len(splits) >= 2
    assert splits[0].test_idx.max() >= splits[1].test_idx.min()
