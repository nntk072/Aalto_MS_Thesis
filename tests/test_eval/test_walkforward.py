"""Unit tests for purged walk-forward cross-validation splits."""

from __future__ import annotations

from quant_rl.evaluation.walkforward import WFSplit, purged_walk_forward


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


def test_test_length_matches_expected() -> None:
    """len(test_idx) == test_len for each fold."""
    splits = list(purged_walk_forward(n=1000, n_splits=5))
    expected_test_len = int(1000 * 0.2)
    for split in splits:
        assert len(split.test_idx) == expected_test_len


def test_empty_result_when_insufficient_data() -> None:
    """Too-small n yields no splits."""
    splits = list(purged_walk_forward(n=50, n_splits=5))
    assert len(splits) == 0
