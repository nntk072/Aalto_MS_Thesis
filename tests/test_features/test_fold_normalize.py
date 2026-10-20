"""Regression tests for fold-scoped fitted normalization statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from features.pipeline.normalization import rolling_zscore


def test_frozen_fold_does_not_drift_over_validation_rows() -> None:
    frame = pd.DataFrame({"feature": [10.0, 20.0, 30.0, 40.0, 90.0, 110.0]})
    train_mask = pd.Series([True, True, True, True, False, False])

    normalized = rolling_zscore(frame, window=2, train_mask=train_mask, mode="frozen_fold")

    # Terminal train window = (30, 40), mean=35, population std=5.
    assert normalized.loc[4, "feature"] == pytest.approx((90.0 - 35.0) / 5.0)
    assert normalized.loc[5, "feature"] == pytest.approx((110.0 - 35.0) / 5.0)


def test_frozen_fold_validation_poisoning_preserves_train_values() -> None:
    original = pd.DataFrame({"feature": [10.0, 20.0, 30.0, 40.0, 50.0]})
    altered = original.copy()
    altered.loc[4, "feature"] = 1_000_000.0
    mask = pd.Series([True, True, True, True, False])

    a = rolling_zscore(original, window=2, train_mask=mask, mode="frozen_fold")
    b = rolling_zscore(altered, window=2, train_mask=mask, mode="frozen_fold")

    pd.testing.assert_frame_equal(a.iloc[:4], b.iloc[:4])
    assert a.loc[4, "feature"] == pytest.approx(3.0)
    assert b.loc[4, "feature"] == pytest.approx((1_000_000.0 - 35.0) / 5.0)
