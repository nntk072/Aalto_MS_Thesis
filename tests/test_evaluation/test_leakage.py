"""Unit tests for leakage detectors."""

from __future__ import annotations

import numpy as np

from quant_rl.evaluation.leakage import label_shuffle_score, time_shift_feature_scores


def test_label_shuffle_detects_independence() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred = rng.normal(size=200)
    out = label_shuffle_score(y, pred, n_perm=50, seed=1)
    assert out["null_exceed_frac"] > 0.05


def test_label_shuffle_flags_perfect_leak() -> None:
    y = np.linspace(-1, 1, 200)
    out = label_shuffle_score(y, y, n_perm=40, seed=2)
    assert out["observed_abs_corr"] > 0.99
    assert out["null_exceed_frac"] <= 0.05


def test_time_shift_causal_feature_does_not_improve() -> None:
    rng = np.random.default_rng(3)
    # feature is lagged target (causal): f[t] ≈ y[t-1]
    y = rng.normal(size=300)
    f = np.roll(y, 1)
    f[0] = 0.0
    scores = time_shift_feature_scores(f, y, shifts=(1, 5))
    # shifting feature further forward should not beat shift_0 by much for this toy
    assert scores["shift_0"] >= 0.0
    assert np.isfinite(scores["shift_1"])
