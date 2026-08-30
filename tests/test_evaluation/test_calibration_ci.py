"""Chain D tests: bootstrap CIs and probability-calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_rl.evaluation import (
    bootstrap_ci,
    calibration_report,
    plot_reliability_diagram,
    sharpe_stat,
)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0005, 0.001, size=500)
    ci = bootstrap_ci(rets, np.mean, n_boot=300, rng_seed=7)
    assert ci.lower <= ci.estimate <= ci.upper
    assert ci.estimate == pytest.approx(float(rets.mean()))
    # interval should be narrow but non-degenerate
    assert 0 < ci.upper - ci.lower < 0.01


def test_bootstrap_ci_sharpe_stat():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0002, 0.001, size=1000)
    ci = bootstrap_ci(rets, sharpe_stat, n_boot=200, rng_seed=3)
    est = sharpe_stat(rets)
    assert ci.estimate == pytest.approx(est)
    assert ci.lower < est < ci.upper


def test_bootstrap_ci_empty():
    ci = bootstrap_ci(np.array([]), np.mean)
    assert np.isnan(ci.estimate) and np.isnan(ci.lower) and np.isnan(ci.upper)


def test_calibration_perfectly_calibrated():
    rng = np.random.default_rng(11)
    # 2000 draws: p uniform, outcome ~ Bernoulli(p) → well calibrated
    p = rng.uniform(0.05, 0.95, size=2000)
    y = (rng.uniform(size=2000) < p).astype(float)
    rep = calibration_report(p, y, n_bins=10)
    assert rep.ece < 0.1, f"expected low ECE for calibrated data, got {rep.ece}"
    # Brier below the naive 0.25 baseline for informative predictions
    assert rep.brier < 0.25
    assert rep.bin_counts.sum() == 2000


def test_calibration_degenerate_extremes():
    # always predicts 1.0, always wins → ECE 0
    rep = calibration_report(np.ones(100), np.ones(100), n_bins=10)
    assert rep.ece == pytest.approx(0.0)
    assert rep.brier == pytest.approx(0.0)
    # predicts 1.0, never wins → maximal miscalibration in the top bin
    rep = calibration_report(np.ones(100), np.zeros(100), n_bins=10)
    assert rep.ece == pytest.approx(1.0)


def test_calibration_length_mismatch():
    with pytest.raises(ValueError):
        calibration_report(np.array([0.5, 0.7]), np.array([1.0]))


def test_reliability_diagram_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    rep = calibration_report(p, y, n_bins=8)
    ax = plot_reliability_diagram(rep)
    assert ax is not None
    assert ax.get_xlabel() == "Predicted probability of win"
