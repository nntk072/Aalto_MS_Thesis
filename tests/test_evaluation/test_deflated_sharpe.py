"""Deflated Sharpe moves the right way as trials and track-record length change."""

from __future__ import annotations

import numpy as np
import pytest

from quant_rl.evaluation.deflated_sharpe import (
    deflated_sharpe_probability,
    expected_maximum_sharpe,
    return_moments,
)


@pytest.mark.unit
def test_more_trials_raise_the_benchmark() -> None:
    few = expected_maximum_sharpe(sharpe_std=0.2, n_trials=2)
    many = expected_maximum_sharpe(sharpe_std=0.2, n_trials=6)
    assert many > few
    assert expected_maximum_sharpe(sharpe_std=0.2, n_trials=1) == 0.0


@pytest.mark.unit
def test_a_strong_short_record_is_not_deflated_away() -> None:
    # One trial, Gaussian returns, clearly positive mean.
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.01, size=500)
    sr, skew, kurt, n = return_moments(returns)
    prob = deflated_sharpe_probability(sr, sr0=0.0, n_obs=n, skew=skew, kurtosis=kurt)
    assert prob > 0.95


@pytest.mark.unit
def test_selection_among_noise_is_not_a_discovery() -> None:
    # The best of six zero-mean Sharpes should not clear a high deflated probability
    # once the benchmark uses that cross-trial dispersion.
    rng = np.random.default_rng(1)
    sharpes = []
    moments = []
    for _ in range(6):
        sample = rng.normal(0.0, 1.0, size=80)
        sr, skew, kurt, n = return_moments(sample)
        sharpes.append(sr)
        moments.append((sr, skew, kurt, n))
    best = max(range(6), key=lambda i: moments[i][0])
    sr, skew, kurt, n = moments[best]
    sr0 = expected_maximum_sharpe(float(np.std(sharpes, ddof=1)), n_trials=6)
    prob = deflated_sharpe_probability(sr, sr0=sr0, n_obs=n, skew=skew, kurtosis=kurt)
    assert prob < 0.95
