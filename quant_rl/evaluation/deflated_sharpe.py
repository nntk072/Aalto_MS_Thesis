"""Deflated Sharpe ratio (Bailey and López de Prado, 2014).

The Sharpe ratio that survives after accounting for the number of trials
and for skew and kurtosis. ``sr`` is the non-annualised per-observation
Sharpe (mean / sample std), not the annualised number stored on run summaries.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EULER = 0.5772156649015329


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def expected_maximum_sharpe(sharpe_std: float, n_trials: int) -> float:
    """Sharpe expected from the best of ``n_trials`` under a zero-mean null.

    Args:
        sharpe_std: Cross-trial standard deviation of the estimated Sharpes.
        n_trials: Number of configurations that were compared.

    Returns:
        The benchmark Sharpe ``SR0``. Zero when there is a single trial.
    """
    if n_trials <= 1 or sharpe_std <= 0.0:
        return 0.0
    z_one = _norm_ppf(1.0 - 1.0 / n_trials)
    z_exp = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sharpe_std * ((1.0 - _EULER) * z_one + _EULER * z_exp))


def deflated_sharpe_probability(
    sr: float,
    sr0: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability that the true Sharpe exceeds ``sr0``.

    Args:
        sr: Non-annualised Sharpe of the selected track record.
        sr0: Benchmark from :func:`expected_maximum_sharpe`.
        n_obs: Number of returns used to estimate ``sr``.
        skew: Standardized skewness of those returns.
        kurtosis: Non-excess kurtosis (Gaussian is 3).

    Returns:
        A probability in ``[0, 1]``. ``nan`` when the variance term is not positive.
    """
    if n_obs <= 1:
        return float("nan")
    variance = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if variance <= 0.0:
        return float("nan")
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return _norm_cdf(z)


def return_moments(returns: np.ndarray[Any, Any]) -> tuple[float, float, float, int]:
    """Non-annualised Sharpe, skew, non-excess kurtosis, and length.

    Args:
        returns: Per-observation P&L or simple returns.

    Returns:
        ``(sharpe, skew, kurtosis, n)``. Sharpe is 0 when the sample is too short
        or has no variance.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return 0.0, 0.0, 3.0, n
    std = float(x.std(ddof=1))
    sharpe = 0.0 if std == 0.0 else float(x.mean() / std)
    centered = x - x.mean()
    m2 = float(np.mean(centered**2))
    if m2 == 0.0:
        return sharpe, 0.0, 3.0, n
    skew = float(np.mean(centered**3) / m2**1.5)
    kurtosis = float(np.mean(centered**4) / m2**2)
    return sharpe, skew, kurtosis, n
