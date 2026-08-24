"""Distributional performance metrics over per-observation data.

The distribution layer answers questions that point estimates (Sharpe,
mean PnL) cannot: how dispersed are outcomes, how heavy are the tails,
and how stable is performance. It is strictly additive with respect to
the existing metrics in :mod:`quant_rl.evaluation.metrics`.

Conventions
-----------
- One observation = one completed trade (or one qualifying sweep event).
- ``VaR5`` is the empirical 5th percentile, using the sign convention of
  the input (negative for PnL). ``CVaR5`` is the mean of values at or
  below ``VaR5``.
- Skewness/kurtosis use the sample (g1/g2) formulas implemented with
  NumPy - no scipy dependency.
- Empty input returns ``count=0`` with NaN fields; non-finite values are
  removed from the cleaned vector (their count is reported) - they are
  never silently mixed into a statistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DistributionMetrics:
    """One distribution summary of a sequence of observations."""

    count: int
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    p01: float
    p05: float
    p25: float
    p75: float
    p95: float
    p99: float
    skewness: float
    kurtosis: float
    var_05: float
    cvar_05: float
    dropped: int


_QUANTILES = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)


def compute_distribution_metrics(values: Sequence[float]) -> DistributionMetrics:
    """Summarise a sequence of observations with distribution statistics.

    Args:
        values: Observations (per-trade PnL, holding time, sweep delay).

    Returns:
        A frozen :class:`DistributionMetrics`; empty or all-non-finite input
        yields ``count=0`` and NaN numeric fields.
    """
    raw = np.asarray(values, dtype=float)
    dropped = int(np.isnan(raw).sum() + np.isinf(raw).sum())
    clean = raw[np.isfinite(raw)]

    if clean.size == 0:
        nan = float("nan")
        return DistributionMetrics(
            count=0,
            mean=nan,
            median=nan,
            std=nan,
            min_value=nan,
            max_value=nan,
            p01=nan,
            p05=nan,
            p25=nan,
            p75=nan,
            p95=nan,
            p99=nan,
            skewness=nan,
            kurtosis=nan,
            var_05=nan,
            cvar_05=nan,
            dropped=dropped,
        )

    mean = float(clean.mean())
    std = float(clean.std(ddof=1) if clean.size > 1 else 0.0)
    p01, p05, p25, p75, p95, p99 = (float(q) for q in np.quantile(clean, _QUANTILES))
    median = float(np.median(clean))
    skewness = _sample_skewness(clean, mean, std)
    kurtosis = _sample_kurtosis(clean, mean, std)

    var_05 = float(np.quantile(clean, 0.05))
    tail = clean[clean <= var_05]
    cvar_05 = float(tail.mean()) if tail.size else var_05

    return DistributionMetrics(
        count=int(clean.size),
        mean=mean,
        median=median,
        std=std,
        min_value=float(clean.min()),
        max_value=float(clean.max()),
        p01=p01,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        p99=p99,
        skewness=skewness,
        kurtosis=kurtosis,
        var_05=var_05,
        cvar_05=cvar_05,
        dropped=dropped,
    )


def _sample_skewness(x: NDArray[np.float64], mean: float, std: float) -> float:
    """Return the sample skewness (g1); 0 when degenerate or too few values."""
    if x.size < 3 or std == 0.0:
        return 0.0
    m3 = float(((x - mean) ** 3).mean())
    return float(m3 / (std**3))


def _sample_kurtosis(x: NDArray[np.float64], mean: float, std: float) -> float:
    """Return the sample excess kurtosis (g2); 0 when degenerate."""
    if x.size < 4 or std == 0.0:
        return 0.0
    m4 = float(((x - mean) ** 4).mean())
    var = max(std * std, 1e-24)
    return float(m4 / (var * var) - 3.0)
