"""Bootstrap confidence intervals for performance metrics (Chain D).

Resamples bar-level returns (stationary bootstrap preserves mild
autocorrelation) and trade PnLs to produce percentile confidence
intervals for Sharpe, Sortino, max drawdown and win rate. Multiple
seeds per variant are the primary uncertainty reduction; these CIs
quantify within-run sampling error on top of that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .metrics import DEFAULT_PERIODS_PER_YEAR


@dataclass(frozen=True)
class CI:
    """A percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.lower, self.estimate, self.upper)


def bootstrap_ci(
    values: NDArray[np.float64],
    stat_fn: Callable[[NDArray[np.float64]], float],
    n_boot: int = 2000,
    level: float = 0.95,
    rng_seed: int = 42,
    block_len: int = 8,
) -> CI:
    """Percentile bootstrap CI for ``stat_fn(values)``.

    Uses a moving-block bootstrap (blocks of ``block_len``) so mild
    autocorrelation in bar returns is preserved rather than destroyed
    by an i.i.d. resample.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), level)
    estimate = float(stat_fn(arr))
    if n == 1 or n_boot <= 0:
        return CI(estimate, estimate, estimate, level)

    # Moving-block starts require ``n - block_len + 1 > 0``; shrink the block
    # (down to i.i.d. resampling) when the sample is shorter than ``block_len``.
    block_len = max(1, min(int(block_len), n))
    rng = np.random.default_rng(rng_seed)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pieces = [arr[s : s + block_len] for s in starts[b]]
        sample = np.concatenate(pieces)[:n]
        stats[b] = stat_fn(sample)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return CI(estimate, float("nan"), float("nan"), level)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return CI(estimate, float(lo), float(hi), level)


def paired_metric_difference_ci(
    treatment_returns: NDArray[np.float64],
    control_returns: NDArray[np.float64],
    stat_fn: Callable[[NDArray[np.float64]], float],
    n_boot: int = 2000,
    level: float = 0.95,
    rng_seed: int = 42,
    block_len: int = 8,
) -> CI:
    """Moving-block CI for treatment minus control on aligned returns.

    Each bootstrap draw selects the *same* contiguous time blocks for both
    series.  The caller must supply matching periods from matched systems;
    this is within-path uncertainty, separate from paired-seed uncertainty.
    """
    treatment = np.asarray(treatment_returns, dtype=np.float64)
    control = np.asarray(control_returns, dtype=np.float64)
    if treatment.ndim != 1 or control.ndim != 1 or treatment.size != control.size:
        raise ValueError("Paired return series must be one-dimensional and have the same length")
    if not np.isfinite(treatment).all() or not np.isfinite(control).all():
        raise ValueError("Paired return series must contain only finite returns")
    n = treatment.size
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), level)
    estimate = float(stat_fn(treatment) - stat_fn(control))
    if n == 1 or n_boot <= 0:
        return CI(estimate, estimate, estimate, level)

    block_len = max(1, min(int(block_len), n))
    rng = np.random.default_rng(rng_seed)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    stats = np.empty(n_boot, dtype=np.float64)
    for i, row in enumerate(starts):
        indices = np.concatenate([np.arange(s, s + block_len) for s in row])[:n]
        stats[i] = stat_fn(treatment[indices]) - stat_fn(control[indices])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return CI(estimate, float("nan"), float("nan"), level)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(finite, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return CI(estimate, float(lo), float(hi), level)


def sharpe_stat(
    returns: NDArray[np.float64], periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    """Sharpe of bar returns (annualisation matching evaluation/metrics)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(r.mean() / std * np.sqrt(periods_per_year))


def sortino_stat(
    returns: NDArray[np.float64], periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> float:
    """Sortino of bar returns (downside deviation vs. zero)."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r, 0.0)
    dd = np.sqrt((downside**2).mean())
    if dd == 0.0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown_stat(equity: NDArray[np.float64]) -> float:
    """Peak-to-trough equity decline as a positive fraction."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.where(peaks == 0, np.nan, peaks)
    return float(np.nanmax(dd))


def win_rate_stat(trade_pnls: NDArray[np.float64]) -> float:
    """Fraction of winning trades."""
    p = np.asarray(trade_pnls, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float((p > 0).sum() / p.size)


def metrics_with_ci(
    equity_curve: NDArray[np.float64],
    trade_pnls: NDArray[np.float64],
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    n_boot: int = 2000,
    level: float = 0.95,
    rng_seed: int = 42,
    initial_balance: float | None = None,
) -> dict[str, CI]:
    """Bootstrap CIs for Sharpe/Sortino (bar returns) and MDD/win-rate.

    Max drawdown resamples contiguous blocks of returns, then reconstructs
    each equity path from the original initial equity. Resampling equity
    *levels* would create impossible crashes in monotonic runs.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if initial_balance is not None:
        if not np.isfinite(initial_balance) or initial_balance <= 0:
            raise ValueError("initial_balance must be finite and positive")
        if eq.size and eq[0] != initial_balance:
            eq = np.concatenate(([float(initial_balance)], eq))
    rets = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([])

    def reconstructed_drawdown(sample: NDArray[np.float64]) -> float:
        if eq.size == 0:
            return float("nan")
        path = np.empty(sample.size + 1, dtype=np.float64)
        path[0] = eq[0]
        path[1:] = eq[0] * np.cumprod(1.0 + sample)
        return max_drawdown_stat(path)

    mdd_ci = bootstrap_ci(rets, reconstructed_drawdown, n_boot, level, rng_seed)
    # Bootstrap estimate should describe the observed path, including the
    # initial-capital observation and its first loss.
    mdd_ci = CI(max_drawdown_stat(eq), mdd_ci.lower, mdd_ci.upper, mdd_ci.level)
    return {
        "sharpe": bootstrap_ci(
            rets, lambda r: sharpe_stat(r, periods_per_year), n_boot, level, rng_seed
        ),
        "sortino": bootstrap_ci(
            rets, lambda r: sortino_stat(r, periods_per_year), n_boot, level, rng_seed
        ),
        "max_drawdown": mdd_ci,
        "win_rate": bootstrap_ci(
            np.asarray(trade_pnls, dtype=np.float64), win_rate_stat, n_boot, level, rng_seed
        ),
    }
