"""Trading performance metrics shared by baselines, supervised models, and RL agents.

All metrics are computed from an equity curve (one value per bar) and,
optionally, a list of per-trade PnL values. Functions are pure and
framework-free so they can be unit-tested against hand-computed fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

DEFAULT_PERIODS_PER_YEAR = 252 * 24  # hourly bars fallback


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable container for evaluation output.

    Attributes:
        sharpe: Annualised Sharpe ratio of bar returns.
        sortino: Annualised Sortino ratio (downside deviation).
        calmar: Annualised return divided by max drawdown.
        max_drawdown: Peak-to-trough equity decline as a positive fraction.
        total_return_pct: Total return over the period, in percent.
        total_pnl: Absolute PnL (final equity minus initial balance).
        profit_factor: Gross profit / abs(gross loss); inf if no losses.
        win_rate: Fraction of winning trades.
        expectancy: Mean PnL per trade.
        n_trades: Number of closed trades used for trade statistics.
        breach_count: Number of risk-limit breaches recorded by the caller.
    """

    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    n_trades: int = 0
    breach_count: int = 0
    extras: dict[str, float] = field(default_factory=dict)


def _bar_returns(equity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute simple bar-over-bar returns from an equity curve."""
    if equity.size < 2:
        return np.empty(0)
    return equity[1:] / equity[:-1] - 1.0


def max_drawdown(equity: Sequence[float] | NDArray[np.float64]) -> float:
    """Return peak-to-trough decline as a positive fraction.

    Args:
        equity: Equity values ordered in time.

    Returns:
        Largest relative decline from a running peak, in ``[0, 1]``.
    """
    curve = np.asarray(equity, dtype=float)
    if curve.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(curve)
    drawdowns = 1.0 - curve / running_peak
    return float(np.clip(drawdowns.max(), 0.0, 1.0))


def _sharpe(returns: NDArray[np.float64], periods_per_year: int) -> float:
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: NDArray[np.float64], periods_per_year: int) -> float:
    if returns.size < 2:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt((downside**2).mean()))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean() / downside_std * np.sqrt(periods_per_year))


def _trade_stats(trade_pnls: NDArray[np.float64]) -> dict[str, float]:
    """Profit factor, win rate and expectancy from per-trade PnLs."""
    n = int(trade_pnls.size)
    if n == 0:
        return {"profit_factor": 0.0, "win_rate": 0.0, "expectancy": 0.0}
    gains = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    gross_loss = float(-losses.sum())
    if gross_loss == 0.0:
        profit_factor = float("inf") if gains.size else 0.0
    else:
        profit_factor = float(gains.sum() / gross_loss)
    return {
        "profit_factor": profit_factor,
        "win_rate": float(gains.size / n),
        "expectancy": float(trade_pnls.mean()),
    }


def compute_metrics(
    equity_curve: Sequence[float],
    initial_balance: float,
    trade_pnls: Sequence[float] | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    breach_count: int = 0,
    extras: dict[str, float] | None = None,
) -> PerformanceMetrics:
    """Compute the standard metric set for one evaluation run.

    Args:
        equity_curve: Equity after each bar, starting at ``initial_balance``.
        initial_balance: Account balance before the first bar.
        trade_pnls: Per-trade PnL of closed trades, if available.
        periods_per_year: Bars per year used to annualise ratios.
        breach_count: Risk-limit breaches recorded during the run.
        extras: Caller-specific metrics (e.g. sweep delay) merged into output.

    Returns:
        A frozen :class:`PerformanceMetrics` instance.

    Raises:
        ValueError: If the equity curve is empty or the initial balance is
            not strictly positive.
    """
    curve = np.asarray(equity_curve, dtype=float)
    if curve.size == 0:
        raise ValueError("equity_curve must contain at least one value")
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")

    returns = _bar_returns(curve)
    mdd = max_drawdown(curve)
    total_pnl = float(curve[-1] - initial_balance)
    years = curve.size / periods_per_year
    annualised = ((curve[-1] / initial_balance) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    pnls = np.asarray(trade_pnls or [], dtype=float)
    stats = _trade_stats(pnls)

    calmar = float(annualised / mdd) if mdd > 0 else 0.0
    return PerformanceMetrics(
        sharpe=_sharpe(returns, periods_per_year),
        sortino=_sortino(returns, periods_per_year),
        calmar=calmar,
        max_drawdown=mdd,
        total_return_pct=float(total_pnl / initial_balance * 100.0),
        total_pnl=total_pnl,
        profit_factor=stats["profit_factor"],
        win_rate=stats["win_rate"],
        expectancy=stats["expectancy"],
        n_trades=int(pnls.size),
        breach_count=breach_count,
        extras=dict(extras or {}),
    )
