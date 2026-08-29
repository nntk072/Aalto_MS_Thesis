"""Trading performance metrics shared by baselines, supervised models, and RL agents.

All metrics are computed from an equity curve (one value per bar) and,
optionally, a list of per-trade PnL values. Functions are pure and
framework-free so they can be unit-tested against hand-computed fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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
        breach_count: Number of risk-limit breach events recorded by the caller.
        breach_rate: Fraction of sessions with at least one breach event
            (``breach_count / n_sessions``).  Session-level semantics from the
            legacy ``quant_rl.eval.metrics.Metrics`` dataclass, kept alongside
            the raw event count because they answer different questions.
        turnover: Number of trades per evaluation bar.
        max_consec_loss: Longest streak of consecutive losing trades.
        avg_trade: Alias of ``expectancy`` (mean PnL per trade).
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
    breach_rate: float = 0.0
    turnover: float = 0.0
    max_consec_loss: int = 0
    avg_trade: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def sweep_delay_breakdown(
    trade_log: Sequence[dict[str, Any]],
    delay_key: str = "sweep_delay_s",
    level_key: str = "level_type",
) -> dict[str, float]:
    """Summarise Sweep Delay and Asian-vs-London targeting from open trades.

    Args:
        trade_log: The environment's ``trade_log`` list.
        delay_key: Trade-log key holding the delay in seconds.
        level_key: Trade-log key holding the swept level name.

    Returns:
        Dict with ``sweep_delay_mean_s``, ``sweep_delay_median_s``,
        ``n_entries``, ``london_pct`` and ``asian_pct`` (percent of
        entries attributed to London/Asian levels).
    """
    delays: list[float] = []
    london = 0
    asian = 0
    n_entries = 0
    for trade in trade_log:
        if trade.get("type") != "open":
            continue
        n_entries += 1
        delay = trade.get(delay_key)
        if delay is not None and np.isfinite(delay):
            delays.append(float(delay))
        level = str(trade.get(level_key, ""))
        if "london" in level:
            london += 1
        elif "asian" in level:
            asian += 1

    total = float(london + asian)
    return {
        "sweep_delay_mean_s": float(np.mean(delays)) if delays else 0.0,
        "sweep_delay_median_s": float(np.median(delays)) if delays else 0.0,
        "n_entries": float(n_entries),
        "london_pct": london / total * 100.0 if total else 0.0,
        "asian_pct": asian / total * 100.0 if total else 0.0,
    }


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


def _max_consec_loss(pnls: NDArray[np.float64]) -> int:
    """Longest streak of consecutive negative trade PnLs."""
    max_streak = current = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_metrics(
    equity_curve: Sequence[float] | NDArray[np.floating[Any]],
    initial_balance: float,
    trade_pnls: Sequence[float] | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    breach_count: int = 0,
    n_sessions: int = 1,
    extras: dict[str, float] | None = None,
) -> PerformanceMetrics:
    """Compute the standard metric set for one evaluation run.

    Args:
        equity_curve: Equity after each bar, starting at ``initial_balance``.
        initial_balance: Account balance before the first bar.
        trade_pnls: Per-trade PnL of closed trades, if available.
        periods_per_year: Bars per year used to annualise ratios.
        breach_count: Risk-limit breach events recorded during the run.
        n_sessions: Number of trading sessions in the run; used for
            ``breach_rate`` (fraction of sessions with a breach).
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
        breach_rate=(breach_count / n_sessions) if n_sessions > 0 else 0.0,
        turnover=float(pnls.size / curve.size) if curve.size else 0.0,
        max_consec_loss=_max_consec_loss(pnls),
        avg_trade=stats["expectancy"],
        extras=dict(extras or {}),
    )


# M1 bars per year: kept as the default for the pandas adapter below so the
# migrated ``quant_rl.train`` runners preserve their historical annualisation.
LEGACY_M1_PERIODS_PER_YEAR = 252 * 390


def calculate_metrics(
    equity: Any,
    trades: Any = None,
    n_sessions: int = 1,
    n_breach_sessions: int = 0,
    periods_per_year: int = LEGACY_M1_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    """Pandas-friendly adapter matching the legacy ``eval.metrics`` signature.

    Accepts a ``pd.Series`` (or any sequence) equity curve and an optional
    ``pd.DataFrame`` of trades with a ``pnl`` column, and delegates to
    :func:`compute_metrics`.  The initial balance is taken from the first
    equity value, matching the legacy behaviour of the migrated
    ``quant_rl.train`` runners.

    Args:
        equity: Bar-level equity curve (``pd.Series`` or sequence).
        trades: Trade records (``pd.DataFrame`` with a ``pnl`` column) or None.
        n_sessions: Number of trading sessions, for ``breach_rate``.
        n_breach_sessions: Sessions in which a guardrail breach occurred.
        periods_per_year: Bars per year for annualisation.  Defaults to M1
            bars (``252 * 390``) to match the legacy ``quant_rl.eval`` runs.

    Returns:
        A frozen :class:`PerformanceMetrics` instance.
    """
    curve = (
        equity.to_numpy(dtype=float)
        if hasattr(equity, "to_numpy")
        else np.asarray(equity, dtype=float)
    )  # noqa: E501
    if curve.size == 0:
        raise ValueError("equity must contain at least one value")

    trade_pnls: list[float] | None = None
    if trades is not None and len(trades) > 0 and "pnl" in trades:
        pnl = trades["pnl"]
        pnl = pnl.dropna() if hasattr(pnl, "dropna") else pnl
        trade_pnls = [float(v) for v in pnl]

    return compute_metrics(
        curve,
        initial_balance=float(curve[0]),
        trade_pnls=trade_pnls,
        periods_per_year=periods_per_year,
        breach_count=int(n_breach_sessions),
        n_sessions=n_sessions,
    )
