"""Trading performance metrics.

Includes all standard metrics plus FTMO-specific ``breach_rate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    total_return: float
    profit_factor: float
    expectancy: float
    win_rate: float
    total_trades: int
    turnover: float
    breach_rate: float  # fraction of sessions that hit an FTMO guardrail
    total_pnl: float = 0.0
    avg_trade: float = 0.0
    max_consec_loss: int = 0


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    mu = returns.mean()
    downside = returns[returns < 0]
    ds_std = downside.std(ddof=1) if len(downside) > 1 else 1e-9
    return float(mu / ds_std * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (roll_max - equity) / roll_max.replace(0, np.nan)
    return float(drawdown.max())


def _calmar(equity: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> float:
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / len(equity)) - 1
    mdd = _max_drawdown(equity)
    return float(ann_ret / mdd) if mdd > 0 else 0.0


def calculate_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    n_sessions: int = 1,
    n_breach_sessions: int = 0,
    periods_per_year: int = 252 * 390,  # M1 bars per year (approx)
) -> Metrics:
    """Compute all metrics.

    Parameters
    ----------
    equity:
        Bar-level equity curve.
    trades:
        DataFrame with a ``pnl`` column (can be None).
    n_sessions, n_breach_sessions:
        Used for ``breach_rate`` calculation.
    """
    bar_returns = equity.pct_change().dropna()

    sharpe = _sharpe(bar_returns, periods_per_year)
    sortino = _sortino(bar_returns, periods_per_year)
    max_dd = _max_drawdown(equity)
    calmar = _calmar(equity, bar_returns, periods_per_year)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0.0

    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        pnl = trades["pnl"].dropna()
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float(len(wins) / len(pnl)) if len(pnl) > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(losses.abs().sum()) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss
        expectancy = float(pnl.mean())
        total_trades = len(pnl)
        turnover = total_trades / max(len(equity), 1)
        total_pnl = float(pnl.sum())
        avg_trade = float(pnl.mean())
        # max consecutive losses
        max_c = cur = 0
        for v in pnl:
            if v < 0:
                cur += 1
                max_c = max(max_c, cur)
            else:
                cur = 0
        max_consec_loss = max_c
    else:
        win_rate = profit_factor = expectancy = 0.0
        total_pnl = avg_trade = 0.0
        total_trades = max_consec_loss = 0
        turnover = 0.0

    breach_rate = float(n_breach_sessions / n_sessions) if n_sessions > 0 else 0.0

    return Metrics(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        total_return=total_return,
        profit_factor=profit_factor,
        expectancy=expectancy,
        win_rate=win_rate,
        total_trades=total_trades,
        turnover=turnover,
        breach_rate=breach_rate,
        total_pnl=total_pnl,
        avg_trade=avg_trade,
        max_consec_loss=max_consec_loss,
    )
