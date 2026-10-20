"""Account loss floors and remaining risk capacity."""

from __future__ import annotations


def canonical_daily_loss(session_start_equity: float, equity: float) -> float:
    return max(0.0, float(session_start_equity) - float(equity))


def daily_floor(session_start_equity: float, daily_loss_limit: float) -> float:
    return float(session_start_equity) - float(daily_loss_limit)


def remaining_daily_capacity(
    session_start_equity: float, equity: float, daily_loss_limit: float
) -> float:
    return max(0.0, float(equity) - daily_floor(session_start_equity, daily_loss_limit))


def remaining_account_capacity(equity: float, account_floor: float) -> float:
    return max(0.0, float(equity) - float(account_floor))
