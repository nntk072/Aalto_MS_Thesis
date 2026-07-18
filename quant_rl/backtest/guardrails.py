"""FTMO hard guardrail checks.

All limits are *hard kill-switches*: once breached the episode/day is over.
"""
from __future__ import annotations

from dataclasses import dataclass

from .account import AccountState


@dataclass(frozen=True)
class FTMOGuardrails:
    daily_loss_limit: float = 5_000.0
    max_loss_limit: float = 10_000.0
    risk_per_trade_limit: float = 1_000.0

    # ------------------------------------------------------------------

    def check_daily(self, acc: AccountState) -> bool:
        """Return True (breached) if daily loss exceeds the limit."""
        return acc.daily_loss >= self.daily_loss_limit

    def check_max_drawdown(self, acc: AccountState) -> bool:
        """Return True (breached) if max drawdown exceeds the limit."""
        return acc.max_drawdown >= self.max_loss_limit

    def check_trade_risk(self, risk: float) -> bool:
        """Return True (breached) if *risk* $ per trade exceeds the limit."""
        return risk > self.risk_per_trade_limit

    def any_breach(self, acc: AccountState) -> bool:
        return self.check_daily(acc) or self.check_max_drawdown(acc)

    def breach_reason(self, acc: AccountState) -> str | None:
        if self.check_max_drawdown(acc):
            return "max_drawdown"
        if self.check_daily(acc):
            return "daily_loss"
        return None
