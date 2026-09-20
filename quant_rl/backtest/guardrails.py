"""FTMO guardrail checks.

Hard limits (``daily_loss_limit``, ``max_loss_limit``) are kill-switches.
``soft_daily_loss_limit`` is a soft brick: block new entries / reward pressure
only — it does **not** force-close or end the day.
"""

from __future__ import annotations

from dataclasses import dataclass

from .account import AccountState


@dataclass(frozen=True)
class FTMOGuardrails:
    daily_loss_limit: float = 5_000.0
    max_loss_limit: float = 10_000.0
    risk_per_trade_limit: float = 1_000.0
    soft_daily_loss_limit: float = 2_000.0
    # Reward-only year-loss shaping floor (not a hard kill-switch).
    soft_max_loss_limit: float = 5_000.0

    # ------------------------------------------------------------------

    def check_daily(self, acc: AccountState) -> bool:
        """Return True (breached) if daily loss exceeds the hard limit."""
        return acc.daily_loss >= self.daily_loss_limit

    def check_soft_daily(self, acc: AccountState) -> bool:
        """Return True when the soft brick is active (no new entries)."""
        soft = float(self.soft_daily_loss_limit)
        if soft <= 0.0:
            return False
        return acc.daily_loss >= soft

    def check_max_drawdown(self, acc: AccountState) -> bool:
        """Return True if absolute loss from *initial* balance ≥ ``max_loss_limit``.

        This is FTMO-style max loss (fixed from deposit), **not** trailing
        high-water-mark drawdown from peak equity.
        """
        return acc.loss_from_initial() >= self.max_loss_limit

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
