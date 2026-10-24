"""FTMO guardrail checks.

Hard limits (``daily_loss_limit``, ``max_loss_limit``) are FTMO kill-switches.
``trailing_dd_limit`` is an optional PPO-only dollar cap from peak (0 = off).
The selected ``limit_value_mode`` can use fixed dollars or percentages. In
percentage mode, per-trade and daily values use current equity, while max loss
uses initial equity. Soft bricks
(``soft_daily_loss_limit``, ``soft_trailing_dd_limit``) block new entries /
add reward pressure only — they do **not** force-close or end the day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading.account.state import AccountState


@dataclass(frozen=True)
class FTMOGuardrails:
    daily_loss_limit: float = 5_000.0
    max_loss_limit: float = 10_000.0
    risk_per_trade_limit: float = 1_000.0
    soft_daily_loss_limit: float = 5_000.0
    # Matches the hard limit, so there is no earlier soft shaping band.
    soft_max_loss_limit: float = 10_000.0
    # Optional PPO-only fraction of *initial* balance as a dollar cap from peak.
    trailing_dd_limit: float = 0.0
    soft_trailing_dd_limit: float = 0.0
    limit_value_mode: str = "fixed_usd"
    risk_per_trade_limit_pct: float = 0.01
    daily_loss_limit_pct: float = 0.05
    soft_daily_loss_limit_pct: float = 0.05
    max_loss_limit_initial_pct: float = 0.10
    soft_max_loss_limit_initial_pct: float = 0.10

    def __post_init__(self) -> None:
        if self.limit_value_mode not in {"fixed_usd", "percent_equity"}:
            raise ValueError("ftmo.limit_value_mode must be 'fixed_usd' or 'percent_equity'")
        percentages = (
            self.risk_per_trade_limit_pct,
            self.daily_loss_limit_pct,
            self.soft_daily_loss_limit_pct,
            self.max_loss_limit_initial_pct,
            self.soft_max_loss_limit_initial_pct,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in percentages):
            raise ValueError("FTMO percentage limits must be finite and non-negative")
        if any(float(value) > 1.0 for value in percentages):
            raise ValueError("FTMO percentage limits must be fractions in [0, 1]")

    def daily_limit_for(self, acc: AccountState) -> float:
        """Resolve the hard daily-loss threshold from the selected value mode."""
        if self.limit_value_mode == "percent_equity":
            return max(0.0, float(acc.equity)) * float(self.daily_loss_limit_pct)
        return float(self.daily_loss_limit)

    def soft_daily_limit_for(self, acc: AccountState) -> float:
        """Resolve the soft daily threshold from current equity when requested."""
        if self.limit_value_mode == "percent_equity":
            return max(0.0, float(acc.equity)) * float(self.soft_daily_loss_limit_pct)
        return float(self.soft_daily_loss_limit)

    def max_loss_limit_for(self, acc: AccountState) -> float:
        """Resolve the maximum-loss threshold; percentage mode uses initial equity."""
        if self.limit_value_mode == "percent_equity":
            return max(0.0, float(acc.initial_balance)) * float(self.max_loss_limit_initial_pct)
        return float(self.max_loss_limit)

    def soft_max_loss_limit_for(self, acc: AccountState) -> float:
        """Resolve the soft maximum-loss threshold from initial equity."""
        if self.limit_value_mode == "percent_equity":
            return max(0.0, float(acc.initial_balance)) * float(
                self.soft_max_loss_limit_initial_pct
            )
        return float(self.soft_max_loss_limit)

    def risk_per_trade_limit_for(self, acc: AccountState) -> float:
        """Resolve the hard per-trade threshold from current equity if selected."""
        if self.limit_value_mode == "percent_equity":
            return max(0.0, float(acc.equity)) * float(self.risk_per_trade_limit_pct)
        return float(self.risk_per_trade_limit)

    # ------------------------------------------------------------------

    def check_daily(self, acc: AccountState) -> bool:
        """Return True (breached) if daily loss exceeds the hard limit."""
        return acc.canonical_daily_loss() >= self.daily_limit_for(acc)

    def check_soft_daily(self, acc: AccountState) -> bool:
        """Return True when the soft brick is active (no new entries)."""
        soft = self.soft_daily_limit_for(acc)
        if soft <= 0.0:
            return False
        return acc.canonical_daily_loss() >= soft

    def check_max_drawdown(self, acc: AccountState) -> bool:
        """Return True if absolute loss from *initial* balance ≥ ``max_loss_limit``.

        This is FTMO-style max loss (fixed from deposit), **not** trailing
        high-water-mark drawdown from peak equity.
        """
        return acc.loss_from_initial() >= self.max_loss_limit_for(acc)

    def check_trailing_dd(self, acc: AccountState) -> bool:
        """Return True if enabled peak-to-trough DD reaches its initial-balance cap."""
        hard = float(self.trailing_dd_limit)
        if hard <= 0.0:
            return False
        cap = hard * float(acc.initial_balance)
        return (acc.peak_equity - acc.equity) >= cap

    def check_soft_trailing_dd(self, acc: AccountState) -> bool:
        """Return True when the trailing-DD soft brick is active."""
        soft = float(self.soft_trailing_dd_limit)
        if soft <= 0.0:
            return False
        cap = soft * float(acc.initial_balance)
        return (acc.peak_equity - acc.equity) >= cap

    def check_trade_risk(self, risk: float, acc: AccountState | None = None) -> bool:
        """Return True (breached) if *risk* $ per trade exceeds the limit."""
        limit = self.risk_per_trade_limit if acc is None else self.risk_per_trade_limit_for(acc)
        return risk > limit

    def any_breach(self, acc: AccountState) -> bool:
        return self.check_trailing_dd(acc) or self.check_daily(acc) or self.check_max_drawdown(acc)

    def breach_reason(self, acc: AccountState) -> str | None:
        if self.check_trailing_dd(acc):
            return "trailing_dd"
        if self.check_max_drawdown(acc):
            return "max_drawdown"
        if self.check_daily(acc):
            return "daily_loss"
        return None
