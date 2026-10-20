"""Backend-neutral account state and risk-policy decision contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RiskState:
    equity: float
    daily_pnl: float
    open_risk: float
    daily_loss_limit: float


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    risk_amount: float
    reason: str | None = None


class RiskPolicy(Protocol):
    """Evaluate requested risk without depending on a broker or policy model."""

    def decide(self, state: RiskState, requested_risk: float) -> RiskDecision: ...


__all__ = ["RiskDecision", "RiskPolicy", "RiskState"]
