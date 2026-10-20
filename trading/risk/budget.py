"""Risk-request approval against policy and account capacity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskBudgetDecision:
    requested_usd: float
    approved_usd: float
    policy_cap_usd: float
    execution_reserve_usd: float


def approve_dynamic_risk(
    *,
    equity: float,
    risk_frac: float,
    max_agent_risk_frac: float,
    daily_capacity: float,
    account_capacity: float,
    execution_reserve_usd: float,
) -> RiskBudgetDecision:
    requested = max(0.0, float(equity) * float(risk_frac))
    policy_cap = min(
        max(0.0, float(max_agent_risk_frac) * float(equity)),
        max(0.0, float(daily_capacity)),
        max(0.0, float(account_capacity)),
    )
    reserve = max(0.0, float(execution_reserve_usd))
    approved = max(0.0, min(requested, max(0.0, policy_cap - reserve)))
    return RiskBudgetDecision(requested, approved, policy_cap, reserve)
