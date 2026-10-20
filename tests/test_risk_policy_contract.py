from __future__ import annotations

from dataclasses import dataclass

from trading.risk.contracts import RiskDecision, RiskPolicy, RiskState


@dataclass(frozen=True)
class _FixedCapPolicy:
    cap: float

    def decide(self, state: RiskState, requested_risk: float) -> RiskDecision:
        del state
        amount = min(requested_risk, self.cap)
        return RiskDecision(allowed=amount > 0.0, risk_amount=amount)


class _DailyLossPolicy:
    def decide(self, state: RiskState, requested_risk: float) -> RiskDecision:
        remaining = state.daily_loss_limit + state.daily_pnl - state.open_risk
        if remaining <= 0.0:
            return RiskDecision(False, 0.0, "daily_loss_limit")
        amount = min(requested_risk, remaining)
        return RiskDecision(allowed=amount > 0.0, risk_amount=amount)


def test_two_risk_policies_consume_the_same_semantic_state() -> None:
    state = RiskState(equity=100_000.0, daily_pnl=-400.0, open_risk=100.0, daily_loss_limit=500.0)
    policies: tuple[RiskPolicy, ...] = (_FixedCapPolicy(50.0), _DailyLossPolicy())

    decisions = tuple(policy.decide(state, requested_risk=200.0) for policy in policies)

    assert decisions == (
        RiskDecision(True, 50.0),
        RiskDecision(False, 0.0, "daily_loss_limit"),
    )
