"""Repeated-payout policy state transitions shared by simulation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from trading.account.accounting import payout_amount, payout_due


@dataclass(frozen=True, slots=True)
class PayoutDecision:
    due: bool
    amount: float


def evaluate_payout(
    balance: float, operating_balance: float, target_balance: float
) -> PayoutDecision:
    due = payout_due(balance, target_balance)
    return PayoutDecision(due=due, amount=payout_amount(balance, operating_balance) if due else 0.0)
