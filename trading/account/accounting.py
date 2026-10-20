"""Reusable account accounting primitives."""

from __future__ import annotations


def payout_due(balance: float, target_balance: float) -> bool:
    return float(balance) >= float(target_balance)


def payout_amount(balance: float, operating_balance: float) -> float:
    return max(0.0, float(balance) - float(operating_balance))
