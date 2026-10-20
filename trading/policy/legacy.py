"""Legacy policy marker and compatibility semantics."""

from __future__ import annotations

LEGACY_POLICY_MODE = "legacy"


def uses_fixed_risk_budget(risk_mode: str, repeated_payout: bool) -> bool:
    return str(risk_mode) == "fixed" and not bool(repeated_payout)
