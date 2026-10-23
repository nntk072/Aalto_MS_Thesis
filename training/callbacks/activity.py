"""In-sample activity admissibility for completed training runs."""

from __future__ import annotations

from typing import Any


def assess_train_activity(
    n_trades: int,
    *,
    minimum_trades: int = 1,
    sparse_warning_below: int = 5,
) -> dict[str, object]:
    """Classify completed train-replay trades independently of survival."""
    if isinstance(n_trades, bool) or not isinstance(n_trades, int) or n_trades < 0:
        raise ValueError("n_trades must be a non-negative integer")
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be at least one")
    if sparse_warning_below < minimum_trades:
        raise ValueError("sparse_warning_below must be at least minimum_trades")

    active = n_trades >= minimum_trades
    sparse = active and n_trades < sparse_warning_below
    status = "INACTIVE" if not active else "ACTIVE_SPARSE" if sparse else "ACTIVE"
    return {
        "ok": active,
        "n_trades": n_trades,
        "minimum_trades": minimum_trades,
        "status": status,
        "warning": sparse,
        "reason": "zero_train_replay_trades" if not active else "",
    }


def classify_zero_trade_cause(
    n_trades: int,
    entry_state: dict[str, Any] | None = None,
    entry_diag: dict[str, Any] | None = None,
) -> str | None:
    """Use existing funnel counters to label a zero-trade replay."""
    if n_trades > 0:
        return None
    state = entry_state or {}
    diag = entry_diag or {}
    if bool(state.get("entry_state_machine", False)):
        candidates = int(state.get("candidate_created", 0))
        armed = int(state.get("armed_bars", 0))
        requested = int(state.get("trigger_requested", 0))
        refused = int(state.get("trigger_refused", 0))
        entered = int(state.get("entered", 0))
        closed = int(state.get("closed", 0))
        if candidates == 0:
            return "FAIL_ENTRY_PATH"
        if armed == 0:
            return "FSM_ARMING_FAILURE"
        if requested == 0:
            return "FAIL_POLICY_COLLAPSE"
        if refused > 0:
            return "FAIL_EXECUTION_GATE"
        if entered > 0 and closed == 0:
            return "EXECUTION_FAILURE"
    hold_risk_floor = int(diag.get("hold_risk_floor", 0))
    other_holds = max(
        (
            int(value)
            for key, value in diag.items()
            if key.startswith("hold_") and key != "hold_risk_floor"
        ),
        default=0,
    )
    if hold_risk_floor > 0 and hold_risk_floor >= other_holds:
        return "RISK_COLLAPSE"
    rejected_sl = int(diag.get("rejected_sl", 0))
    rejected_rr = int(diag.get("rejected_rr", 0))
    if rejected_sl > 0 and rejected_sl >= rejected_rr:
        return "STOP_GEOMETRY_FAILURE"
    if rejected_rr > rejected_sl:
        return "TARGET_GEOMETRY_FAILURE"
    return "FAIL_POLICY_COLLAPSE"


__all__ = ["assess_train_activity", "classify_zero_trade_cause"]
