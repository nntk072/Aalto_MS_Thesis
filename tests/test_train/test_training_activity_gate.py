from __future__ import annotations

import pytest

from training.callbacks.activity import assess_train_activity, classify_zero_trade_cause


@pytest.mark.parametrize(
    ("n_trades", "status", "ok"),
    [
        (0, "INACTIVE", False),
        (1, "ACTIVE_SPARSE", True),
        (4, "ACTIVE_SPARSE", True),
        (5, "ACTIVE", True),
    ],
)
def test_train_activity_classification(n_trades: int, status: str, ok: bool) -> None:
    gate = assess_train_activity(n_trades)
    assert gate["n_trades"] == n_trades
    assert gate["status"] == status
    assert gate["ok"] is ok


@pytest.mark.parametrize("n_trades", [-1, True, 1.5])
def test_train_activity_rejects_invalid_trade_count(n_trades: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        assess_train_activity(n_trades)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"entry_state_machine": True, "candidate_created": 0}, "FAIL_ENTRY_PATH"),
        (
            {"entry_state_machine": True, "candidate_created": 1, "armed_bars": 0},
            "FSM_ARMING_FAILURE",
        ),
        (
            {"entry_state_machine": True, "candidate_created": 1, "armed_bars": 2},
            "FAIL_POLICY_COLLAPSE",
        ),
        (
            {
                "entry_state_machine": True,
                "candidate_created": 1,
                "armed_bars": 2,
                "trigger_requested": 1,
                "trigger_refused": 1,
            },
            "FAIL_EXECUTION_GATE",
        ),
    ],
)
def test_zero_trade_diagnosis_uses_fsm_counters(state: dict[str, object], expected: str) -> None:
    assert classify_zero_trade_cause(0, state) == expected


def test_zero_trade_diagnosis_uses_existing_rejection_counters() -> None:
    assert classify_zero_trade_cause(0, entry_diag={"hold_risk_floor": 3}) == "RISK_COLLAPSE"
    assert classify_zero_trade_cause(0, entry_diag={"rejected_sl": 2}) == "STOP_GEOMETRY_FAILURE"
    assert classify_zero_trade_cause(0, entry_diag={"rejected_rr": 2}) == "TARGET_GEOMETRY_FAILURE"
    assert classify_zero_trade_cause(4) is None
