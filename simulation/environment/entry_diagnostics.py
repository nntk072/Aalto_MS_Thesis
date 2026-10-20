"""Entry-funnel counters and rejection audit records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulation.environment.market_view import FeatureRow


def _empty_entry_diag(entry_state_machine: bool = False) -> dict[str, int]:
    """Build per-episode rejection counters from one canonical definition."""
    diagnostics = {
        "hold_low_intensity": 0,
        "hold_no_context": 0,
        "hold_risk_floor": 0,
        "rejected_overnight": 0,
        "rejected_blackout": 0,
        "rejected_session_cap": 0,
        "rejected_cooldown": 0,
        "rejected_manipulation": 0,
        "rejected_gate": 0,
        "rejected_soft_brick": 0,
        "rejected_terminal": 0,
        "rejected_entry_pending": 0,
        "rejected_sl": 0,
        "rejected_rr": 0,
        "rejected_ema": 0,
        "rejected_sl_already_hit": 0,
        "opened": 0,
    }
    if entry_state_machine:
        diagnostics["rejected_entry_state"] = 0
    return diagnostics


_REJECTION_COUNTERS: dict[str, str] = {
    "overnight_boundary": "rejected_overnight",
    "open_blackout": "rejected_blackout",
    "session_cap": "rejected_session_cap",
    "entry_cooldown": "rejected_cooldown",
    "manipulation_unconfirmed": "rejected_manipulation",
    "entry_gate": "rejected_gate",
    "premium_block": "rejected_gate",
    "discount_block": "rejected_gate",
    "missing_previous_session": "rejected_gate",
    "soft_brick": "rejected_soft_brick",
    "entry_state_not_armed": "rejected_entry_state",
    "no_valid_sl": "rejected_sl",
    "no_valid_tp": "rejected_rr",
    "no_ema_exit": "rejected_ema",
    "sl_already_hit": "rejected_sl_already_hit",
    "terminal_no_fill_bar": "rejected_terminal",
    "entry_pending": "rejected_entry_pending",
}


def _previous_session_audit(
    strategy: object, row: FeatureRow | pd.Series, decision_close: float
) -> dict[str, Any]:
    """Return the previous-session reference used by the Idea 2 entry gate."""
    if not bool(getattr(strategy, "previous_session_pd_gate", False)):
        return {}
    code = float(row.get("previous_session_code", np.nan))
    code_to_label = {1: "asian", 2: "london", 3: "ny"}
    completed_ns = float(row.get("previous_session_completed_at_ns", np.nan))
    return {
        "decision_close": float(decision_close),
        "previous_session_high": float(row.get("previous_session_high", np.nan)),
        "previous_session_low": float(row.get("previous_session_low", np.nan)),
        "previous_session_midpoint": float(row.get("previous_session_midpoint", np.nan)),
        "previous_session_code": code,
        "previous_session_label": code_to_label.get(int(code), "") if np.isfinite(code) else "",
        "previous_session_completed_at_ns": int(completed_ns)
        if np.isfinite(completed_ns)
        else None,
    }


__all__ = ["_REJECTION_COUNTERS", "_empty_entry_diag", "_previous_session_audit"]
