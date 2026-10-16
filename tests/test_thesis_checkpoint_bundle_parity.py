"""Checks for the thesis saved-checkpoint backend parity comparison."""

from __future__ import annotations

import pandas as pd
from scripts.thesis.audit_checkpoint_frame_bundle_parity import compare_results


def _result() -> dict[str, object]:
    return {
        "equity": pd.Series([100_000.0, 99_900.0], index=pd.date_range("2025-01-01", periods=2)),
        "trades": pd.DataFrame([{"type": "open", "level_type": "london_low", "pnl": -100.0}]),
        "account": {"equity": 99_900.0},
        "breaches": [],
        "breach_events": [],
        "n_sessions": 1,
        "n_breach_sessions": 0,
        "n_sessions_with_trades": 1,
        "n_sessions_skipped": 0,
        "days_traded": 1,
        "survived_full_year": True,
        "reward_sum": -0.1,
        "reward_attribution": {"dsr": -0.1},
        "n_steps": 1,
        "fail_time": None,
        "max_trailing_dd": 0.001,
        "action_counts": {"-1": 1},
        "entry_diag": {"opened": 1},
        "entry_state_diagnostics": {},
    }


def test_compare_results_accepts_identical_frame_and_bundle_outputs() -> None:
    frame_result = _result()
    bundle_result = _result()

    comparisons = compare_results(frame_result, bundle_result)

    assert comparisons
    assert all(comparisons.values())


def test_compare_results_detects_trade_metadata_divergence() -> None:
    frame_result = _result()
    bundle_result = _result()
    bundle_result["trades"] = pd.DataFrame(
        [{"type": "open", "level_type": "asian_low", "pnl": -100.0}]
    )

    comparisons = compare_results(frame_result, bundle_result)

    assert comparisons["equity"] is True
    assert comparisons["trades"] is False
    assert all(value for key, value in comparisons.items() if key != "trades")
