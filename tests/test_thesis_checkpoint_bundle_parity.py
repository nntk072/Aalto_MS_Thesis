"""Checks for the thesis saved-checkpoint backend parity comparison."""

from __future__ import annotations

import pandas as pd
from scripts.thesis import audit_checkpoint_frame_bundle_parity as parity
from scripts.thesis.audit_checkpoint_frame_bundle_parity import compare_results, select_split


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


def test_select_split_keeps_in_sample_and_report_only_roles_distinct() -> None:
    train_bars = pd.DataFrame({"close": [1.0]})
    test_bars = pd.DataFrame({"close": [2.0]})
    train_features = pd.DataFrame({"x": [3.0]})
    test_features = pd.DataFrame({"x": [4.0]})

    selected_train = select_split("train", train_bars, test_bars, train_features, test_features)
    selected_test = select_split("test", train_bars, test_bars, train_features, test_features)

    assert selected_train[0] is train_bars
    assert selected_train[1] is train_features
    assert selected_train[2] == "in_sample_training"
    assert selected_test[0] is test_bars
    assert selected_test[1] is test_features
    assert selected_test[2] == "previously_inspected_selection_acknowledged_test"


def test_bundle_replay_preserves_overlay_action_vector(monkeypatch) -> None:
    captured = {}

    def fake_evaluate_model(model, bars, features, **kwargs):
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(parity, "evaluate_model", fake_evaluate_model)

    result = parity._evaluate_bundle_result(
        object(), object(), strategy_actions=True, continuous_actions=False
    )

    assert result == {"status": "captured"}
    assert captured["strategy_actions"] is True
    assert captured["continuous_actions"] is False
    assert captured["deterministic"] is True
