"""Opt-in strict research eligibility must reach train and evaluation envs."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from configuration import load_config
from evaluation.rollout.model import evaluate_model
from market_data.loading.ticks import TickBook
from training.setup import data as training_data
from training.setup import environments as training_environments


def test_strict_eligibility_config_override_propagates_to_env_and_eval(monkeypatch) -> None:
    cfg = load_config(["env.strict_research_eligibility=true"])
    assert cfg.env.strict_research_eligibility is True

    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(training_environments, "TradingEnv", fake_env)
    empty = pd.DataFrame()
    training_environments.make_env(empty, empty, cfg, algo="ppo", reward="dsr")
    assert captured["strict_research_eligibility"] is True
    assert captured["tick_execution_required"] is True

    eval_kwargs = training_environments._build_eval_common(cfg)[0]
    assert eval_kwargs["strict_research_eligibility"] is True
    assert "strict_research_eligibility" in inspect.signature(evaluate_model).parameters


def test_legacy_default_remains_permissive() -> None:
    assert load_config().env.strict_research_eligibility is False


def test_tick_execution_preflight_reports_train_and_test_gaps() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31T14:29:00Z"),
            pd.Timestamp("2025-12-31T14:30:00Z"),
            pd.Timestamp("2025-12-31T14:31:00Z"),
            pd.Timestamp("2025-12-31T14:32:00Z"),
            pd.Timestamp("2025-12-31T14:33:00Z"),
            pd.Timestamp("2025-12-31T14:34:00Z"),
            pd.Timestamp("2026-01-02T14:29:00Z"),
            pd.Timestamp("2026-01-02T14:30:00Z"),
            pd.Timestamp("2026-01-02T14:31:00Z"),
            pd.Timestamp("2026-01-02T14:32:00Z"),
            pd.Timestamp("2026-01-02T14:33:00Z"),
            pd.Timestamp("2026-01-02T14:34:00Z"),
        ]
    )
    train_mask = pd.Series(index.date <= pd.Timestamp("2025-12-31").date(), index=index)
    session_labels = pd.Series(
        ["closed", "ny", "ny", "ny", "closed", "london"]
        + ["closed", "ny", "ny", "ny", "closed", "london"],
        index=index,
    )
    tick_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31T14:31:00.030Z"),
            pd.Timestamp("2025-12-31T14:33:00.030Z"),
            pd.Timestamp("2026-01-02T14:31:00.030Z"),
            pd.Timestamp("2026-01-02T14:32:00.030Z"),
            pd.Timestamp("2026-01-02T14:33:00.030Z"),
        ]
    )
    tickbook = TickBook(
        tick_times.to_numpy(dtype="datetime64[ns]").astype("int64"),
        np.array([100.0, 101.0]),
        np.array([100.5, 101.5]),
    )

    reports = training_data._tick_execution_split_coverage(
        tickbook,
        index,
        train_mask,
        "2026-01-02",
        session_labels=session_labels,
        max_latency_ns_exclusive=30_000_000,
        max_quote_wait_ms=1_000,
    )

    assert reports["train"]["order_count"] == 3
    assert reports["train"]["unavailable_count"] == 1
    assert reports["train"]["first_unavailable_order_time"] == "2025-12-31T14:32:00+00:00"
    assert reports["test"]["order_count"] == 3
    assert reports["test"]["unavailable_count"] == 0


def test_tick_execution_schedule_freezes_observed_decisions_and_next_bar_quotes() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31T14:29:00Z"),
            pd.Timestamp("2025-12-31T14:30:00Z"),
            pd.Timestamp("2025-12-31T14:31:00Z"),
            pd.Timestamp("2025-12-31T14:32:00Z"),
            pd.Timestamp("2025-12-31T14:33:00Z"),
            pd.Timestamp("2025-12-31T14:34:00Z"),
            pd.Timestamp("2026-01-02T14:29:00Z"),
            pd.Timestamp("2026-01-02T14:30:00Z"),
            pd.Timestamp("2026-01-02T14:31:00Z"),
            pd.Timestamp("2026-01-02T14:32:00Z"),
            pd.Timestamp("2026-01-02T14:33:00Z"),
            pd.Timestamp("2026-01-02T14:34:00Z"),
        ]
    )
    train_mask = pd.Series(index.date <= pd.Timestamp("2025-12-31").date(), index=index)
    session_labels = pd.Series(
        ["closed", "ny", "ny", "ny", "closed", "london"]
        + ["closed", "ny", "ny", "ny", "closed", "london"],
        index=index,
    )

    schedule = training_data._tick_execution_split_schedule(
        index, train_mask, "2026-01-02", session_labels=session_labels
    )

    assert schedule["train"]["decision_positions"].tolist() == [1, 2, 3]
    assert schedule["train"]["quote_anchor_positions"].tolist() == [2, 3, 4]
    assert schedule["train"]["decision_times"].equals(index[[1, 2, 3]])
    assert schedule["train"]["quote_anchor_times"].equals(index[[2, 3, 4]])
    assert schedule["test"]["decision_positions"].tolist() == [7, 8, 9]
    assert schedule["test"]["quote_anchor_positions"].tolist() == [8, 9, 10]
    assert schedule["test"]["decision_times"].equals(index[[7, 8, 9]])
    assert schedule["test"]["quote_anchor_times"].equals(index[[8, 9, 10]])


def test_zero_latency_tick_preflight_checks_the_immediate_order_time() -> None:
    order_time = pd.Timestamp("2025-12-31T14:30:00Z")
    tickbook = TickBook(
        np.asarray([(order_time + pd.Timedelta(milliseconds=5)).value], dtype=np.int64),
        np.asarray([100.0]),
        np.asarray([100.2]),
    )

    report = training_data._tick_execution_split_coverage(
        tickbook,
        pd.DatetimeIndex([order_time]),
        pd.Series([True], index=pd.DatetimeIndex([order_time])),
        "2026-01-01",
        session_labels=pd.Series(["ny"], index=pd.DatetimeIndex([order_time])),
        max_latency_ns_exclusive=0,
        max_quote_wait_ms=1_000,
    )

    assert report["train"]["order_count"] == 1
    assert report["train"]["covered_count"] == 1
    assert report["train"]["unavailable_count"] == 0


def test_management_settings_match_training_and_evaluation_environment_kwargs(monkeypatch) -> None:
    cfg = load_config(
        [
            "risk.sl_mode=trailing",
            "risk.tp_mode=breakeven",
            "risk.tp_reference=prior_session_extreme",
            "env.sl_mode_defer_threshold=0.27",
            "env.tp_mode_defer_threshold=0.31",
            "env.tp_breakeven_alpha=0.42",
            "env.allow_ema_exit=false",
            "env.eod_risk.breakeven_trigger_r=1.3",
            "env.eod_risk.breakeven_buffer_pts=1.7",
            "execution.randomized_fill_latency_ms=30",
            "execution.max_tick_quote_wait_ms=45000",
        ]
    )
    captured = {}
    monkeypatch.setattr(
        training_environments, "TradingEnv", lambda **kwargs: captured.update(kwargs)
    )
    empty = pd.DataFrame()
    training_environments.make_env(empty, empty, cfg, algo="ppo", reward="pnl")
    evaluation_kwargs = training_environments._build_eval_common(cfg)[0]

    for key, expected in {
        "sl_mode": "trailing",
        "tp_mode": "breakeven",
        "tp_reference": "prior_session_extreme",
        "sl_mode_defer_threshold": 0.27,
        "tp_mode_defer_threshold": 0.31,
        "tp_breakeven_alpha": 0.42,
        "randomized_fill_latency_ms": 30.0,
        "max_tick_quote_wait_ms": 45_000,
        "tick_execution_required": True,
    }.items():
        assert captured[key] == expected
        assert evaluation_kwargs[key] == expected
    assert captured["eod_risk"] == evaluation_kwargs["eod_risk"]
    assert captured["allow_ema_exit"] is False
    assert evaluation_kwargs["allow_ema_exit"] is False
    assert captured["eod_risk"]["breakeven_trigger_r"] == 1.3
    assert captured["eod_risk"]["breakeven_buffer_pts"] == 1.7


def test_eval_environment_uses_campaign_seed_for_randomized_fill_latency() -> None:
    from types import SimpleNamespace

    cfg = load_config(["execution.randomized_fill_latency_ms=30"])
    args = SimpleNamespace(algo="ppo", reward="dsr", use_vae=False, seed=23)

    evaluation_kwargs = training_environments._build_eval_common(cfg, args=args)[0]

    assert evaluation_kwargs["execution_seed"] == 23


def test_idea2_environment_rejects_missing_bias_and_ifvg_gate_inputs() -> None:
    cfg = load_config(
        [
            "env.strategy_actions=true",
            "strategy.name=distribution",
            "env.strict_research_eligibility=true",
            "costs.use_tick_execution=false",
            "execution.randomized_fill_latency_ms=0.0",
        ]
    )
    index = pd.date_range("2024-01-02 14:30", periods=2, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "session": ["ny", "ny"],
        },
        index=index,
    )
    empty_features = pd.DataFrame(index=index)
    with pytest.raises(ValueError, match="po3_distribution_direction|htf_day_bias"):
        training_environments.make_env(bars, empty_features, cfg, algo="ppo", reward="pnl")
