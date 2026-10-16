"""Opt-in strict research eligibility must reach train and evaluation envs."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from quant_rl.config import load_config
from quant_rl.eval.rollout import evaluate_model
from quant_rl.train import train_rl


def test_strict_eligibility_config_override_propagates_to_env_and_eval(monkeypatch) -> None:
    cfg = load_config(["env.strict_research_eligibility=true"])
    assert cfg.env.strict_research_eligibility is True

    captured = {}

    def fake_env(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(train_rl, "TradingEnv", fake_env)
    empty = pd.DataFrame()
    train_rl.make_env(empty, empty, cfg, algo="ppo", reward="dsr")
    assert captured["strict_research_eligibility"] is True

    eval_kwargs = train_rl._build_eval_common(cfg)[0]
    assert eval_kwargs["strict_research_eligibility"] is True
    assert "strict_research_eligibility" in inspect.signature(evaluate_model).parameters


def test_legacy_default_remains_permissive() -> None:
    assert load_config().env.strict_research_eligibility is False


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
        ]
    )
    captured = {}
    monkeypatch.setattr(train_rl, "TradingEnv", lambda **kwargs: captured.update(kwargs))
    empty = pd.DataFrame()
    train_rl.make_env(empty, empty, cfg, algo="ppo", reward="pnl")
    evaluation_kwargs = train_rl._build_eval_common(cfg)[0]

    for key, expected in {
        "sl_mode": "trailing",
        "tp_mode": "breakeven",
        "tp_reference": "prior_session_extreme",
        "sl_mode_defer_threshold": 0.27,
        "tp_mode_defer_threshold": 0.31,
        "tp_breakeven_alpha": 0.42,
    }.items():
        assert captured[key] == expected
        assert evaluation_kwargs[key] == expected
    assert captured["eod_risk"] == evaluation_kwargs["eod_risk"]
    assert captured["allow_ema_exit"] is False
    assert evaluation_kwargs["allow_ema_exit"] is False
    assert captured["eod_risk"]["breakeven_trigger_r"] == 1.3
    assert captured["eod_risk"]["breakeven_buffer_pts"] == 1.7


def test_idea2_environment_rejects_missing_bias_and_ifvg_gate_inputs() -> None:
    cfg = load_config(
        [
            "env.strategy_actions=true",
            "strategy.name=distribution",
            "env.strict_research_eligibility=true",
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
        train_rl.make_env(bars, empty_features, cfg, algo="ppo", reward="pnl")
