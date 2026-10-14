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
