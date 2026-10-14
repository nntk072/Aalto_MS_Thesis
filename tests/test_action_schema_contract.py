"""The environment decoder and action audit share one flag-derived schema."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from quant_rl.action_schema import build_action_schema
from quant_rl.envs.trading_env import TradingEnv, _decision_from_action


@pytest.mark.parametrize(
    "flags",
    [
        {"has_sl_mode": True, "has_multi_tp": True},
        {"has_sl_mode": True, "has_tp_mode": True, "has_multi_tp": True},
        {"has_sl_mode": True, "has_tp_mode": True, "has_multi_tp": True, "has_simplex": True},
        {
            "has_direction": True,
            "has_sl_mode": True,
            "has_tp_mode": True,
            "has_multi_tp": True,
            "has_simplex": True,
        },
    ],
)
def test_decision_log_uses_action_schema_indices(flags: dict[str, bool]) -> None:
    schema = build_action_schema(**flags)
    raw = np.linspace(-0.9, 0.9, schema.width, dtype=np.float32)
    unit = (raw + 1.0) / 2.0
    record = _decision_from_action(raw, unit, schema)
    for name, idx in schema.indices.items():
        if name == "target":
            assert record["action_target"] == pytest.approx(raw[idx])
        else:
            assert record[f"action_{name}"] == pytest.approx(raw[idx])
    assert record["exit_u"] == pytest.approx(unit[schema.indices["exit"]])


def test_decoder_rejects_action_shape_that_disagrees_with_flags() -> None:
    schema = build_action_schema(has_sl_mode=True, has_tp_mode=True, has_multi_tp=True)
    env: Any = SimpleNamespace(strategy_actions=True, action_schema=schema)
    with pytest.raises(ValueError, match="action width"):
        TradingEnv._decode_action(env, np.zeros(schema.width - 1, dtype=np.float32))


def test_eval_common_preserves_multi_tp_without_simplex() -> None:
    """A 9-D config must survive the train/eval configuration boundary."""
    from omegaconf import OmegaConf

    from quant_rl.train.train_rl import _build_eval_common

    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    cfg.env.strategy_actions = True
    cfg.env.allow_multi_tp = True
    cfg.env.allow_agent_tp_mode = True
    cfg.env.allow_simplex = False
    kwargs, *_ = _build_eval_common(cfg)
    assert kwargs["allow_simplex"] is False
