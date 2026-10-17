"""CPU-only SB3 persistence contract for versioned action semantics."""

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO

from quant_rl.action_schema import build_action_schema, build_discrete_trading_action_schema
from quant_rl.models.agent import (
    bind_discrete_action_schema,
    validate_model_action_schema,
    validate_model_discrete_action_schema,
)


class _TinyEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}


class _TinyDiscreteEnv(gym.Env[npt.NDArray[np.float32], np.int64]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Discrete(20)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}


def test_version_and_layout_survive_sb3_save_load_without_training(tmp_path):
    schema = build_action_schema(has_direction=True)
    model = PPO("MlpPolicy", _TinyEnv(), n_steps=2, batch_size=2, device="cpu")
    setattr(model, "_action_schema_version", schema.version)
    setattr(model, "_action_layout_key", schema.key)
    path = tmp_path / "schema_saved.zip"
    model.save(path)

    loaded = PPO.load(path, device="cpu")
    assert getattr(loaded, "_action_schema_version") == schema.version
    assert getattr(loaded, "_action_layout_key") == schema.key
    validate_model_action_schema(loaded, schema)

    same_width_other_layout = build_action_schema(has_sl_mode=True)
    assert same_width_other_layout.width == schema.width
    with pytest.raises(ValueError, match="layout"):
        validate_model_action_schema(loaded, same_width_other_layout)


def test_discrete_action_contract_survives_sb3_save_load(tmp_path):
    schema = build_discrete_trading_action_schema()
    model = PPO("MlpPolicy", _TinyDiscreteEnv(), n_steps=2, batch_size=2, device="cpu")
    bind_discrete_action_schema(model, schema)
    path = tmp_path / "discrete_schema_saved.zip"
    model.save(path)

    loaded = PPO.load(path, device="cpu")

    assert validate_model_discrete_action_schema(loaded, schema, schema.to_dict()) == "verified"
    assert getattr(loaded, "_action_schema")["coordinates"] == list(schema.names)
