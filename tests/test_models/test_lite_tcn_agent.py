from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from quant_rl.action_schema import build_discrete_trading_action_schema
from quant_rl.config import load_config
from quant_rl.models.agent import build_agent, validate_model_architecture
from quant_rl.models.encoder import LiteTCNEncoder

FloatArray = npt.NDArray[np.float32]


class _TinySequenceEnv(gym.Env[dict[str, FloatArray], np.int64]):
    observation_space = spaces.Dict(
        {
            "seq": spaces.Box(-1.0, 1.0, shape=(60, 4), dtype=np.float32),
            "seq_mask": spaces.Box(0.0, 1.0, shape=(60,), dtype=np.float32),
            "account": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
        }
    )
    action_space: spaces.Discrete[np.int64] = spaces.Discrete(20)
    action_schema = build_discrete_trading_action_schema()

    def _observation(self) -> dict[str, FloatArray]:
        return {
            "seq": np.zeros((60, 4), dtype=np.float32),
            "seq_mask": np.ones(60, dtype=np.float32),
            "account": np.zeros(6, dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, FloatArray], dict[str, Any]]:
        super().reset(seed=seed)
        return self._observation(), {}

    def step(
        self, action: np.int64
    ) -> tuple[dict[str, FloatArray], float, bool, bool, dict[str, Any]]:
        return self._observation(), 0.0, False, False, {}


def test_lite_tcn_is_configurable_in_build_agent_and_roundtrips(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = _TinySequenceEnv()
    model: PPO | None = None
    try:
        cfg = load_config()
        cfg.env.n_envs = 1
        cfg.env.obs_window = 60
        cfg.ppo.n_steps = 8
        cfg.ppo.batch_size = 8
        cfg.ppo.n_epochs = 1
        cfg.agent.lite_tcn.channels = 16
        cfg.agent.lite_tcn.dilations = [1, 4, 10]

        model = build_agent(env, cfg, arch="lite_tcn", algo="ppo", device="cpu")
        assert isinstance(model.policy.features_extractor, LiteTCNEncoder)
        assert model.policy.mlp_extractor.policy_net[0].out_features == 128
        model.learn(total_timesteps=16)
        assert model.num_timesteps == 16

        obs, _ = env.reset(seed=42)
        expected_action, _ = model.predict(obs, deterministic=True)
        checkpoint = tmp_path / "lite_tcn_ppo"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, device="cpu")
        validate_model_architecture(loaded, "lite_tcn")
        with pytest.raises(ValueError, match="does not match requested architecture 'tcn'"):
            validate_model_architecture(loaded, "tcn")
        actual_action, _ = loaded.predict(obs, deterministic=True)

        np.testing.assert_array_equal(expected_action, actual_action)
        assert torch.isfinite(torch.as_tensor(expected_action)).all()
    finally:
        if model is not None:
            model_env = model.get_env()
            if model_env is not None:
                model_env.close()
        env.close()
        torch.set_num_threads(previous_threads)
