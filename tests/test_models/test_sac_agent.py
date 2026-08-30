"""Tests for SAC agent wiring (plan 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_rl.models.agent import build_agent


def make_env(continuous: bool = True, n_bars: int = 120) -> Any:
    """Create a small TradingEnv for agent tests."""
    from quant_rl.envs.trading_env import TradingEnv

    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="1min")
    bars = pd.DataFrame(
        {
            "open": rng.uniform(100, 110, n_bars),
            "high": rng.uniform(105, 115, n_bars),
            "low": rng.uniform(90, 100, n_bars),
            "close": rng.uniform(95, 110, n_bars),
            "volume": rng.integers(1000, 5000, n_bars),
            "session_id": [0] * n_bars,
        },
        index=dates,
    )
    bars["high"] = bars[["open", "close", "high"]].max(axis=1)
    bars["low"] = bars[["open", "close", "low"]].min(axis=1)

    features = pd.DataFrame(index=bars.index)
    features["london_high"] = 112.0
    features["london_low"] = 92.0
    features["asian_high"] = 108.0
    features["asian_low"] = 96.0
    features["volume_spike"] = rng.uniform(0.5, 2.5, n_bars)

    return TradingEnv(
        bars,
        features,
        continuous_actions=continuous,
        obs_window=10,
        episodic=True,
    )


class TestSACAgentBuild:
    """Tests for building SAC agents via build_agent."""

    @pytest.fixture
    def cfg(self) -> Any:
        """Minimal OmegaConf config for SAC."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "env": {"obs_window": 10},
                "ppo": {"batch_size": 32},
                "sac": {
                    "learning_rate": 3e-4,
                    "buffer_size": 5000,
                    "batch_size": 16,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "ent_coef": 0.01,
                    "learning_starts": 200,
                },
            }
        )

    def test_build_sac_on_continuous_env(self, cfg: Any) -> None:
        """SAC builds successfully when the action space is Box(-1, 1)."""
        from stable_baselines3 import SAC

        env = make_env(continuous=True)
        model = build_agent(env, cfg, arch="tcn", algo="sac")

        assert isinstance(model, SAC)

    def test_sac_rejects_discrete_action_space(self, cfg: Any) -> None:
        """SAC requires a continuous Box action space."""
        env = make_env(continuous=False)
        with pytest.raises(ValueError, match="continuous"):
            build_agent(env, cfg, arch="tcn", algo="sac")

    def test_sac_uses_config_values(self, cfg: Any) -> None:
        """SAC hyperparameters come from the ``sac`` config section."""
        env = make_env(continuous=True)
        model = build_agent(env, cfg, arch="tcn", algo="sac")

        assert model.learning_rate == pytest.approx(cfg.sac.learning_rate)
        assert model.buffer_size == cfg.sac.buffer_size
        assert model.batch_size == cfg.sac.batch_size
        assert model.gamma == pytest.approx(cfg.sac.gamma)


class TestSACAgentAct:
    """Tests for SAC inference behaviour."""

    def test_act_returns_valid_continuous_actions(self) -> None:
        """model.predict returns actions within [-1, 1]."""
        from omegaconf import OmegaConf

        from quant_rl.models.agent import build_agent

        cfg = OmegaConf.create({"env": {"obs_window": 10}, "sac": {}, "ppo": {"batch_size": 32}})
        env = make_env(continuous=True)
        model = build_agent(env, cfg, arch="tcn", algo="sac")

        vec_env = model.get_env()
        assert vec_env is not None
        obs = vec_env.reset()

        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            assert action.shape == (1, 1)
            assert float(action[0][0]) >= -1.0 - 1e-6
            assert float(action[0][0]) <= 1.0 + 1e-6
            obs, _, _, _ = vec_env.step(np.array([action[0]], dtype=np.float32))

    def test_act_responds_to_nonzero_obs(self) -> None:
        """Encoder features should differ when observations differ."""
        from gymnasium import spaces

        from quant_rl.models.encoder import TCNEncoder

        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(10, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )
        encoder = TCNEncoder(
            observation_space=observation_space, seq_len=10, n_features=64, latent_dim=128
        )
        encoder.eval()

        obs1 = {
            "seq": np.zeros((1, 10, 64), dtype=np.float32),
            "account": np.zeros((1, 5), dtype=np.float32),
        }
        obs2 = {
            "seq": np.ones((1, 10, 64), dtype=np.float32),
            "account": np.ones((1, 5), dtype=np.float32),
        }

        import torch

        with torch.no_grad():
            out1 = encoder({k: torch.from_numpy(v) for k, v in obs1.items()})
            out2 = encoder({k: torch.from_numpy(v) for k, v in obs2.items()})
        assert not torch.allclose(out1, out2), (
            "Encoder features should differ when observations differ"
        )
