from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from configuration import load_config
from training.callbacks.core import LagrangianRiskCallback
from training.setup.callbacks import _build_lagrangian_callback
from training.setup.models import _validate_q5_resume_state


class _Logger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.values[key] = value


def _model(state: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        n_envs=2,
        num_timesteps=2,
        _constrained_rl_state=state,
        logger=_Logger(),
    )


def test_q5_callback_penalizes_breach_and_updates_bounded_dual_state() -> None:
    callback = LagrangianRiskCallback(
        episode_cost_budget=0.05,
        dual_learning_rate=0.5,
        lambda_initial=2.0,
        lambda_max=3.0,
        gamma=1.0,
    )
    model = _model()
    callback.init_callback(model)  # type: ignore[arg-type]
    callback._on_training_start()

    rewards = np.array([1.0, 1.0], dtype=np.float32)
    callback.locals = {
        "rewards": rewards,
        "dones": np.array([False, False]),
        "infos": [
            {},
            {
                "constraints": {
                    "daily_cost": 0.02,
                    "max_loss_cost": 0.1,
                    "max_loss_breach": False,
                }
            },
        ],
    }
    assert callback._on_step()
    np.testing.assert_array_equal(rewards, np.array([1.0, 0.8], dtype=np.float32))

    next_rewards = np.array([1.0, 1.0], dtype=np.float32)
    callback.locals = {
        "rewards": next_rewards,
        "dones": np.array([False, True]),
        "infos": [{}, {"constraints": {"max_loss_cost": 0.2}}],
    }
    assert callback._on_step()
    np.testing.assert_array_equal(next_rewards, np.array([1.0, 0.6], dtype=np.float32))
    callback._on_rollout_end()

    assert callback.lagrange_multiplier == pytest.approx(2.125)
    assert model._constrained_rl_state["lagrange_multiplier"] == pytest.approx(2.125)
    assert model._constrained_rl_state["dual_updates"] == 1
    assert model.logger.values["constrained_rl/mean_completed_episode_cost"] == pytest.approx(0.3)


def test_q5_callback_restores_checkpoint_state_and_rejects_config_drift() -> None:
    original = LagrangianRiskCallback(episode_cost_budget=0.1, lambda_initial=1.0)
    model = _model()
    original.init_callback(model)  # type: ignore[arg-type]
    original._on_training_start()

    restored = LagrangianRiskCallback(episode_cost_budget=0.1, lambda_initial=1.0)
    restored.init_callback(_model(model._constrained_rl_state))  # type: ignore[arg-type]
    restored._on_training_start()
    assert restored.lagrange_multiplier == 1.0

    incompatible = LagrangianRiskCallback(episode_cost_budget=0.2, lambda_initial=1.0)
    incompatible.init_callback(_model(model._constrained_rl_state))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="differs from config"):
        incompatible._on_training_start()


def test_q5_only_accepts_full_split_ppo_not_mvp() -> None:
    cfg = load_config()
    assert _build_lagrangian_callback(cfg, algo="ppo", mvp=False) is None

    cfg.constrained_rl.enabled = True
    callback = _build_lagrangian_callback(cfg, algo="ppo", mvp=False)
    assert isinstance(callback, LagrangianRiskCallback)
    with pytest.raises(ValueError, match="PPO only"):
        _build_lagrangian_callback(cfg, algo="sac", mvp=False)
    with pytest.raises(ValueError, match="MVP is invalid"):
        _build_lagrangian_callback(cfg, algo="ppo", mvp=True)

    risk_terminated = OmegaConf.create(
        {
            "constrained_rl": {"enabled": True},
            "training": {"episode_mode": "risk_terminated"},
        }
    )
    with pytest.raises(ValueError, match="full_split"):
        _build_lagrangian_callback(risk_terminated, algo="ppo", mvp=False)


@pytest.mark.parametrize(
    "settings",
    [
        {"episode_cost_budget": float("nan")},
        {"dual_learning_rate": float("inf")},
        {"lambda_initial": float("nan")},
        {"lambda_max": float("inf")},
    ],
)
def test_q5_rejects_nonfinite_dual_settings(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must all be finite"):
        LagrangianRiskCallback(**settings)


def test_q5_discounted_cost_matches_ppo_reward_discount() -> None:
    callback = LagrangianRiskCallback(
        episode_cost_budget=0.05,
        dual_learning_rate=0.5,
        lambda_initial=2.0,
        lambda_max=3.0,
        gamma=0.9,
    )
    model = _model()
    callback.init_callback(model)  # type: ignore[arg-type]
    callback._on_training_start()
    callback.locals = {
        "rewards": np.array([0.0, 0.0], dtype=np.float32),
        "dones": np.array([False, False]),
        "infos": [{}, {"constraints": {"daily_cost": 0.1}}],
    }
    assert callback._on_step()
    callback.locals = {
        "rewards": np.array([0.0, 0.0], dtype=np.float32),
        "dones": np.array([False, True]),
        "infos": [{}, {"constraints": {"daily_cost": 0.2}}],
    }
    assert callback._on_step()
    callback._on_rollout_end()

    assert model.logger.values["constrained_rl/mean_completed_episode_cost"] == pytest.approx(0.28)
    assert callback.lagrange_multiplier == pytest.approx(2.115)


def test_q5_resume_is_rejected_without_training_environment_state() -> None:
    cfg = OmegaConf.create({"constrained_rl": {"enabled": True}})

    _validate_q5_resume_state(cfg, resume_checkpoint=None)
    with pytest.raises(ValueError, match="training environment/account state is not restored"):
        _validate_q5_resume_state(cfg, resume_checkpoint="checkpoint.zip")

    disabled = OmegaConf.create({"constrained_rl": {"enabled": False}})
    _validate_q5_resume_state(disabled, resume_checkpoint="checkpoint.zip")


class _TinyEnv(gym.Env[npt.NDArray[np.float32], np.int64]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space: spaces.Discrete[np.int64] = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


def test_q5_multiplier_state_survives_sb3_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = PPO("MlpPolicy", _TinyEnv(), n_steps=8, batch_size=4, n_epochs=1, device="cpu")
    setattr(
        model,
        "_constrained_rl_state",
        {
            "version": 2,
            "episode_cost_budget": 1.0,
            "dual_learning_rate": 0.1,
            "lambda_initial": 0.0,
            "lambda_max": 100.0,
            "gamma": 0.99,
            "lagrange_multiplier": 2.5,
            "dual_updates": 7,
        },
    )
    path = tmp_path / "q5_checkpoint"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    assert getattr(restored, "_constrained_rl_state") == getattr(model, "_constrained_rl_state")
