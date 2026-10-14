"""Regression coverage for vectorized environment attachment on PPO resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from quant_rl.models.agent import _build_vector_env


def _make_cartpole() -> gym.Env[Any, Any]:
    return gym.make("CartPole-v1")


def test_resumed_ppo_uses_configured_vector_environment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "single_env_ppo"
    original = PPO("MlpPolicy", _make_cartpole(), n_steps=8, batch_size=8, device="cpu")
    original.save(checkpoint)
    original_env = original.get_env()
    assert original_env is not None
    original_env.close()

    cfg = OmegaConf.create({"env": {"n_envs": 2, "data_source": "frames"}})
    resume_env = _build_vector_env(_make_cartpole(), cfg, _make_cartpole)
    try:
        resumed = PPO.load(checkpoint, env=resume_env, device="cpu")
        assert resumed.n_envs == 2
        assert resumed.n_steps == 8
    finally:
        resume_env.close()
