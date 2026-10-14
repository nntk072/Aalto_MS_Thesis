"""Regression coverage for vectorized environment attachment on PPO resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

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


def test_serial_backend_keeps_logical_environment_count() -> None:
    cfg = OmegaConf.create(
        {"env": {"n_envs": 3, "data_source": "bundle", "vector_backend": "serial"}}
    )
    parent_env = _make_cartpole()
    vec_env = _build_vector_env(parent_env, cfg, _make_cartpole)
    try:
        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 3
        assert not isinstance(vec_env, SubprocVecEnv)
    finally:
        vec_env.close()
        parent_env.close()


def test_serial_and_subprocess_backends_match_seeded_traces() -> None:
    serial_cfg = OmegaConf.create(
        {"env": {"n_envs": 2, "data_source": "bundle", "vector_backend": "serial"}}
    )
    subproc_cfg = OmegaConf.create(
        {"env": {"n_envs": 2, "data_source": "bundle", "vector_backend": "subproc"}}
    )
    serial_parent = _make_cartpole()
    subproc_parent = _make_cartpole()
    serial = _build_vector_env(serial_parent, serial_cfg, _make_cartpole)
    subproc = _build_vector_env(subproc_parent, subproc_cfg, _make_cartpole)
    try:
        serial.seed(314)
        subproc.seed(314)
        serial_obs = serial.reset()
        subproc_obs = subproc.reset()
        assert (serial_obs == subproc_obs).all()

        actions = [0, 1]
        for _ in range(25):
            serial_obs, serial_rewards, serial_dones, serial_infos = serial.step(actions)
            subproc_obs, subproc_rewards, subproc_dones, subproc_infos = subproc.step(actions)
            np.testing.assert_array_equal(serial_obs, subproc_obs)
            np.testing.assert_array_equal(serial_rewards, subproc_rewards)
            np.testing.assert_array_equal(serial_dones, subproc_dones)
            for serial_info, subproc_info in zip(serial_infos, subproc_infos, strict=True):
                assert serial_info.keys() == subproc_info.keys()
                for key in serial_info:
                    serial_value = serial_info[key]
                    subproc_value = subproc_info[key]
                    if isinstance(serial_value, np.ndarray):
                        np.testing.assert_array_equal(serial_value, subproc_value)
                    else:
                        assert serial_value == subproc_value
    finally:
        serial.close()
        subproc.close()
        serial_parent.close()
        subproc_parent.close()
