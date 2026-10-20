"""Regression coverage for vectorized environment attachment on PPO resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from learning.rl.agent import build_vector_env
from learning.rl.environment_adapter import ShardedBundleVecEnv


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
    resume_env = build_vector_env(_make_cartpole(), cfg, _make_cartpole)
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
    vec_env = build_vector_env(parent_env, cfg, _make_cartpole)
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
    serial = build_vector_env(serial_parent, serial_cfg, _make_cartpole)
    subproc = build_vector_env(subproc_parent, subproc_cfg, _make_cartpole)
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


def test_sharded_bundle_backend_matches_subprocess_traces_and_index_order() -> None:
    serial_cfg = OmegaConf.create(
        {"env": {"n_envs": 3, "data_source": "bundle", "vector_backend": "serial"}}
    )
    sharded_cfg = OmegaConf.create(
        {
            "env": {
                "n_envs": 3,
                "data_source": "bundle",
                "vector_backend": "sharded_bundle",
                "vector_shards": 2,
            }
        }
    )
    serial_parent = _make_cartpole()
    sharded_parent = _make_cartpole()
    serial = build_vector_env(serial_parent, serial_cfg, _make_cartpole)
    sharded = build_vector_env(sharded_parent, sharded_cfg, _make_cartpole)
    try:
        assert isinstance(sharded, ShardedBundleVecEnv)
        assert sharded.num_envs == 3
        assert len(sharded.processes) == 2
        assert sharded.num_shards == 2
        assert sharded.logical_envs_per_shard == [2, 1]
        serial.seed(314)
        sharded.seed(314)
        np.testing.assert_array_equal(serial.reset(), sharded.reset())

        assert serial.get_attr("np_random_seed", indices=[2, 0]) == [316, 314]
        assert sharded.get_attr("np_random_seed", indices=[2, 0]) == [316, 314]

        actions = np.asarray([0, 1, 0])
        for _ in range(80):
            serial_obs, serial_rewards, serial_dones, serial_infos = serial.step(actions)
            sharded_obs, sharded_rewards, sharded_dones, sharded_infos = sharded.step(actions)
            np.testing.assert_array_equal(serial_obs, sharded_obs)
            np.testing.assert_array_equal(serial_rewards, sharded_rewards)
            np.testing.assert_array_equal(serial_dones, sharded_dones)
            for serial_info, sharded_info in zip(serial_infos, sharded_infos, strict=True):
                assert serial_info.keys() == sharded_info.keys()
                for key, serial_value in serial_info.items():
                    sharded_value = sharded_info[key]
                    if isinstance(serial_value, np.ndarray):
                        np.testing.assert_array_equal(serial_value, sharded_value)
                    else:
                        assert serial_value == sharded_value
            assert serial.reset_infos == sharded.reset_infos
    finally:
        serial.close()
        sharded.close()
        serial_parent.close()
        sharded_parent.close()


def test_sharded_backend_requires_bundle_data_source() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "n_envs": 2,
                "data_source": "frames",
                "vector_backend": "sharded_bundle",
                "vector_shards": 1,
            }
        }
    )
    parent_env = _make_cartpole()
    try:
        try:
            build_vector_env(parent_env, cfg, _make_cartpole)
        except ValueError as exc:
            assert "requires env.data_source='bundle'" in str(exc)
        else:
            raise AssertionError("sharded_bundle must reject frame-backed environments")
    finally:
        parent_env.close()


def test_sharded_backend_auto_shards_never_exceed_logical_envs() -> None:
    cfg = OmegaConf.create(
        {"env": {"n_envs": 3, "data_source": "bundle", "vector_backend": "sharded_bundle"}}
    )
    parent_env = _make_cartpole()
    vec_env = build_vector_env(parent_env, cfg, _make_cartpole)
    try:
        assert isinstance(vec_env, ShardedBundleVecEnv)
        assert vec_env.num_shards == 3
        assert vec_env.logical_envs_per_shard == [1, 1, 1]
    finally:
        vec_env.close()
        parent_env.close()
