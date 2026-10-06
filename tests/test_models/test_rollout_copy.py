"""The rollout buffer stores each observation with one copy."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer

from quant_rl.models.rollout_buffer import install_single_copy_add


def _space() -> spaces.Dict:
    return spaces.Dict(
        {
            "seq": spaces.Box(-np.inf, np.inf, (4, 3), dtype=np.float32),
            "account": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        }
    )


def test_single_copy_add_matches_sb3_and_keeps_source() -> None:
    obs_space = _space()
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    reference = DictRolloutBuffer(4, obs_space, action_space, device="cpu", n_envs=2)
    updated = DictRolloutBuffer(4, obs_space, action_space, device="cpu", n_envs=2)
    install_single_copy_add(updated)

    seq_arr = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    seq = np.ascontiguousarray(seq_arr[:, ::-1, :])  # non-contiguous window
    obs = {"seq": seq, "account": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    action = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    reward = np.array([0.5, -0.5], dtype=np.float32)
    starts = np.array([1.0, 0.0], dtype=np.float32)
    value = torch.tensor([[1.5], [2.5]])
    log_prob = torch.tensor([-0.25, -0.75])
    source = obs["seq"].copy()

    reference.add(obs, action, reward, starts, value, log_prob)
    updated.add(obs, action, reward, starts, value, log_prob)

    assert np.array_equal(updated.observations["seq"][0], reference.observations["seq"][0])
    assert np.array_equal(updated.observations["account"][0], reference.observations["account"][0])
    assert np.array_equal(updated.actions[0], reference.actions[0])
    assert np.array_equal(updated.rewards[0], reference.rewards[0])
    assert np.array_equal(updated.values[0], reference.values[0])
    assert np.array_equal(updated.log_probs[0], reference.log_probs[0])
    assert np.array_equal(obs["seq"], source)
    assert updated.pos == 1
