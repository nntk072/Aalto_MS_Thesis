"""The rollout buffer stores each observation with one copy."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from learning.rl.rollout_buffer import DeviceDictRolloutBuffer, install_single_copy_add
from stable_baselines3.common.buffers import DictRolloutBuffer


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


def test_device_rollout_buffer_matches_sb3_minibatches_and_resets() -> None:
    obs_space = _space()
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    reference = DictRolloutBuffer(4, obs_space, action_space, device="cpu", n_envs=2)
    staged = DeviceDictRolloutBuffer(4, obs_space, action_space, device="cpu", n_envs=2)
    rng = np.random.default_rng(17)

    for key in reference.observations:
        reference.observations[key][:] = rng.standard_normal(
            reference.observations[key].shape, dtype=np.float32
        )
        staged.observations[key][:] = reference.observations[key]
    for name in ("actions", "values", "log_probs", "advantages", "returns"):
        target = getattr(reference, name)
        target[:] = rng.standard_normal(target.shape, dtype=np.float32)
        getattr(staged, name)[:] = target
    for buffer in (reference, staged):
        buffer.pos = buffer.buffer_size
        buffer.full = True

    np.random.seed(314)
    reference_batches = list(reference.get(batch_size=5))
    np.random.seed(314)
    staged_batches = list(staged.get(batch_size=5))

    assert len(reference_batches) == len(staged_batches)
    for reference_batch, staged_batch in zip(reference_batches, staged_batches, strict=True):
        for field in (
            "observations",
            "actions",
            "old_values",
            "old_log_prob",
            "advantages",
            "returns",
        ):
            reference_value = getattr(reference_batch, field)
            staged_value = getattr(staged_batch, field)
            if isinstance(reference_value, dict):
                assert reference_value.keys() == staged_value.keys()
                for key in reference_value:
                    assert torch.equal(reference_value[key], staged_value[key])
            else:
                assert torch.equal(reference_value, staged_value)

    assert staged._device_rollout
    staged.reset()
    assert staged._device_rollout == {}
    assert not staged.generator_ready
