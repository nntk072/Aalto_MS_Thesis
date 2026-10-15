"""One-copy insert into SB3's dict rollout buffer.

SB3's ``DictRolloutBuffer.add`` does ``np.array(obs)`` and then assigns that
temporary into the buffer slot. Both steps copy the window. The parent does
this on every collect step, after the workers have already returned.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples


def install_single_copy_add(buffer: Any) -> None:
    """Replace ``buffer.add`` so each observation is copied once."""
    if getattr(buffer, "_single_copy_add", False):
        return
    if not isinstance(getattr(buffer, "observations", None), dict):
        return
    buffer.add = _single_copy_add.__get__(buffer, type(buffer))
    buffer._single_copy_add = True


def _single_copy_add(
    self: Any,
    obs: dict[str, np.ndarray[Any, Any]],
    action: np.ndarray[Any, Any],
    reward: np.ndarray[Any, Any],
    episode_start: np.ndarray[Any, Any],
    value: Any,
    log_prob: Any,
) -> None:
    if len(log_prob.shape) == 0:
        log_prob = log_prob.reshape(-1, 1)

    for key in self.observations.keys():
        src = obs[key]
        if isinstance(self.observation_space.spaces[key], spaces.Discrete):
            src = np.reshape(src, (self.n_envs,) + self.obs_shape[key])
        dst = self.observations[key][self.pos]
        if src.shape != dst.shape or src.dtype != dst.dtype:
            src = np.array(src, copy=True, dtype=dst.dtype).reshape(dst.shape)
        np.copyto(dst, src)

    action = np.reshape(action, (self.n_envs, self.action_dim))
    self.actions[self.pos] = np.array(action)
    self.rewards[self.pos] = np.array(reward)
    self.episode_starts[self.pos] = np.array(episode_start)
    self.values[self.pos] = value.clone().cpu().numpy().flatten()
    self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
    self.pos += 1
    if self.pos == self.buffer_size:
        self.full = True


class DeviceDictRolloutBuffer(DictRolloutBuffer):
    """Keep a flattened PPO rollout on the learner device during updates.

    Collection remains backed by the normal NumPy buffer. On the first
    ``get`` call after collection, this class stages each complete array once
    on ``self.device``. Minibatches then use device-side indexing across every
    PPO epoch. NumPy still generates the permutation so its sampling sequence
    matches ``DictRolloutBuffer``.
    """

    def reset(self) -> None:
        self._device_rollout: dict[str, torch.Tensor] = {}
        super().reset()

    def get(self, batch_size: int | None = None) -> Generator[DictRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        sample_count = self.buffer_size * self.n_envs
        indices = np.random.permutation(sample_count)

        if not self.generator_ready:
            for key, observations in self.observations.items():
                flattened = self.swap_and_flatten(observations)
                self._device_rollout[f"observation:{key}"] = torch.as_tensor(
                    flattened, device=self.device
                )
                self.observations[key] = flattened

            for name in ("actions", "values", "log_probs", "advantages", "returns"):
                flattened = self.swap_and_flatten(getattr(self, name))
                if name == "actions":
                    flattened = flattened.astype(np.float32, copy=False)
                self._device_rollout[name] = torch.as_tensor(flattened, device=self.device)
                setattr(self, name, flattened)
            self.generator_ready = True

        if batch_size is None:
            batch_size = sample_count
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        for start_idx in range(0, sample_count, batch_size):
            batch_indices = torch.as_tensor(
                indices[start_idx : start_idx + batch_size],
                dtype=torch.long,
                device=self.device,
            )
            yield DictRolloutBufferSamples(
                observations={
                    key: self._device_rollout[f"observation:{key}"].index_select(0, batch_indices)
                    for key in self.observations
                },
                actions=self._device_rollout["actions"].index_select(0, batch_indices),
                old_values=self._device_rollout["values"].index_select(0, batch_indices).flatten(),
                old_log_prob=self._device_rollout["log_probs"]
                .index_select(0, batch_indices)
                .flatten(),
                advantages=self._device_rollout["advantages"]
                .index_select(0, batch_indices)
                .flatten(),
                returns=self._device_rollout["returns"].index_select(0, batch_indices).flatten(),
            )
