"""One-copy insert into SB3's dict rollout buffer.

SB3's ``DictRolloutBuffer.add`` does ``np.array(obs)`` and then assigns that
temporary into the buffer slot. Both steps copy the window. The parent does
this on every collect step, after the workers have already returned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


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
