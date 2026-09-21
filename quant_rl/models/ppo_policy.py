"""PPO Gaussian policy with clamped log_std for Box overlay actions."""

from __future__ import annotations

from typing import Any

import torch

try:
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    _SB3_AVAILABLE = True
except ImportError:
    MultiInputActorCriticPolicy = object  # type: ignore[assignment,misc]
    _SB3_AVAILABLE = False


def clip_policy_log_std(policy: Any, log_std_min: float, log_std_max: float) -> bool:
    """Clamp ``policy.log_std`` in-place. Returns False when the param is absent."""
    log_std = getattr(policy, "log_std", None)
    if log_std is None:
        return False
    with torch.no_grad():
        log_std.clamp_(float(log_std_min), float(log_std_max))
    return True


class ClampedStdMultiInputPolicy(MultiInputActorCriticPolicy):
    """Multi-input PPO policy that clamps Gaussian ``log_std`` on every dist build.

    SB3's entropy bonus on ``Box[-1, 1]`` overlay actions otherwise drives
    ``std`` to thousands (seed 50 at 20M) and saturates tanh/env clipping.
    """

    def __init__(
        self,
        *args: Any,
        log_std_min: float = -2.0,
        log_std_max: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for ClampedStdMultiInputPolicy")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return data

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        clip_policy_log_std(self, self.log_std_min, self.log_std_max)
        return super()._get_action_dist_from_latent(latent_pi)
