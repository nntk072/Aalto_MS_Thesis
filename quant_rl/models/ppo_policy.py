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


def install_post_update_std_clip(model: Any, log_std_min: float, log_std_max: float) -> None:
    """Clamp ``log_std`` after ``model.train`` and log the post-clamp std.

    Rollout samples are already clamped inside the distribution build. SB3
    records ``train/std`` from the parameter immediately after the optimizer
    step, before that next forward. This clip runs when ``train`` returns, so
    the entropy bonus does not keep a value outside the band, and the logged
    number is the one the next rollout will use.
    """
    orig = model.train

    def _train() -> None:
        orig()
        if not clip_policy_log_std(model.policy, log_std_min, log_std_max):
            return
        log_std = model.policy.log_std.detach()
        std = float(torch.exp(log_std).mean().item())
        logger = getattr(model, "logger", None)
        if logger is not None:
            logger.record("train/std", std)

    model.train = _train


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

    ``account_value_weight`` is a zero-start linear read of the raw account
    vector, added to the value MLP. Open PnL tracks the return target, and the
    shared embedding plus value MLP were collapsing to a constant.
    """

    def __init__(
        self,
        *args: Any,
        log_std_min: float = -0.7,
        log_std_max: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for ClampedStdMultiInputPolicy")
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        super().__init__(*args, **kwargs)
        # Optimizer is built inside super().__init__, so a parameter created
        # after that has to join the existing Adam group or it never moves.
        self.account_value_weight = torch.nn.Parameter(torch.zeros(6, device=self.device))
        group = {
            key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"
        }
        group["params"] = [self.account_value_weight]
        self.optimizer.add_param_group(group)

    def _account_value(self, obs: Any) -> torch.Tensor:
        account = obs["account"] if isinstance(obs, dict) else obs.account
        if not isinstance(account, torch.Tensor):
            account = torch.as_tensor(account, dtype=torch.float32, device=self.device)
        extra = account.float() @ self.account_value_weight
        return extra.unsqueeze(-1)

    def forward(
        self, obs: Any, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic=deterministic)
        return actions, values + self._account_value(obs), log_prob

    def predict_values(self, obs: Any) -> torch.Tensor:
        return super().predict_values(obs) + self._account_value(obs)

    def evaluate_actions(
        self, obs: Any, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        return values + self._account_value(obs), log_prob, entropy

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        return data

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        clip_policy_log_std(self, self.log_std_min, self.log_std_max)
        return super()._get_action_dist_from_latent(latent_pi)
