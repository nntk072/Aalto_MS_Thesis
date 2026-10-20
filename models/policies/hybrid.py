"""Categorical modes plus Gaussian continuous coordinates for PPO Box actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from stable_baselines3.common.distributions import Distribution as SB3Distribution
from torch import nn
from torch.distributions import Categorical, Normal

from models.policies.ppo import ClampedStdMultiInputPolicy, clip_policy_log_std

_CATEGORICAL_COORDINATES = {"sl_mode", "tp_mode", "exit"}


def _encoded_centers(name: str, threshold: float) -> tuple[float, ...]:
    """Return normalized raw-Box values that the existing env decoder maps exactly."""
    if name == "exit":
        return (-0.5, 0.5)  # structural, EMA-21
    if name in {"sl_mode", "tp_mode"}:
        if not 0.0 <= threshold < (1.0 / 6.0 - 1e-3):
            raise ValueError(f"hybrid {name} requires a defer threshold below 1/6, got {threshold}")
        # The center category means defer to the configured environment mode.
        # The breakeven value sits just outside the defer band but still decodes
        # to bucket one in the existing three-mode decoder.
        breakeven_u = 0.5 - threshold - 1e-3
        unit_values = (1.0 / 6.0, breakeven_u, 0.5, 5.0 / 6.0)
        return tuple(2.0 * value - 1.0 for value in unit_values)
    raise ValueError(f"unsupported categorical action coordinate: {name!r}")


class HybridActorHeads(nn.Module):
    """Separate categorical logits and Gaussian means over semantic coordinates."""

    def __init__(
        self, latent_dim: int, coordinates: Sequence[str], thresholds: dict[str, float]
    ) -> None:
        super().__init__()
        self.coordinates = tuple(str(value) for value in coordinates)
        self.categorical_indices = {
            name: index
            for index, name in enumerate(self.coordinates)
            if name in _CATEGORICAL_COORDINATES
        }
        if not self.categorical_indices:
            raise ValueError("hybrid distribution requires sl_mode, tp_mode, or exit coordinates")
        self.continuous_indices = tuple(
            index
            for index, name in enumerate(self.coordinates)
            if name not in self.categorical_indices
        )
        if not self.continuous_indices:
            raise ValueError("hybrid distribution requires at least one continuous coordinate")
        self.centers = {
            name: _encoded_centers(name, float(thresholds.get(name, 0.0)))
            for name in self.categorical_indices
        }
        self.continuous_mean = nn.Linear(latent_dim, len(self.continuous_indices))
        self.categorical_logits = nn.ModuleDict(
            {name: nn.Linear(latent_dim, len(centers)) for name, centers in self.centers.items()}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.continuous_mean.weight, gain=0.01)
        nn.init.zeros_(self.continuous_mean.bias)
        for layer in self.categorical_logits.values():
            if not isinstance(layer, nn.Linear):
                raise TypeError("categorical action head must be a linear layer")
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = {name: layer(latent) for name, layer in self.categorical_logits.items()}
        return self.continuous_mean(latent), logits


class HybridBoxDistribution(SB3Distribution):
    """Factorized categorical/Gaussian distribution encoded in the env's Box schema."""

    def __init__(self, actor: HybridActorHeads, action_width: int):
        super().__init__()  # type: ignore[no-untyped-call]
        self.actor = actor
        self.action_width = int(action_width)
        self.mean: torch.Tensor | None = None
        self.normal: Normal | None = None
        self.categorical: dict[str, Categorical] = {}

    def proba_distribution_net(self, *args: Any, **kwargs: Any) -> nn.Module:
        del args, kwargs
        raise NotImplementedError("hybrid heads are built by HybridActorHeads")

    def proba_distribution(
        self,
        means: torch.Tensor,
        log_std: torch.Tensor,
        logits: dict[str, torch.Tensor],
    ) -> HybridBoxDistribution:
        self.mean = means
        self.normal = Normal(means, log_std.exp().expand_as(means))
        self.categorical = {name: Categorical(logits=value) for name, value in logits.items()}
        return self

    def _pack(self, continuous: torch.Tensor, classes: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("call proba_distribution before sampling actions")
        output = self.mean.new_zeros((self.mean.shape[0], self.action_width))
        continuous_index = torch.as_tensor(
            self.actor.continuous_indices, dtype=torch.long, device=output.device
        )
        output = output.index_copy(1, continuous_index, continuous)
        for name, class_ids in classes.items():
            centers = self.actor.centers[name]
            center_tensor = output.new_tensor(centers)
            output[:, self.actor.categorical_indices[name]] = center_tensor[class_ids]
        return output

    def sample(self) -> torch.Tensor:
        if self.normal is None:
            raise RuntimeError("call proba_distribution before sampling actions")
        continuous = self.normal.rsample()
        classes = {
            name: cast(torch.Tensor, dist.sample())  # type: ignore[no-untyped-call]
            for name, dist in self.categorical.items()
        }
        return self._pack(continuous, classes)

    def mode(self) -> torch.Tensor:
        if self.normal is None:
            raise RuntimeError("call proba_distribution before choosing actions")
        classes = {name: dist.logits.argmax(dim=-1) for name, dist in self.categorical.items()}
        return self._pack(self.normal.mean, classes)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.normal is None:
            raise RuntimeError("call proba_distribution before evaluating actions")
        continuous_index = torch.as_tensor(
            self.actor.continuous_indices, dtype=torch.long, device=actions.device
        )
        result = cast(
            torch.Tensor,
            self.normal.log_prob(  # type: ignore[no-untyped-call]
                actions.index_select(1, continuous_index)
            ),
        ).sum(dim=1)
        for name, dist in self.categorical.items():
            index = self.actor.categorical_indices[name]
            centers = actions.new_tensor(self.actor.centers[name])
            classes = (actions[:, index : index + 1] - centers[None, :]).abs().argmin(dim=1)
            result = result + cast(
                torch.Tensor,
                dist.log_prob(classes),  # type: ignore[no-untyped-call]
            )
        return result

    def entropy(self) -> torch.Tensor:
        if self.normal is None:
            raise RuntimeError("call proba_distribution before evaluating entropy")
        result = cast(
            torch.Tensor,
            self.normal.entropy(),  # type: ignore[no-untyped-call]
        ).sum(dim=1)
        for dist in self.categorical.values():
            result = result + cast(
                torch.Tensor,
                dist.entropy(),  # type: ignore[no-untyped-call]
            )
        return result

    def actions_from_params(
        self,
        means: torch.Tensor,
        log_std: torch.Tensor,
        logits: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        self.proba_distribution(means, log_std, logits)
        return self.get_actions(deterministic)

    def log_prob_from_params(
        self,
        means: torch.Tensor,
        log_std: torch.Tensor,
        logits: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(means, log_std, logits)
        return actions, self.log_prob(actions)


class HybridClampedStdMultiInputPolicy(ClampedStdMultiInputPolicy):
    """SB3 PPO policy with categorical SL/TP/exit modes and Gaussian remainder."""

    def __init__(
        self,
        *args: Any,
        hybrid_action_coordinates: Sequence[str],
        sl_mode_defer_threshold: float = 0.1,
        tp_mode_defer_threshold: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.hybrid_action_coordinates = tuple(str(value) for value in hybrid_action_coordinates)
        self.hybrid_thresholds = {
            "sl_mode": float(sl_mode_defer_threshold),
            "tp_mode": float(tp_mode_defer_threshold),
        }
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        action_shape = getattr(self.action_space, "shape", None)
        if action_shape is None or len(action_shape) != 1:
            raise ValueError("hybrid PPO requires a one-dimensional Box action space")
        if len(self.hybrid_action_coordinates) != int(action_shape[0]):
            raise ValueError("hybrid coordinate count differs from action-space width")
        old_parameters = set(self.action_net.parameters()) | {self.log_std}
        actor_heads = HybridActorHeads(
            self.mlp_extractor.latent_dim_pi,
            self.hybrid_action_coordinates,
            self.hybrid_thresholds,
        )
        self.action_net = actor_heads
        continuous_count = len(actor_heads.continuous_indices)
        self.log_std = nn.Parameter(
            torch.full((continuous_count,), float(self.log_std_init), device=self.device)
        )
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if parameter not in old_parameters
            ]
        self.optimizer.add_param_group(
            {"params": list(self.action_net.parameters()) + [self.log_std], "lr": lr_schedule(1)}
        )
        self.hybrid_distribution = HybridBoxDistribution(actor_heads, int(action_shape[0]))

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> HybridBoxDistribution:
        clip_policy_log_std(self, self.log_std_min, self.log_std_max)
        means, logits = self.action_net(latent_pi)
        return self.hybrid_distribution.proba_distribution(means, self.log_std, logits)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            hybrid_action_coordinates=self.hybrid_action_coordinates,
            sl_mode_defer_threshold=self.hybrid_thresholds["sl_mode"],
            tp_mode_defer_threshold=self.hybrid_thresholds["tp_mode"],
        )
        return data


__all__ = [
    "HybridActorHeads",
    "HybridBoxDistribution",
    "HybridClampedStdMultiInputPolicy",
]
