"""Semantic actor output heads for the continuous overlay-action schema."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from quant_rl.models.ppo_policy import ClampedStdMultiInputPolicy


def semantic_action_groups(coordinates: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Map the stable action coordinate names into four decision families."""
    groups: dict[str, list[int]] = {
        "entry": [],
        "risk_sl": [],
        "targets": [],
        "management": [],
    }
    for index, coordinate in enumerate(coordinates):
        if coordinate in {"direction", "intensity"}:
            groups["entry"].append(index)
        elif coordinate in {"stop", "risk", "sl_mode"}:
            groups["risk_sl"].append(index)
        elif coordinate in {
            "target",
            "tp1_sel",
            "tp2_sel",
            "tp3_sel",
            "tp_mode",
            "z1",
            "z2",
        }:
            groups["targets"].append(index)
        elif coordinate == "exit":
            groups["management"].append(index)
        else:
            raise ValueError(f"unassigned semantic action coordinate: {coordinate!r}")
    result = {name: tuple(indices) for name, indices in groups.items() if indices}
    if not result or sum(map(len, result.values())) != len(coordinates):
        raise ValueError("semantic actor groups must cover every action coordinate exactly once")
    return result


class SemanticActorHeads(nn.Module):
    """Independent output MLP per decision family, preserving Box coordinate order."""

    def __init__(
        self,
        latent_dim: int,
        coordinates: Sequence[str],
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.coordinates = tuple(str(value) for value in coordinates)
        self.groups = semantic_action_groups(self.coordinates)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, len(indices)),
                )
                for name, indices in self.groups.items()
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.heads.values():
            if not isinstance(module, nn.Sequential):
                raise TypeError("semantic actor head must be a sequential network")
            head = module
            first = head[0]
            final = head[2]
            if not isinstance(first, nn.Linear) or not isinstance(final, nn.Linear):
                raise TypeError("semantic actor head has an unexpected module layout")
            nn.init.orthogonal_(first.weight, gain=2**0.5)
            nn.init.zeros_(first.bias)
            nn.init.orthogonal_(final.weight, gain=0.01)
            nn.init.zeros_(final.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        output = latent.new_zeros((latent.shape[0], len(self.coordinates)))
        for name, indices in self.groups.items():
            values = self.heads[name](latent)
            index = torch.as_tensor(indices, dtype=torch.long, device=latent.device)
            output = output.index_copy(1, index, values)
        return output


class SemanticClampedStdMultiInputPolicy(ClampedStdMultiInputPolicy):
    """Current Gaussian PPO policy with independent semantic output heads."""

    def __init__(
        self,
        *args: Any,
        semantic_action_coordinates: Sequence[str],
        semantic_head_width: int = 64,
        **kwargs: Any,
    ) -> None:
        self.semantic_action_coordinates = tuple(
            str(value) for value in semantic_action_coordinates
        )
        self.semantic_head_width = int(semantic_head_width)
        if self.semantic_head_width <= 0:
            raise ValueError("semantic_head_width must be positive")
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        if not isinstance(self.action_net, nn.Linear):
            raise TypeError("semantic actor heads currently require a diagonal Gaussian Box policy")
        action_shape = getattr(self.action_space, "shape", None)
        if action_shape is None or len(action_shape) != 1:
            raise ValueError("semantic actor requires a one-dimensional Box action space")
        if len(self.semantic_action_coordinates) != int(action_shape[0]):
            raise ValueError("semantic coordinate count differs from action-space width")
        old_parameters = set(self.action_net.parameters())
        self.action_net = SemanticActorHeads(
            self.mlp_extractor.latent_dim_pi,
            self.semantic_action_coordinates,
            self.semantic_head_width,
        )
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"] if parameter not in old_parameters
            ]
        self.optimizer.add_param_group(
            {"params": list(self.action_net.parameters()), "lr": lr_schedule(1)}
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            semantic_action_coordinates=self.semantic_action_coordinates,
            semantic_head_width=self.semantic_head_width,
        )
        return data


__all__ = ["SemanticActorHeads", "SemanticClampedStdMultiInputPolicy", "semantic_action_groups"]
