"""Shared mechanics for reusable sequence encoders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn as nn
from decisioning.observation_state import BASE_ACCOUNT_DIM

ACCOUNT_DIM = BASE_ACCOUNT_DIM


ACCOUNT_EMB_DIM = 32


class Chomp1d(nn.Module):
    """Trim right-side convolution padding to preserve causal output length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[:, :, : -self.chomp_size].contiguous()


class BaseFeaturesExtractor(nn.Module):
    """Framework-neutral input/output contract for reusable encoders."""

    def __init__(self, observation_space: Any, features_dim: int) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.features_dim = features_dim


def _resolve_account_dim(observation_space: Any, account_dim: int | None) -> int:
    """Read the account input width from the configured observation space."""
    observation_spaces = getattr(observation_space, "spaces", None)
    if not isinstance(observation_spaces, Mapping):
        raise TypeError("sequence encoders require a Dict observation space")
    account_space = observation_spaces.get("account")
    shape = getattr(account_space, "shape", None)
    if shape is None or len(shape) != 1:
        raise TypeError("observation space must contain a Box account vector")
    observed_dim = int(shape[0])
    if account_dim is not None and int(account_dim) != observed_dim:
        raise ValueError(
            f"account_dim {account_dim} does not match observation space width {observed_dim}"
        )
    return observed_dim


class AccountMLP(nn.Module):
    """Map normalized account values to a fixed embedding."""

    def __init__(self, account_dim: int = BASE_ACCOUNT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(account_dim, ACCOUNT_EMB_DIM),
            nn.LayerNorm(ACCOUNT_EMB_DIM),
            nn.SiLU(),
            nn.Linear(ACCOUNT_EMB_DIM, ACCOUNT_EMB_DIM),
        )

    def forward(self, account: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(account))


def _valid_lengths(seq: torch.Tensor, observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-row count of real timesteps. A missing mask means the window is full."""
    batch, timesteps, _ = seq.shape
    mask = observations.get("seq_mask")
    if mask is None:
        return torch.full((batch,), timesteps, device=seq.device, dtype=torch.long)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1)
    lengths = mask.to(device=seq.device, dtype=torch.float32).sum(dim=-1).to(dtype=torch.long)
    return lengths.clamp(min=1, max=timesteps)


def _encode_valid_suffix(
    seq: torch.Tensor,
    lengths: torch.Tensor,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run ``encode_fn`` on each distinct valid suffix. Pad values never enter."""
    unique = torch.unique(lengths)
    if int(unique.numel()) == 1:
        length = int(unique[0].item())
        return encode_fn(seq[:, -length:, :])
    parts: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    for length in unique.tolist():
        idx = torch.nonzero(lengths == int(length), as_tuple=False).flatten()
        parts.append(encode_fn(seq.index_select(0, idx)[:, -int(length) :, :]))
        rows.append(idx)
    stacked = torch.cat(parts, dim=0)
    row_index = torch.cat(rows, dim=0)
    order = torch.empty(row_index.shape[0], dtype=torch.long, device=row_index.device)
    order[row_index] = torch.arange(row_index.shape[0], device=row_index.device)
    return stacked.index_select(0, order)
