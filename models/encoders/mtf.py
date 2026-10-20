"""Multi-timeframe causal convolution encoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn

from models.encoders.common import (
    ACCOUNT_EMB_DIM,
    AccountMLP,
    BaseFeaturesExtractor,
    _encode_valid_suffix,
    _resolve_account_dim,
    _valid_lengths,
)
from models.encoders.tcn import _TemporalBlock

_MTF_STREAMS: tuple[tuple[str, str], ...] = (
    ("seq", "seq_mask"),
    ("seq_m5", "mask_m5"),
    ("seq_m15", "mask_m15"),
    ("seq_h1", "mask_h1"),
)


class _CausalTCN(nn.Module):
    """One causal TCN. The last valid step is projected to ``latent_dim``."""

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        channels: tuple[int, ...] = (128, 128, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dilations = tuple(2**i for i in range(len(channels)))
        layers: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            in_ch = n_features if i == 0 else channels[i - 1]
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilations[i], dropout))
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = _valid_lengths(seq, {"seq_mask": mask})

        def _encode(valid: torch.Tensor) -> torch.Tensor:
            hidden = self.tcn(valid.transpose(1, 2))
            return cast(torch.Tensor, self.proj(hidden[:, :, -1]))

        return _encode_valid_suffix(seq, lengths, _encode)


class MTFEncoder(BaseFeaturesExtractor):
    """Four causal TCNs, one per window, fused back to the 128-wide latent.

    The account embedding is unchanged, so ``features_dim`` stays 160 and the
    actor and critic are the same heads used by ``TCNEncoder``.
    """

    def __init__(
        self,
        observation_space: Any,
        latent_dim: int = 128,
        channels: tuple[int, ...] = (128, 128, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.1,
        account_dim: int | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        spaces_map = getattr(observation_space, "spaces", None)
        if not isinstance(spaces_map, Mapping):
            raise TypeError("MTFEncoder requires a Dict observation space")
        self.latent_dim = latent_dim
        self.channels = tuple(int(channel) for channel in channels)
        self.dilations = tuple(2**i for i in range(len(self.channels)))
        self.kernel_size = kernel_size
        resolved_account_dim = _resolve_account_dim(observation_space, account_dim)
        self.account_mlp = AccountMLP(resolved_account_dim)
        self._keys = _MTF_STREAMS
        self.branches = nn.ModuleList()
        for seq_key, _mask_key in _MTF_STREAMS:
            box = spaces_map[seq_key]
            shape = getattr(box, "shape", None)
            if shape is None or len(shape) < 2:
                raise TypeError(f"{seq_key} must be a Box")
            n_features = int(shape[1])
            self.branches.append(
                _CausalTCN(n_features, latent_dim, self.channels, kernel_size, dropout)
            )
        width = latent_dim * len(_MTF_STREAMS)
        self.fuse = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            branch(observations[seq_key], observations[mask_key])
            for branch, (seq_key, mask_key) in zip(self.branches, self._keys, strict=True)
        ]
        latent = self.fuse(torch.cat(parts, dim=1))
        return torch.cat([latent, self.account_mlp(observations["account"])], dim=1)
