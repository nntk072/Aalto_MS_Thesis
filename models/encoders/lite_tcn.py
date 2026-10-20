"""Lightweight depthwise-separable temporal convolution encoder."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from models.encoders.common import (
    ACCOUNT_EMB_DIM,
    AccountMLP,
    BaseFeaturesExtractor,
    Chomp1d,
    _encode_valid_suffix,
    _resolve_account_dim,
    _valid_lengths,
)


class _DepthwiseSeparableConv1d(nn.Module):
    """Causal temporal convolution with depthwise and pointwise factors."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.chomp = Chomp1d(padding)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.chomp(x)
        x = self.activation(x)
        x = self.dropout(x)
        return cast(torch.Tensor, self.pointwise(x))


class _LiteTemporalBlock(nn.Module):
    """Residual block with two depthwise-separable causal convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            _DepthwiseSeparableConv1d(channels, kernel_size, dilation, dropout),
            _DepthwiseSeparableConv1d(channels, kernel_size, dilation, dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(x + self.temporal(x)))


class LiteTCNEncoder(BaseFeaturesExtractor):
    """Compact causal depthwise-separable TCN with the standard 160-wide output.

    The default three blocks use two kernel-3 convolutions at dilations 1, 4,
    and 10. This gives a 61-bar receptive field while preserving the normal
    128-value sequence latent and 32-value account embedding.
    """

    def __init__(
        self,
        observation_space: Any,
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 4, 10),
        kernel_size: int = 3,
        dropout: float = 0.1,
        account_dim: int | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("LiteTCN dilations must be a nonempty sequence of positive integers")
        if channels < 1 or kernel_size < 1:
            raise ValueError("LiteTCN channels and kernel_size must be positive")
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.channels = channels
        self.dilations = tuple(int(dilation) for dilation in dilations)
        self.kernel_size = kernel_size
        resolved_account_dim = _resolve_account_dim(observation_space, account_dim)
        self.account_mlp = AccountMLP(resolved_account_dim)
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            _LiteTemporalBlock(channels, kernel_size, dilation, dropout)
            for dilation in self.dilations
        )
        self.proj = nn.Linear(channels, latent_dim)

    @property
    def receptive_field(self) -> int:
        """Return the number of input bars visible at a block's last timestep."""
        return int(1 + 2 * (self.kernel_size - 1) * sum(self.dilations))

    def _encode_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Return every causal hidden state as ``[batch, time, channels]``."""
        hidden = self.input_proj(seq.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return cast(torch.Tensor, hidden.transpose(1, 2))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = observations["seq"]
        account = observations["account"]
        lengths = _valid_lengths(seq, observations)

        def _encode(valid: torch.Tensor) -> torch.Tensor:
            hidden = self._encode_hidden(valid)
            return cast(torch.Tensor, self.proj(hidden[:, -1, :]))

        latent = _encode_valid_suffix(seq, lengths, _encode)
        return torch.cat([latent, self.account_mlp(account)], dim=1)
