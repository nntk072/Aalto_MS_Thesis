"""GRU sequence encoder."""

from __future__ import annotations

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


class GRUEncoder(BaseFeaturesExtractor):
    """GRU-based sequence encoder for RL.

    A 2-layer Gated Recurrent Unit that processes the last N volume bars.
    Simpler and more stable than Transformer but may suffer from memory decay
    and delayed reaction to sudden spikes.

    Parameters (pass via ``policy_kwargs["features_extractor_kwargs"]``)
    --------------------------------------------------------------------
    seq_len    : T – observation window length
    n_features : F – features per bar
    latent_dim : D – encoder output dimension
    hidden_size: GRU hidden state size
    num_layers : number of GRU layers
    dropout    : dropout rate between GRU layers
    """

    def __init__(
        self,
        observation_space: Any,
        seq_len: int = 128,
        n_features: int = 64,
        latent_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        account_dim: int | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        resolved_account_dim = _resolve_account_dim(observation_space, account_dim)
        self.account_mlp = AccountMLP(resolved_account_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = observations["seq"]  # [B, T, F]
        account = observations["account"]  # [B, A]
        lengths = _valid_lengths(seq, observations)

        def _encode(valid: torch.Tensor) -> torch.Tensor:
            hidden, _ = self.gru(valid)
            return cast(torch.Tensor, self.proj(hidden[:, -1, :]))

        latent = _encode_valid_suffix(seq, lengths, _encode)
        return torch.cat([latent, self.account_mlp(account)], dim=1)
