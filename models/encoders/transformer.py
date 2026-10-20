"""Causal Transformer sequence encoder."""

from __future__ import annotations

import math
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


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        return x + pe[:, : x.size(1)]


class TransformerEncoder(BaseFeaturesExtractor):
    """Causal Transformer encoder (standard PyTorch ``TransformerEncoderLayer``).

    An upper-triangular ``-inf`` mask makes each position attend only to ≤ t.

    Parameters (pass via ``policy_kwargs["features_extractor_kwargs"]``)
    --------------------------------------------------------------------
    seq_len        : T
    n_features     : F
    latent_dim     : D
    d_model        : internal transformer embedding dim
    nhead          : number of attention heads (d_model must be divisible)
    num_layers     : stacked encoder layers
    dim_feedforward: FFN hidden dim
    dropout        : dropout inside transformer layers
    """

    def __init__(
        self,
        observation_space: Any,
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        account_dim: int | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        resolved_account_dim = _resolve_account_dim(observation_space, account_dim)
        self.account_mlp = AccountMLP(resolved_account_dim)

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max(seq_len, 512))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, latent_dim)

        # Pre-compute causal mask once; resized in forward if needed
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        self.register_buffer("_causal_mask", mask)

    def _get_mask(self, t: int, device: torch.device) -> torch.Tensor:
        causal_mask = cast(torch.Tensor, self._causal_mask)
        if t <= causal_mask.size(0):
            return causal_mask[:t, :t]
        mask = torch.triu(torch.ones(t, t, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = torch.nan_to_num(observations["seq"], nan=0.0, posinf=0.0, neginf=0.0)
        account = observations["account"]  # [B, A]
        lengths = _valid_lengths(seq, observations)

        def _encode(valid: torch.Tensor) -> torch.Tensor:
            # Rows in one length group have no padding left, so only the causal mask.
            hidden_in = self.pos_enc(self.input_proj(valid))
            causal = self._get_mask(hidden_in.size(1), hidden_in.device)
            hidden = self.transformer(hidden_in, mask=causal)
            return cast(torch.Tensor, self.proj(hidden[:, -1, :]))

        latent = _encode_valid_suffix(seq, lengths, _encode)
        return torch.cat([latent, self.account_mlp(account)], dim=1)
