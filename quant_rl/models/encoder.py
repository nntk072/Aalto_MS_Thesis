"""Sequence encoders as SB3 feature extractors.

Two architectures are provided:
  - ``TCNEncoder``         – dilated causal temporal convolution network (default)
  - ``TransformerEncoder`` – causal self-attention encoder

Both consume the Gymnasium ``Dict`` observation space and return a flat feature
vector that SB3 feeds directly to the PPO policy/value MLP heads.

Shape contract
--------------
Input  : obs["seq"]      float32  [batch, T, F]
         obs["seq_mask"] float32  [batch, T]   (optional; 1 = real bar)
         obs["account"]  float32  [batch, A]   (A = ACCOUNT_DIM = 6, already normalized)
Output : float32  [batch, latent_dim + ACCOUNT_EMB_DIM]

Switch architecture via ``agent.build_agent(env, cfg, arch="transformer")``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ACCOUNT_DIM = 6
ACCOUNT_EMB_DIM = 32


class AccountMLP(nn.Module):
    """Map the six normalized account values to a fixed embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ACCOUNT_DIM, ACCOUNT_EMB_DIM),
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


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------


class _Chomp1d(nn.Module):
    """Trim future padding to enforce strict causality."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """Dilated causal conv block with residual connection and weight-norm."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
            ),
            _Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return cast(torch.Tensor, self.relu(out + res))


# ---------------------------------------------------------------------------
# TCNEncoder
# ---------------------------------------------------------------------------


class TCNEncoder(BaseFeaturesExtractor):
    """Dilated causal TCN that maps a sequence window to a latent vector.

    Receptive field with ``L`` levels, kernel ``k``, grows as
    ``2^0 + … + 2^{L-1}) * (k-1) + 1`` bars — easily covers T=60.

    Parameters (pass via ``policy_kwargs["features_extractor_kwargs"]``)
    --------------------------------------------------------------------
    seq_len    : T – observation window length
    n_features : F – features per bar
    latent_dim : D – encoder output dimension
    channels   : conv channels at each dilation level
    kernel_size: conv kernel width
    dropout    : dropout rate inside each temporal block
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        channels: tuple[int, ...] = (128, 128, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dilations = tuple(2**i for i in range(len(channels)))
        self.account_mlp = AccountMLP()

        tcn_layers: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            in_ch = n_features if i == 0 else channels[i - 1]
            tcn_layers.append(
                _TemporalBlock(in_ch, out_ch, kernel_size, self.dilations[i], dropout)
            )
        self.tcn = nn.Sequential(*tcn_layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    @property
    def receptive_field(self) -> int:
        """Bars visible at the last step. Two kernel-sized convs per dilation."""
        return int(1 + 2 * (self.kernel_size - 1) * sum(self.dilations))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = observations["seq"]  # [B, T, F]
        account = observations["account"]  # [B, A]
        lengths = _valid_lengths(seq, observations)

        def _encode(valid: torch.Tensor) -> torch.Tensor:
            hidden = self.tcn(valid.transpose(1, 2))
            return cast(torch.Tensor, self.proj(hidden[:, :, -1]))

        latent = _encode_valid_suffix(seq, lengths, _encode)
        return torch.cat([latent, self.account_mlp(account)], dim=1)


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------


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
        observation_space: spaces.Space[Any],
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.account_mlp = AccountMLP()

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


# ---------------------------------------------------------------------------
# GRUEncoder
# ---------------------------------------------------------------------------


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
        observation_space: spaces.Space[Any],
        seq_len: int = 128,
        n_features: int = 64,
        latent_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_EMB_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.account_mlp = AccountMLP()
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


# Default alias (TCN is faster to train; swap to Transformer/GRU for ablation)
SequenceEncoder = TCNEncoder
