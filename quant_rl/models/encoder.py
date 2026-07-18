"""Sequence encoders as SB3 feature extractors.

Two architectures are provided:
  - ``TCNEncoder``         – dilated causal temporal convolution network (default)
  - ``TransformerEncoder`` – causal self-attention encoder

Both consume the Gymnasium ``Dict`` observation space and return a flat feature
vector that SB3 feeds directly to the PPO policy/value MLP heads.

Shape contract
--------------
Input  : obs["seq"]     float32  [batch, T, F]
         obs["account"] float32  [batch, A]   (A = ACCOUNT_DIM = 5)
Output : float32  [batch, latent_dim + A]

Switch architecture via ``agent.build_agent(env, cfg, arch="transformer")``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ACCOUNT_DIM = 5


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
        return self.relu(out + res)


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
        observation_space,
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        channels: tuple[int, ...] = (64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        tcn_layers: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            in_ch = n_features if i == 0 else channels[i - 1]
            tcn_layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
        self.tcn = nn.Sequential(*tcn_layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = observations["seq"]          # [B, T, F]
        account = observations["account"]  # [B, A]
        h = self.tcn(seq.transpose(1, 2))  # [B, F, T] → [B, C, T]
        latent = self.proj(h[:, :, -1])    # last (causal) step  [B, D]
        return torch.cat([latent, account], dim=1)  # [B, D+A]


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
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


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
        observation_space,
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim=latent_dim + ACCOUNT_DIM)
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max(seq_len, 512))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, latent_dim)

        # Pre-compute causal mask once; resized in forward if needed
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        self.register_buffer("_causal_mask", mask)

    def _get_mask(self, t: int, device: torch.device) -> torch.Tensor:
        if t <= self._causal_mask.size(0):  # type: ignore[attr-defined]
            return self._causal_mask[:t, :t]  # type: ignore[attr-defined]
        mask = torch.triu(torch.ones(t, t, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = observations["seq"]          # [B, T, F]
        account = observations["account"]  # [B, A]
        x = self.pos_enc(self.input_proj(seq))            # [B, T, d_model]
        mask = self._get_mask(x.size(1), x.device)
        h = self.transformer(x, mask=mask)                # [B, T, d_model]
        latent = self.proj(h[:, -1, :])                   # [B, D]
        return torch.cat([latent, account], dim=1)        # [B, D+A]


# Default alias (TCN is faster to train; swap to Transformer for ablation)
SequenceEncoder = TCNEncoder
