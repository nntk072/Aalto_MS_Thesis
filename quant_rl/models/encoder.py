"""Encoder stub for the user to implement.

This module provides a stub SB3 BaseFeaturesExtractor that raises
``NotImplementedError`` until you replace the body with your TCN or
Transformer encoder.

Shape contract
--------------
Input  : obs["seq"]     float32  [batch, T, F]   (flat in SB3 = [batch, T*F])
         obs["account"] float32  [batch, A]
Output : float32  [batch, latent_dim + A]  fed to SB3 MlpPolicy head.

How to fill this in
-------------------
1. Subclass or replace ``SequenceEncoder`` with your TCN / Transformer.
2. Override ``forward`` so it maps ``[batch, T, F]`` → ``[batch, D]``.
3. Set ``self.features_dim = latent_dim + account_dim`` so SB3 builds the
   correct policy head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ACCOUNT_DIM = 5


class SequenceEncoder(BaseFeaturesExtractor):
    """STUB – replace with your TCN/Transformer implementation.

    Parameters passed via ``policy_kwargs["features_extractor_kwargs"]``:
      - ``seq_len``   : T (obs window, default 60)
      - ``n_features``: F (number of features per bar)
      - ``latent_dim``: D (encoder output dim, your choice)
    """

    def __init__(
        self,
        observation_space,
        seq_len: int = 60,
        n_features: int = 64,
        latent_dim: int = 128,
    ) -> None:
        features_dim = latent_dim + ACCOUNT_DIM
        super().__init__(observation_space, features_dim=features_dim)

        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        # ----------------------------------------------------------------
        # TODO: replace the placeholder below with your encoder layers
        # e.g.:
        #   self.encoder = TCN(n_features, latent_dim, ...)
        #   self.encoder = TransformerEncoder(n_features, latent_dim, ...)
        # ----------------------------------------------------------------
        self._placeholder = nn.Identity()  # keeps the module non-empty

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Implement SequenceEncoder.forward in quant_rl/models/encoder.py.\n"
            "Expected: observations dict with keys 'seq' [B,T,F] and 'account' [B,A].\n"
            "Return: tensor [B, latent_dim + ACCOUNT_DIM]."
        )
