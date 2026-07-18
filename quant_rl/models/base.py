"""Typed interfaces for the encoder and policy.

YOU implement:
  - ``quant_rl/models/encoder.py``  (TCN or Transformer as SB3 BaseFeaturesExtractor)
  - ``quant_rl/models/agent.py``    (PPO policy wiring)

Observation contract
--------------------
``obs["seq"]``:     float32 tensor of shape ``[batch, T, F]``
``obs["account"]``: float32 tensor of shape ``[batch, A]``  (A = 5)

The encoder maps ``[batch, T, F]`` → ``[batch, D]`` (latent dim D is yours to choose).
The flattened input to the PPO policy head is ``[batch, D + A]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(ABC, nn.Module):
    """Abstract sequence encoder interface.

    Parameters
    ----------
    obs_dim : int
        Number of features F per time-step.
    seq_len : int
        Look-back window T.
    latent_dim : int
        Output embedding dimension D.
    """

    def __init__(self, obs_dim: int, seq_len: int, latent_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences.

        Parameters
        ----------
        seq : torch.Tensor
            Shape ``[batch, T, F]``.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, D]``.
        """


class Policy(ABC, nn.Module):
    """Abstract policy interface (wraps encoder + action head)."""

    @abstractmethod
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Map observation dict to action logits / distribution parameters."""
