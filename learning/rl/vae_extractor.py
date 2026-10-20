"""SB3 feature-extractor adapter for the reusable VAE architecture."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.auxiliary.vae import VAE


class VAEFeatureExtractor(BaseFeaturesExtractor):
    """Expose the VAE latent and account vector through an SB3 extractor."""

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        vae: VAE,
        freeze: bool = True,
        account_dim: int | None = None,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("VAEFeatureExtractor requires a Dict observation space")
        account_space = observation_space.spaces.get("account")
        if not isinstance(account_space, spaces.Box) or account_space.shape is None:
            raise TypeError("observation space must contain a Box account vector")
        observed_account_dim = int(account_space.shape[0])
        if account_dim is not None and int(account_dim) != observed_account_dim:
            raise ValueError(
                f"account_dim {account_dim} does not match observation space width "
                f"{observed_account_dim}"
            )
        super().__init__(
            observation_space, features_dim=vae.encoder.latent_dim + observed_account_dim
        )
        self.vae = vae
        self.freeze = freeze
        if freeze:
            for parameter in vae.parameters():
                parameter.requires_grad = False

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode ``pre_ny_seq`` (or use precomputed ``vae_z``) and append account state."""
        account = observations["account"]
        if "vae_z" in observations:
            latent = observations["vae_z"]
        else:
            pre_ny_seq = observations["pre_ny_seq"]
            latent, _ = self.vae.encode(pre_ny_seq.to(next(self.vae.parameters()).device))
        if latent.device != account.device:
            latent = latent.to(account.device)
        return torch.cat([latent, account], dim=1)


__all__ = ["VAEFeatureExtractor"]
