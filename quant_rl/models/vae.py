"""Variational Autoencoder for full-day narrative embedding.

This module provides a VAE that compresses the entire pre-NY sequence
(01:05-16:29 UTC+3) into a latent "daily narrative" embedding (z).

The VAE is trained purely via reconstruction loss + KL divergence with no labels.
After training, the encoder is frozen and used as a fixed feature extractor
for the RL agent's observation space.

Architecture:
- Encoder: 1D-CNN or LSTM (or Transformer for longer sequences)
- Latent Space: z ∈ R^16 (continuous, Gaussian)
- Decoder: Symmetric 1D-CNN or LSTM that reconstructs the original sequence

Input: The entire 15.5-hour pre-NY window, downsampled to 5-minute OHLC bars (~186 timesteps)
Output: Latent vector z of shape (16,)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .encoder import ACCOUNT_DIM


class VAEEncoder(nn.Module):
    """VAE encoder that compresses pre-NY sequence into latent embedding.

    Uses 1D CNN layers to extract features from the sequence and produce
    mean and log-variance for the latent distribution.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence (number of 5-minute bars)
    n_features : int
        Number of features per bar (OHLCV = 5)
    latent_dim : int
        Dimension of the latent space (default: 16)
    channels : tuple[int, ...]
        Number of channels for each CNN layer
    kernel_size : int
        Kernel size for CNN layers
    stride : int
        Stride for CNN layers
    """

    def __init__(
        self,
        seq_len: int = 186,
        n_features: int = 5,
        latent_dim: int = 16,
        channels: tuple[int, ...] = (64, 32, 16),
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        # Encoder layers
        layers: list[nn.Module] = []
        in_channels = n_features
        for out_channels in channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.encoder_cnn = nn.Sequential(*layers)

        # Calculate the flattened size after CNN
        # This is a simplified calculation - actual size depends on padding and stride
        self._flattened_size = self._calculate_flattened_size(
            seq_len, n_features, channels, kernel_size, stride
        )

        # Fully connected layers to produce mu and log_var
        self.fc_mu = nn.Linear(self._flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self._flattened_size, latent_dim)

    def _calculate_flattened_size(
        self,
        seq_len: int,
        n_features: int,
        channels: tuple[int, ...],
        kernel_size: int,
        stride: int,
    ) -> int:
        """Calculate the flattened size after all CNN layers."""
        current_len = seq_len
        for _ in channels:
            # Simplified: assume padding maintains length for odd kernel sizes
            current_len = (current_len + 2 * (kernel_size // 2) - kernel_size) // stride + 1
        return channels[-1] * current_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_features)

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent distribution, shape (batch_size, latent_dim)
        log_var : torch.Tensor
            Log-variance of the latent distribution, shape (batch_size, latent_dim)
        """
        # x shape: [B, T, F] -> [B, F, T] for CNN
        x = x.transpose(1, 2)

        # Pass through CNN layers
        h: torch.Tensor = self.encoder_cnn(x)

        # Flatten
        h = h.flatten(start_dim=1)

        # Produce mu and log_var
        mu: torch.Tensor = self.fc_mu(h)
        log_var: torch.Tensor = self.fc_logvar(h)

        return mu, log_var


class VAEDecoder(nn.Module):
    """VAE decoder that reconstructs the sequence from latent embedding.

    Uses transposed 1D CNN layers to upsample the latent vector back to
    the original sequence length.

    Parameters
    ----------
    seq_len : int
        Length of the output sequence (number of 5-minute bars)
    n_features : int
        Number of features per bar (OHLCV = 5)
    latent_dim : int
        Dimension of the latent space (default: 16)
    channels : tuple[int, ...]
        Number of channels for each CNN layer (in reverse order of encoder)
    kernel_size : int
        Kernel size for CNN layers
    stride : int
        Stride for CNN layers
    """

    def __init__(
        self,
        seq_len: int = 186,
        n_features: int = 5,
        latent_dim: int = 16,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        # Fully connected layer to expand latent to CNN input size
        self.fc = nn.Linear(latent_dim, channels[0] * (seq_len // (2 ** len(channels))))

        # Decoder layers (transposed convolutions)
        layers: list[nn.Module] = []
        in_channels = channels[0]
        for i, out_channels in enumerate(channels[1:]):
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final layer to get back to n_features
        layers.append(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=n_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=stride - 1,
            )
        )

        self.decoder_cnn = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch_size, latent_dim)

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed sequence of shape (batch_size, seq_len, n_features)
        """
        # Expand latent to CNN input size
        h = self.fc(z)  # [B, C * T']
        h = h.unsqueeze(-1)  # [B, C * T', 1]

        # Reshape to [B, C, T']. Batch size must be explicit (at most one -1).
        c = self.decoder_cnn[0].in_channels  # number of decoder input channels (C)
        h = h.reshape(h.size(0), c, -1)  # [B, C, T']

        # x_recon shape: [B, F, T]
        x_recon: torch.Tensor = self.decoder_cnn(h)

        # Stride-2 transposed convs exactly double length, so the output may
        # not equal seq_len (e.g. 23 -> 46 -> 92 -> 184 for seq_len=186).
        # Resample to the exact target length to honour the output contract.
        if x_recon.size(2) != self.seq_len:
            x_recon = F.interpolate(x_recon, size=self.seq_len, mode="linear", align_corners=True)

        # [B, F, T] -> [B, T, F]
        x_recon = x_recon.transpose(1, 2)

        return x_recon


class VAE(nn.Module):
    """Variational Autoencoder for full-day narrative embedding.

    This VAE compresses the entire pre-NY sequence into a latent embedding z
    that captures the day's narrative (e.g., tight range vs. breakout).

    The model is trained using:
    - Reconstruction loss (MSE between input and output)
    - KL divergence loss (to regularize the latent space)

    After training, the encoder is frozen and used as a feature extractor
    for the RL agent.

    Parameters
    ----------
    seq_len : int
        Length of the sequence (number of 5-minute bars)
    n_features : int
        Number of features per bar (OHLCV = 5)
    latent_dim : int
        Dimension of the latent space (default: 16)
    encoder_channels : tuple[int, ...]
        Channels for encoder CNN layers
    decoder_channels : tuple[int, ...]
        Channels for decoder CNN layers
    kernel_size : int
        Kernel size for CNN layers
    stride : int
        Stride for CNN layers
    """

    def __init__(
        self,
        seq_len: int = 186,
        n_features: int = 5,
        latent_dim: int = 16,
        encoder_channels: tuple[int, ...] = (64, 32, 16),
        decoder_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = VAEEncoder(
            seq_len=seq_len,
            n_features=n_features,
            latent_dim=latent_dim,
            channels=encoder_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.decoder = VAEDecoder(
            seq_len=seq_len,
            n_features=n_features,
            latent_dim=latent_dim,
            channels=decoder_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, n_features)

        Returns
        -------
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, latent_dim)
        log_var : torch.Tensor
            Log-variance of latent distribution, shape (batch_size, latent_dim)
        """
        mu: torch.Tensor
        log_var: torch.Tensor
        mu, log_var = self.encoder(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed sequence.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch_size, latent_dim)

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed sequence of shape (batch_size, seq_len, n_features)
        """
        x_recon: torch.Tensor = self.decoder(z)
        return x_recon

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, latent_dim)
        log_var : torch.Tensor
            Log-variance of latent distribution, shape (batch_size, latent_dim)

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector, shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, n_features)

        Returns
        -------
        x_recon : torch.Tensor
            Reconstructed sequence, shape (batch_size, seq_len, n_features)
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, latent_dim)
        log_var : torch.Tensor
            Log-variance of latent distribution, shape (batch_size, latent_dim)
        """
        mu: torch.Tensor
        log_var: torch.Tensor
        mu, log_var = self.encode(x)
        z: torch.Tensor = self.reparameterize(mu, log_var)
        x_recon: torch.Tensor = self.decode(z)
        return x_recon, mu, log_var


class VAEFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor wrapper for VAE encoder.

    This wraps the VAE encoder to be compatible with Stable Baselines3
    feature extractors. It extracts the latent vector z from the pre-NY
    sequence and concatenates it with the account state (matching the
    TCNEncoder/GRUEncoder/TransformerEncoder pattern in encoder.py).

    Parameters
    ----------
    observation_space : spaces.Space
        The observation space of the environment
    vae : VAE
        The trained VAE model (encoder will be used)
    freeze : bool
        Whether to freeze the VAE encoder weights (default: True)
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        vae: VAE,
        freeze: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=vae.encoder.latent_dim + ACCOUNT_DIM,
        )
        self.vae = vae
        self.freeze = freeze

        if freeze:
            # Freeze all VAE parameters
            for param in vae.parameters():
                param.requires_grad = False

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations.

        Parameters
        ----------
        observations : dict[str, torch.Tensor]
            Dictionary containing 'pre_ny_seq' and 'account' keys

        Returns
        -------
        features : torch.Tensor
            Latent vector z concatenated with account state:
            (batch_size, latent_dim + ACCOUNT_DIM)
        """
        pre_ny_seq = observations["pre_ny_seq"]  # [B, T, F]
        account = observations["account"]  # [B, A]
        mu: torch.Tensor
        mu, _ = self.vae.encode(pre_ny_seq)
        return torch.cat([mu, account], dim=1)  # [B, latent_dim + A]
