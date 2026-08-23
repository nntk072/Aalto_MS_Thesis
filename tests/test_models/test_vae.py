"""Tests for VAE model implementation."""

from __future__ import annotations

import torch

from quant_rl.models.vae import VAE, VAEDecoder, VAEEncoder


class TestVAEEncoder:
    """Tests for VAEEncoder class."""

    def test_encoder_forward_shape(self) -> None:
        """Test that encoder produces correct output shapes."""
        seq_len = 186
        n_features = 5
        latent_dim = 16

        encoder = VAEEncoder(
            seq_len=seq_len,
            n_features=n_features,
            latent_dim=latent_dim,
            channels=(64, 32, 16),
            kernel_size=3,
            stride=2,
        )

        # Create dummy input
        batch_size = 8
        x = torch.randn(batch_size, seq_len, n_features)

        mu, log_var = encoder(x)

        assert mu.shape == (batch_size, latent_dim), (
            f"Expected mu shape ({batch_size}, {latent_dim}), got {mu.shape}"
        )
        assert log_var.shape == (batch_size, latent_dim), (
            f"Expected log_var shape ({batch_size}, {latent_dim}), got {log_var.shape}"
        )

    def test_encoder_output_type(self) -> None:
        """Test that encoder outputs are tensors."""
        encoder = VAEEncoder(seq_len=186, n_features=5, latent_dim=16)
        x = torch.randn(2, 186, 5)
        mu, log_var = encoder(x)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(log_var, torch.Tensor)


class TestVAEDecoder:
    """Tests for VAEDecoder class."""

    def test_decoder_forward_shape(self) -> None:
        """Test that decoder produces correct output shapes."""
        seq_len = 186
        n_features = 5
        latent_dim = 16

        decoder = VAEDecoder(
            seq_len=seq_len,
            n_features=n_features,
            latent_dim=latent_dim,
            channels=(16, 32, 64),
            kernel_size=3,
            stride=2,
        )

        # Create dummy input
        batch_size = 8
        z = torch.randn(batch_size, latent_dim)

        x_recon = decoder(z)

        assert x_recon.shape == (batch_size, seq_len, n_features), (
            f"Expected x_recon shape ({batch_size}, {seq_len}, {n_features}), got {x_recon.shape}"
        )

    def test_decoder_output_type(self) -> None:
        """Test that decoder output is a tensor."""
        decoder = VAEDecoder(seq_len=186, n_features=5, latent_dim=16)
        z = torch.randn(2, 16)
        x_recon = decoder(z)

        assert isinstance(x_recon, torch.Tensor)


class TestVAE:
    """Tests for VAE class."""

    def test_vae_encode_decode(self) -> None:
        """Test VAE encode and decode methods."""
        seq_len = 186
        n_features = 5
        latent_dim = 16

        vae = VAE(
            seq_len=seq_len,
            n_features=n_features,
            latent_dim=latent_dim,
            encoder_channels=(64, 32, 16),
            decoder_channels=(16, 32, 64),
            kernel_size=3,
            stride=2,
        )

        x = torch.randn(4, seq_len, n_features)

        # Test encode
        mu, log_var = vae.encode(x)
        assert mu.shape == (4, latent_dim)
        assert log_var.shape == (4, latent_dim)

        # Test decode
        z = torch.randn(4, latent_dim)
        x_recon = vae.decode(z)
        assert x_recon.shape == (4, seq_len, n_features)

    def test_vae_forward(self) -> None:
        """Test VAE forward pass."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)
        x = torch.randn(4, 186, 5)

        x_recon, mu, log_var = vae(x)

        assert x_recon.shape == (4, 186, 5)
        assert mu.shape == (4, 16)
        assert log_var.shape == (4, 16)

    def test_vae_reparameterize(self) -> None:
        """Test VAE reparameterization trick."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)

        mu = torch.randn(4, 16)
        log_var = torch.randn(4, 16)

        z = vae.reparameterize(mu, log_var)

        assert z.shape == (4, 16)
        assert isinstance(z, torch.Tensor)

    def test_vae_reconstruction_error(self) -> None:
        """Test that VAE can reconstruct input (basic sanity check)."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)
        x = torch.randn(4, 186, 5)

        x_recon, _, _ = vae(x)

        # Reconstruction should have the same shape as input
        assert x_recon.shape == x.shape

        # Reconstruction error should be finite
        recon_error = torch.nn.functional.mse_loss(x_recon, x)
        assert torch.isfinite(recon_error), f"Reconstruction error is not finite: {recon_error}"


class TestVAETraining:
    """Tests for VAE training behavior."""

    def test_vae_parameters(self) -> None:
        """Test that VAE has trainable parameters."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)

        # Count parameters
        total_params = sum(p.numel() for p in vae.parameters())
        assert total_params > 0, "VAE should have trainable parameters"

    def test_vae_trainable_by_default(self) -> None:
        """Test that VAE parameters are trainable by default."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)

        for param in vae.parameters():
            assert param.requires_grad, "VAE parameters should be trainable by default"

    def test_vae_freeze(self) -> None:
        """Test that VAE parameters can be frozen."""
        vae = VAE(seq_len=186, n_features=5, latent_dim=16)

        # Freeze all parameters
        for param in vae.parameters():
            param.requires_grad = False

        for param in vae.parameters():
            assert not param.requires_grad, "VAE parameters should be frozen"
