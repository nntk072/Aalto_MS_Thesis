"""Tests for sequence encoder implementations (TCN, Transformer, GRU)."""

from __future__ import annotations

import torch
from gymnasium import spaces

from quant_rl.models.encoder import GRUEncoder, TCNEncoder, TransformerEncoder


class TestTCNEncoder:
    """Tests for TCNEncoder class."""

    def test_tcn_observation_space(self) -> None:
        """Test TCNEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = TCNEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 5  # latent_dim + ACCOUNT_DIM

    def test_tcn_forward(self) -> None:
        """Test TCNEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = TCNEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        # Create dummy observations
        batch_size = 8
        obs = {
            "seq": torch.randn(batch_size, 60, 64),
            "account": torch.randn(batch_size, 5),
        }

        output = encoder(obs)

        assert output.shape == (batch_size, 128 + 5)


class TestTransformerEncoder:
    """Tests for TransformerEncoder class."""

    def test_transformer_observation_space(self) -> None:
        """Test TransformerEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = TransformerEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 5

    def test_transformer_forward(self) -> None:
        """Test TransformerEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = TransformerEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        obs = {
            "seq": torch.randn(8, 60, 64),
            "account": torch.randn(8, 5),
        }

        output = encoder(obs)

        assert output.shape == (8, 128 + 5)


class TestGRUEncoder:
    """Tests for GRUEncoder class."""

    def test_gru_observation_space(self) -> None:
        """Test GRUEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = GRUEncoder(
            observation_space=observation_space,
            seq_len=128,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 5

    def test_gru_forward(self) -> None:
        """Test GRUEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = GRUEncoder(
            observation_space=observation_space,
            seq_len=128,
            n_features=64,
            latent_dim=128,
        )

        obs = {
            "seq": torch.randn(8, 128, 64),
            "account": torch.randn(8, 5),
        }

        output = encoder(obs)

        assert output.shape == (8, 128 + 5)

    def test_gru_parameters(self) -> None:
        """Test GRUEncoder has trainable parameters."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = GRUEncoder(observation_space=observation_space)

        total_params = sum(p.numel() for p in encoder.parameters())
        assert total_params > 0, "GRUEncoder should have trainable parameters"

    def test_gru_default_parameters(self) -> None:
        """Test GRUEncoder with default parameters."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        encoder = GRUEncoder(observation_space=observation_space)

        # Check default values
        assert encoder.seq_len == 128
        assert encoder.n_features == 64
        assert encoder.latent_dim == 128
        assert encoder.hidden_size == 256
        assert encoder.num_layers == 2


class TestEncoderComparison:
    """Tests comparing different encoder implementations."""

    def test_encoders_same_output_shape(self) -> None:
        """Test that all encoders produce the same output shape."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        tcn = TCNEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        transformer = TransformerEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        gru = GRUEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )

        obs = {
            "seq": torch.randn(8, 60, 64),
            "account": torch.randn(8, 5),
        }

        tcn_out = tcn(obs)
        transformer_out = transformer(obs)
        gru_out = gru(obs)

        assert tcn_out.shape == (8, 128 + 5)
        assert transformer_out.shape == (8, 128 + 5)
        assert gru_out.shape == (8, 128 + 5)

    def test_encoders_accept_dict_input(self) -> None:
        """Test that all encoders accept dict input."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=float),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=float),
            }
        )

        for EncoderClass in [TCNEncoder, TransformerEncoder, GRUEncoder]:
            encoder = EncoderClass(
                observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
            )

            obs = {
                "seq": torch.randn(4, 60, 64),
                "account": torch.randn(4, 5),
            }

            # Should not raise an error
            output = encoder(obs)
            assert output.shape[0] == 4
