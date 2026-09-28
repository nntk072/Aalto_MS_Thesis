"""Tests for sequence encoder implementations (TCN, Transformer, GRU)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from quant_rl.models.agent import encoder_for
from quant_rl.models.encoder import GRUEncoder, MTFEncoder, TCNEncoder, TransformerEncoder


class TestTCNEncoder:
    """Tests for TCNEncoder class."""

    def test_tcn_observation_space(self) -> None:
        """Test TCNEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        encoder = TCNEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 32  # latent_dim + account embedding
        assert encoder.dilations == (1, 2, 4, 8)
        assert encoder.receptive_field == 61

    def test_tcn_forward(self) -> None:
        """Test TCNEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
            "account": torch.randn(batch_size, 6),
        }

        output = encoder(obs)

        assert output.shape == (batch_size, 128 + 32)


class TestTransformerEncoder:
    """Tests for TransformerEncoder class."""

    def test_transformer_observation_space(self) -> None:
        """Test TransformerEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        encoder = TransformerEncoder(
            observation_space=observation_space,
            seq_len=60,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 32

    def test_transformer_forward(self) -> None:
        """Test TransformerEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
            "account": torch.randn(8, 6),
        }

        output = encoder(obs)

        assert output.shape == (8, 128 + 32)


class TestGRUEncoder:
    """Tests for GRUEncoder class."""

    def test_gru_observation_space(self) -> None:
        """Test GRUEncoder with proper observation space."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        encoder = GRUEncoder(
            observation_space=observation_space,
            seq_len=128,
            n_features=64,
            latent_dim=128,
        )

        assert encoder.features_dim == 128 + 32

    def test_gru_forward(self) -> None:
        """Test GRUEncoder forward pass."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
            "account": torch.randn(8, 6),
        }

        output = encoder(obs)

        assert output.shape == (8, 128 + 32)

    def test_gru_parameters(self) -> None:
        """Test GRUEncoder has trainable parameters."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        encoder = GRUEncoder(observation_space=observation_space)

        total_params = sum(p.numel() for p in encoder.parameters())
        assert total_params > 0, "GRUEncoder should have trainable parameters"

    def test_gru_default_parameters(self) -> None:
        """Test GRUEncoder with default parameters."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(128, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
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
            "account": torch.randn(8, 6),
        }

        tcn_out = tcn(obs)
        transformer_out = transformer(obs)
        gru_out = gru(obs)

        assert tcn_out.shape == (8, 128 + 32)
        assert transformer_out.shape == (8, 128 + 32)
        assert gru_out.shape == (8, 128 + 32)

    def test_encoders_accept_dict_input(self) -> None:
        """Test that all encoders accept dict input."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        for EncoderClass in [TCNEncoder, TransformerEncoder, GRUEncoder]:
            encoder = EncoderClass(
                observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
            )

            obs = {
                "seq": torch.randn(4, 60, 64),
                "account": torch.randn(4, 6),
            }

            # Should not raise an error
            output = encoder(obs)
            assert output.shape[0] == 4

    def test_encoder_output_changes_with_seq(self) -> None:
        """Encoder output must depend on the sequence input."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )
        encoder = TCNEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        obs1 = {
            "seq": torch.randn(2, 60, 64),
            "account": torch.randn(2, 6),
        }
        obs2 = {
            "seq": torch.randn(2, 60, 64),
            "account": obs1["account"],
        }
        out1 = encoder(obs1)
        out2 = encoder(obs2)
        assert not torch.allclose(out1, out2)

    def test_encoder_output_changes_with_account(self) -> None:
        """Encoder output must depend on the account input."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )
        encoder = TransformerEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        obs1 = {
            "seq": torch.randn(2, 60, 64),
            "account": torch.randn(2, 6),
        }
        obs2 = {
            "seq": obs1["seq"],
            "account": torch.randn(2, 6),
        }
        out1 = encoder(obs1)
        out2 = encoder(obs2)
        assert not torch.allclose(out1, out2)

    def test_encoder_deterministic_tcn(self) -> None:
        """Same input must produce identical output."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )
        encoder = TCNEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        encoder.eval()
        seq = torch.randn(2, 60, 64)
        account = torch.randn(2, 6)
        obs = {"seq": seq, "account": account}
        with torch.no_grad():
            out1 = encoder(obs)
            out2 = encoder(obs)
        assert torch.allclose(out1, out2)

    def test_encoder_deterministic_transformer(self) -> None:
        """Same input must produce identical output."""
        observation_space = spaces.Dict(
            {
                "seq": spaces.Box(low=-1.0, high=1.0, shape=(60, 64), dtype=np.float32),
                "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )
        encoder = TransformerEncoder(
            observation_space=observation_space, seq_len=60, n_features=64, latent_dim=128
        )
        encoder.eval()
        seq = torch.randn(2, 60, 64)
        account = torch.randn(2, 6)
        obs = {"seq": seq, "account": account}
        with torch.no_grad():
            out1 = encoder(obs)
            out2 = encoder(obs)
        assert torch.allclose(out1, out2)


def _small_encoder(kind: str, timesteps: int = 12, n_features: int = 4) -> Any:
    observation_space = spaces.Dict(
        {
            "seq": spaces.Box(
                low=-np.inf, high=np.inf, shape=(timesteps, n_features), dtype=np.float32
            ),
            "account": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        }
    )
    if kind == "tcn":
        model: Any = TCNEncoder(
            observation_space,
            seq_len=timesteps,
            n_features=n_features,
            latent_dim=8,
            channels=(8, 8),
            kernel_size=3,
            dropout=0.0,
        )
    elif kind == "gru":
        model = GRUEncoder(
            observation_space,
            seq_len=timesteps,
            n_features=n_features,
            latent_dim=8,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )
    else:
        model = TransformerEncoder(
            observation_space,
            seq_len=timesteps,
            n_features=n_features,
            latent_dim=8,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        )
    model.eval()
    return model


def test_valid_suffix_ignores_left_pad_values() -> None:
    """Same valid suffix, pad 0 versus 9/-3/100. First real bar is position 0."""
    torch.manual_seed(0)
    timesteps, length, n_features = 12, 5, 4
    suffix = torch.randn(2, length, n_features)
    account = torch.randn(2, 6)
    zeros = torch.zeros(2, timesteps, n_features)
    zeros[:, -length:] = suffix
    poison = zeros.clone()
    poison[:, :-length] = torch.tensor([9.0, -3.0, 100.0, 0.0])
    mask = torch.zeros(2, timesteps)
    mask[:, -length:] = 1.0
    for kind in ("tcn", "gru", "transformer"):
        model = _small_encoder(kind, timesteps, n_features)
        with torch.no_grad():
            padded = model({"seq": zeros, "seq_mask": mask, "account": account})
            poisoned = model({"seq": poison, "seq_mask": mask, "account": account})
            short = model({"seq": suffix, "account": account})
        assert torch.allclose(padded, poisoned, atol=1e-5), kind
        assert torch.allclose(padded, short, atol=1e-5), kind


def test_mixed_valid_lengths_match_rowwise_suffix() -> None:
    torch.manual_seed(1)
    timesteps, n_features = 10, 4
    seq = torch.randn(2, timesteps, n_features)
    account = torch.randn(2, 6)
    lengths = (7, 4)
    mask = torch.zeros(2, timesteps)
    for row, length in enumerate(lengths):
        mask[row, -length:] = 1.0
    for kind in ("tcn", "gru", "transformer"):
        model = _small_encoder(kind, timesteps, n_features)
        with torch.no_grad():
            mixed = model({"seq": seq, "seq_mask": mask, "account": account})
            for row, length in enumerate(lengths):
                alone = model(
                    {
                        "seq": seq[row : row + 1, -length:],
                        "account": account[row : row + 1],
                    }
                )
                assert torch.allclose(mixed[row : row + 1], alone, atol=1e-5), kind


def _mtf_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "seq": spaces.Box(low=-1.0, high=1.0, shape=(8, 3), dtype=np.float32),
            "seq_mask": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "seq_m5": spaces.Box(low=-1.0, high=1.0, shape=(6, 2), dtype=np.float32),
            "mask_m5": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            "seq_m15": spaces.Box(low=-1.0, high=1.0, shape=(4, 1), dtype=np.float32),
            "mask_m15": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "seq_h1": spaces.Box(low=-1.0, high=1.0, shape=(4, 1), dtype=np.float32),
            "mask_h1": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        }
    )


def test_mtf_encoder_returns_160() -> None:
    encoder = MTFEncoder(_mtf_space(), latent_dim=128, channels=(8, 8), dropout=0.0)
    encoder.eval()
    batch = 2
    obs = {
        "seq": torch.randn(batch, 8, 3),
        "seq_mask": torch.ones(batch, 8),
        "seq_m5": torch.randn(batch, 6, 2),
        "mask_m5": torch.ones(batch, 6),
        "seq_m15": torch.randn(batch, 4, 1),
        "mask_m15": torch.ones(batch, 4),
        "seq_h1": torch.randn(batch, 4, 1),
        "mask_h1": torch.ones(batch, 4),
        "account": torch.randn(batch, 6),
    }
    assert encoder(obs).shape == (batch, 160)


def test_mtf_pad_prefix_does_not_change_the_branch() -> None:
    torch.manual_seed(0)
    encoder = MTFEncoder(_mtf_space(), latent_dim=128, channels=(8, 8), dropout=0.0)
    encoder.eval()
    obs = {
        "seq": torch.randn(2, 8, 3),
        "seq_mask": torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        ),
        "seq_m5": torch.randn(2, 6, 2),
        "mask_m5": torch.tensor([[0, 0, 1, 1, 1, 1]]).expand(2, -1).contiguous(),
        "seq_m15": torch.randn(2, 4, 1),
        "mask_m15": torch.ones(2, 4),
        "seq_h1": torch.zeros(2, 4, 1),
        "mask_h1": torch.zeros(2, 4),
        "account": torch.randn(2, 6),
    }
    with torch.no_grad():
        first = encoder(obs)
        shifted = {key: value.clone() for key, value in obs.items()}
        shifted["seq_m5"][:, :2] = 50.0
        shifted["seq"][:, :2] = -50.0
        shifted["seq_h1"][:, :3] = 7.0
        second = encoder(shifted)
    assert torch.allclose(first, second, atol=1e-5)


def test_tcn_arch_is_the_single_sequence_extractor() -> None:
    assert encoder_for("tcn") is TCNEncoder
    assert encoder_for("mtf") is MTFEncoder
    assert encoder_for("gru") is GRUEncoder
    assert encoder_for("transformer") is TransformerEncoder
