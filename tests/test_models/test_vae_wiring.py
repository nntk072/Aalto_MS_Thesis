"""Tests for VAE pre-NY helpers and feature-extractor wiring."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from quant_rl.models.vae import VAE, VAEFeatureExtractor
from quant_rl.models.vae_pre_ny import build_pre_ny_by_date, load_vae_from_checkpoint


def _m1_day(day: str, n: int = 20 * 60) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.linspace(0, 10, n)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100.0,
            "tickvol": 50,
        },
        index=idx,
    )


def test_build_pre_ny_by_date_shape() -> None:
    bars = pd.concat([_m1_day("2025-01-06"), _m1_day("2025-01-07")])
    by_date = build_pre_ny_by_date(bars, seq_len=186, n_features=5)
    assert date(2025, 1, 6) in by_date
    assert by_date[date(2025, 1, 6)].shape == (186, 5)


def test_load_vae_roundtrip(tmp_path: Path) -> None:
    model = VAE(seq_len=186, n_features=5, latent_dim=16)
    path = tmp_path / "vae.pth"
    torch.save(model.state_dict(), path)
    loaded = load_vae_from_checkpoint(path)
    x = torch.randn(2, 186, 5)
    with torch.no_grad():
        mu_a, _ = model.encode(x)
        mu_b, _ = loaded.encode(x)
    assert torch.allclose(mu_a, mu_b)


def test_vae_feature_extractor_accepts_vae_z() -> None:
    vae = VAE(seq_len=186, n_features=5, latent_dim=16)
    obs_space = spaces.Dict(
        {
            "vae_z": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "account": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
            "seq": spaces.Box(-np.inf, np.inf, shape=(60, 8), dtype=np.float32),
        }
    )
    extractor = VAEFeatureExtractor(obs_space, vae=vae, freeze=True)
    batch = {
        "vae_z": torch.randn(4, 16),
        "account": torch.randn(4, 5),
        "seq": torch.randn(4, 60, 8),
    }
    out = extractor(batch)
    assert out.shape == (4, 16 + 5)
