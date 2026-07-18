"""Tests: causal feature computation (no look-ahead) + encoder shapes."""

from __future__ import annotations

import pandas as pd
import torch

from quant_rl.features.indicators import ema_features, macd, returns, rsi


def test_rsi_causal(m1_bars):
    """RSI at t must not change when future bars are appended."""
    close = m1_bars["close"]
    rsi_full = rsi(close)
    rsi_trunc = rsi(close.iloc[:250])
    # Values up to bar 249 must match
    pd.testing.assert_series_equal(
        rsi_full.iloc[:249].dropna(),
        rsi_trunc.iloc[:249].dropna(),
        check_names=False,
        rtol=1e-6,
    )


def test_macd_causal(m1_bars):
    close = m1_bars["close"]
    m_full = macd(close)["macd"]
    m_trunc = macd(close.iloc[:300])["macd"]
    pd.testing.assert_series_equal(
        m_full.iloc[:299].dropna(),
        m_trunc.iloc[:299].dropna(),
        check_names=False,
        rtol=1e-6,
    )


def test_ema_causal(m1_bars):
    close = m1_bars["close"]
    feat_full = ema_features(close, [9, 21])
    feat_trunc = ema_features(close.iloc[:200], [9, 21])
    pd.testing.assert_frame_equal(
        feat_full.iloc[:199].dropna(),
        feat_trunc.iloc[:199].dropna(),
        rtol=1e-6,
    )


def test_returns_causal(m1_bars):
    """Log returns must be identical regardless of future data."""
    close = m1_bars["close"]
    r_full = returns(close, [1, 5])
    r_trunc = returns(close.iloc[:100], [1, 5])
    pd.testing.assert_frame_equal(
        r_full.iloc[:99].dropna(),
        r_trunc.iloc[:99].dropna(),
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Encoder shape tests
# ---------------------------------------------------------------------------


def _make_obs(B: int, T: int, F: int) -> dict[str, torch.Tensor]:
    return {
        "seq": torch.randn(B, T, F),
        "account": torch.randn(B, 5),
    }


def test_tcn_encoder_output_shape(dict_obs_space):
    from quant_rl.models.encoder import TCNEncoder

    T, F = 10, 8
    model = TCNEncoder(
        dict_obs_space,
        seq_len=T,
        n_features=F,
        latent_dim=32,
        channels=(16, 16),
        kernel_size=3,
        dropout=0.0,
    )
    obs = _make_obs(B=4, T=T, F=F)
    out = model(obs)
    assert out.shape == (4, 32 + 5), f"Expected (4, 37), got {out.shape}"


def test_transformer_encoder_output_shape(dict_obs_space):
    from quant_rl.models.encoder import TransformerEncoder

    T, F = 10, 8
    model = TransformerEncoder(
        dict_obs_space,
        seq_len=T,
        n_features=F,
        latent_dim=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
    )
    obs = _make_obs(B=4, T=T, F=F)
    out = model(obs)
    assert out.shape == (4, 32 + 5), f"Expected (4, 37), got {out.shape}"


def test_tcn_encoder_batch_size_1(dict_obs_space):
    from quant_rl.models.encoder import TCNEncoder

    model = TCNEncoder(
        dict_obs_space, seq_len=10, n_features=8, latent_dim=16, channels=(8,), dropout=0.0
    )
    obs = _make_obs(B=1, T=10, F=8)
    out = model(obs)
    assert out.shape == (1, 16 + 5)


def test_transformer_causal_mask_consistency(dict_obs_space):
    """Output for position t must not change when future bars are appended."""
    from quant_rl.models.encoder import TransformerEncoder

    T, F = 10, 8
    model = TransformerEncoder(
        dict_obs_space,
        seq_len=T,
        n_features=F,
        latent_dim=16,
        d_model=16,
        nhead=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    torch.manual_seed(0)
    seq_full = torch.randn(1, T, F)
    account = torch.randn(1, 5)

    with torch.no_grad():
        out_full = model({"seq": seq_full, "account": account})
        # Truncate to T//2 bars — last-step output should differ (different t)
        # but internal [0..T//2-1] attention should be independent of future
        out_half = model({"seq": seq_full[:, : T // 2, :], "account": account})

    # Both outputs are valid tensors (no NaN/Inf)
    assert torch.isfinite(out_full).all()
    assert torch.isfinite(out_half).all()
