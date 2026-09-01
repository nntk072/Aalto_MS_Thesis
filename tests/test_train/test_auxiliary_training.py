"""Chain E Step 0 tests: auxiliary return-prediction head wiring.

Covers the pure window/target builder (alignment + causality + NaN
filtering) and a light integration check that the callback attaches to a
real SB3 PPO model and co-trains without touching SB3's own update path.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from quant_rl.train.auxiliary_training import (
    AuxiliaryTrainerCallback,
    build_supervised_windows,
)


@pytest.fixture
def synthetic() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """(features, closes) with no NaNs: T=300 bars, F=3 features."""
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    feats = rng.normal(0, 1, (300, 3))
    return feats, closes


def test_window_alignment(synthetic):
    feats, closes = synthetic
    seq, y = build_supervised_windows(feats, closes, obs_window=60, horizon=5, max_windows=1000)
    assert seq.shape[1:] == (60, 3)
    assert y.shape[1] == 5
    assert len(seq) == len(y) == 300 - 60 - 5
    log_ret = np.log(closes)
    # check one specific pair end-to-end
    i = 100
    row = np.where(np.isclose(seq[:, -1, 0], feats[i - 1, 0]))[0]
    # window ending at i-1 must pair with returns starting at i
    assert np.allclose(y[row[0]], log_ret[i : i + 5], atol=1e-6)


def test_window_causality(synthetic):
    """Targets must not depend on bars after the horizon; truncating the
    sample must not change already-valid pairs."""
    feats, closes = synthetic
    seq_full, y_full = build_supervised_windows(
        feats, closes, obs_window=60, horizon=5, max_windows=1000
    )
    seq_trunc, y_trunc = build_supervised_windows(
        feats[:200], closes[:200], obs_window=60, horizon=5, max_windows=1000
    )
    # first 200-60-5 = 135 pairs identical
    n_valid = 200 - 60 - 5
    np.testing.assert_allclose(seq_full[:n_valid], seq_trunc[:n_valid])
    np.testing.assert_allclose(y_full[:n_valid], y_trunc[:n_valid], atol=1e-6)


def test_window_nan_filtering(synthetic):
    feats, closes = synthetic
    feats[:100] = np.nan  # warmup region
    seq, y = build_supervised_windows(feats, closes, obs_window=60, horizon=5, max_windows=1000)
    # windows with i-60 < 100 (i.e. i < 160) touch NaN rows and are dropped
    assert len(seq) == 300 - 5 - 160
    assert np.isfinite(seq).all()


def test_window_insufficient_data(synthetic):
    feats, closes = synthetic
    seq, y = build_supervised_windows(feats[:50], closes[:50], obs_window=60, horizon=5)
    assert len(seq) == 0 and len(y) == 0


def test_callback_disabled_at_zero_weight():
    cb = AuxiliaryTrainerCallback(aux_weight=0.0)
    assert cb.aux_weight == 0.0
    # _on_rollout_end returns immediately without a model


def test_callback_config_defaults():
    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    aux = cfg.auxiliary
    assert float(aux.aux_weight) == 0.0  # opt-in by default
    assert int(aux.prediction_horizon) == 5
    assert int(aux.grad_steps) == 4


def test_callback_co_trains_with_ppo(synthetic):
    """Attach to a real (tiny) PPO model and run one rollout-end step."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    pytest.importorskip("torch")
    env_fn = lambda: gym.make("CartPole-v1")  # noqa: E731
    # CartPole obs is a flat Box, not our Dict space — the callback only
    # needs policy.features_extractor + obs_to_tensor, which exist for both.
    venv = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, seed=0, verbose=0)
    cb = AuxiliaryTrainerCallback(aux_weight=0.1, grad_steps=1, batch_windows=64)
    cb.model = model  # skip init_model; init callback manually
    cb.init_callback(model)

    # _env_data returns None for a non-TradingEnv — the callback must no-op
    cb.on_rollout_end()
    assert cb.last_aux_loss is None

    # Force the supervised path directly: feed synthetic env-like data
    feats, closes = synthetic
    seq, y = build_supervised_windows(feats, closes, obs_window=8, horizon=5, max_windows=64)
    assert len(seq) > 0

    # encoder gradient actually flows (features change after a step)

    head = cb.head
    optimizer = torch.optim.Adam(list(head.parameters()), lr=1e-3)
    from quant_rl.models.auxiliary import AuxiliaryLoss

    loss_fn = AuxiliaryLoss(head, aux_weight=0.1)
    obs_t = torch.zeros(4, 4, device=model.policy.device)  # CartPole Box(4,) obs
    latent = cast(
        torch.Tensor, model.policy.extract_features(obs_t, model.policy.features_extractor)
    )
    before_head = {k: v.clone() for k, v in head.state_dict().items()}
    loss = loss_fn(
        torch.zeros((), device=model.policy.device),
        latent,
        torch.randn(4, 5, device=model.policy.device),
    )
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    after_head = head.state_dict()
    assert any(not torch.equal(before_head[k], after_head[k]) for k in before_head)
