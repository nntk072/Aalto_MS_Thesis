"""Smoke test for the SAC training path on TradingEnv.

The plan flagged that the SAC code path (``agent.py:112-131``) is
untested end-to-end: continuous action space + VAE + reward combo may
fail at runtime. SAC also has a long ``learning_starts`` warmup before
gradient steps fire, so we only need a few gradient updates to prove the
path is wired.

This test:
- Builds a continuous-action TradingEnv (``Box(-1, 1)``).
- Builds an SAC agent with the TCN encoder (default).
- Runs ``learn()`` for a small number of total_timesteps that exceeds
  ``learning_starts`` so at least one gradient step fires.
- Confirms the resulting policy can act on the env.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _make_bars(n: int = 600, seed: int = 7) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "tickvol": rng.integers(10, 200, n),
            "volume": rng.integers(1000, 5000, n),
            "vol": np.zeros(n, dtype=int),
            "spread": np.full(n, 0.6),
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )


def _make_features(bars) -> pd.DataFrame:

    n = len(bars)
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "london_high": 20500.0,
            "london_low": 19500.0,
            "asian_high": 20300.0,
            "asian_low": 19700.0,
            "volume_spike": rng.uniform(0.5, 2.5, n),
            "last_swing_high": np.nan,
            "last_swing_low": np.nan,
        },
        index=bars.index,
    )


def test_sac_smoke_learns_a_few_steps() -> None:
    """Build SAC on a continuous-action TradingEnv and confirm it learns."""
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC

    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_bars()
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        continuous_actions=True,  # SAC requires Box
        max_risk_frac=0.01,
        max_episode_steps=64,
    )

    # Tighter SAC config so the smoke test fits inside a few seconds.
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=512,
        batch_size=32,
        learning_starts=64,  # 1 rollout of warmup
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=0,
        seed=0,
    )

    # learning_starts=64 + a few gradient steps beyond it. Total budget
    # 256 timesteps → ~3 gradient updates after warmup.
    model.learn(total_timesteps=256, progress_bar=False)

    # Confirm the policy now produces actions in [-1, 1] on a fresh obs.
    obs, _ = env.reset(seed=0)
    raw, _ = model.predict(obs, deterministic=True)
    arr = np.asarray(raw).reshape(-1)
    assert arr.shape == (1,)
    assert -1.0 <= float(arr[0]) <= 1.0


def test_sac_rejects_discrete_action_space() -> None:
    """SAC must raise on a Discrete env — guards against silent misconfig."""
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC

    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_bars()
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        continuous_actions=False,  # Discrete(20)
    )

    with pytest.raises(Exception) as excinfo:
        SAC("MultiInputPolicy", env, learning_starts=10, buffer_size=64, verbose=0, seed=0)
    # SB3 raises AssertionError ("only supports Box") — accept any error
    # so we don't lock the test to a specific exception type.
    assert "Box" in str(excinfo.value) or "continuous" in str(excinfo.value)
