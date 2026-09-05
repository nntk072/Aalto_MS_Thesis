"""End-to-end smoke test for the PPO training pipeline.

Runs a tiny PPO agent on a synthetic TradingEnv for a small number of
timesteps to confirm the full stack wires up correctly:

- TradingEnv accepts a Dict observation space and Discrete(20) actions.
- SB3 PPO with ``MultiInputPolicy`` builds and runs ``learn()`` without
  throwing on the env, features, or feature extractor.
- max_episode_steps truncation fires as expected so a small synthetic
  dataset never produces a runaway rollout.
- A loss / return is reported by PPO so we can assert the agent actually
  took gradient steps.

This is a *smoke* test: it does not aim to verify trading profitability,
only that the end-to-end integration is free of obvious runtime errors.
Slow on CPU; marked ``slow`` so it is skipped in fast unit-test runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.slow


def _make_synthetic_bars(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Generate a minimal OHLC bar DataFrame for the env."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(rng.normal(0, 2, n))
    df = pd.DataFrame(
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
    df.index.name = "datetime"
    return df


def _make_synthetic_features(bars: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Generate a feature matrix the env can consume without warnings."""
    rng = np.random.default_rng(seed)
    n = len(bars)
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


@pytest.fixture
def synthetic_env():
    """Build a TradingEnv + a matching bars/features split for PPO smoke."""
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_synthetic_bars()
    features = _make_synthetic_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
        max_episode_steps=64,
    )
    return env


def test_trading_env_max_episode_truncation_fires(synthetic_env) -> None:
    """The env should report truncated=True after exactly max_episode_steps."""
    env = synthetic_env
    obs, _ = env.reset(seed=0)
    truncated = False
    done = False
    steps = 0
    while not (done or truncated):
        _, _, done, truncated, _ = env.step(0)
        steps += 1
        if steps > 1000:  # hard safety net
            pytest.fail("Episode did not terminate within 1000 steps")
    assert truncated is True
    assert steps == 64
    assert done is False  # truncation is not a terminal failure


@pytest.mark.parametrize("arch", ["tcn", "gru", "transformer"])
def test_ppo_smoke_runs_and_takes_gradient_steps(synthetic_env, arch: str) -> None:
    """Build a tiny PPO on a real TradingEnv and confirm gradients flow.

    We don't assert a specific Sharpe or PnL — PPO on a synthetic random-walk
    dataset in 256 steps has no realistic chance of profit. We only assert:
    - the model builds without raising,
    - ``learn()`` returns a trained model whose PPO internals recorded
      policy/value losses (proof that gradient steps actually executed),
    - the trained policy still acts deterministically on the env without
      crashing.
    """
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")

    from stable_baselines3 import PPO

    env = synthetic_env
    n_features = env.observation_space["seq"].shape[1]

    extractor_cls: type[Any]
    extractor_kwargs: dict[str, Any]
    if arch == "transformer":
        from quant_rl.models.encoder import TransformerEncoder

        extractor_cls = TransformerEncoder
        extractor_kwargs = dict(
            seq_len=10,
            n_features=n_features,
            latent_dim=128,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
        )
    elif arch == "gru":
        from quant_rl.models.encoder import GRUEncoder

        extractor_cls = GRUEncoder
        extractor_kwargs = dict(
            seq_len=10,
            n_features=n_features,
            latent_dim=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
        )
    else:
        from quant_rl.models.encoder import TCNEncoder

        extractor_cls = TCNEncoder
        extractor_kwargs = dict(
            seq_len=10,
            n_features=n_features,
            latent_dim=128,
        )

    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
        policy_kwargs=dict(
            features_extractor_class=extractor_cls,
            features_extractor_kwargs=extractor_kwargs,
        ),
    )

    # 256 total timesteps is the plan's specified smoke budget. n_steps=64
    # yields 4 rollouts; the second rollout triggers a PPO update.
    model.learn(total_timesteps=256, progress_bar=False)

    # SB3 records training scalars on the model's logger. We just need
    # proof that the loss-bearing keys exist and are finite scalars — PPO
    # surfaces ``train/policy_gradient_loss`` and ``train/value_loss`` on
    # every update.
    recorded = model.logger.name_to_value
    assert "train/policy_gradient_loss" in recorded, (
        f"PPO did not record policy gradient loss. Saw: {sorted(recorded)[:10]}"
    )
    pg_loss = float(recorded["train/policy_gradient_loss"])
    assert np.isfinite(pg_loss), f"PPO policy loss not finite: {pg_loss}"

    # Confirm the trained policy still acts deterministically on the env.
    obs, _ = env.reset(seed=0)
    raw_action, _ = model.predict(obs, deterministic=True)
    action = int(np.asarray(raw_action).reshape(-1)[0])
    assert 0 <= action < int(getattr(env.action_space, "n", 1))


def test_ppo_smoke_with_sweep_reward() -> None:
    """PPO smoke test with SweepConfirmationReward enabled.

    Verifies that TradingEnv builds with use_sweep_reward=True and that
    PPO can learn on the sweep-reward path without crashing.
    """
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")

    from stable_baselines3 import PPO

    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_synthetic_bars()
    features = _make_synthetic_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
        max_episode_steps=64,
        use_sweep_reward=True,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
    )

    model.learn(total_timesteps=256, progress_bar=False)

    recorded = model.logger.name_to_value
    assert "train/policy_gradient_loss" in recorded
    pg_loss = float(recorded["train/policy_gradient_loss"])
    assert np.isfinite(pg_loss)

    obs, _ = env.reset(seed=0)
    raw_action, _ = model.predict(obs, deterministic=True)
    action = int(np.asarray(raw_action).reshape(-1)[0])
    assert 0 <= action < int(getattr(env.action_space, "n", 1))


def test_default_config_exposes_max_episode_steps() -> None:
    """The config knob the env needs must be present in default.yaml."""
    cfg = OmegaConf.load("quant_rl/config/default.yaml")
    assert int(cfg.env.max_episode_steps) > 0
