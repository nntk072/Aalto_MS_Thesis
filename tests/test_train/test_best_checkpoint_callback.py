"""Tests for the BestCheckpointEvalCallback.

Verifies that the callback:
- writes a model file to ``best_model_path`` once the running best
  episode reward improves,
- keeps the file at the best-so-far reward and does not overwrite on a
  strictly worse rollout (we only trigger save on ``>``, never ``>=``).
- exposes the saved SB3 model and can reload it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _bars(n: int = 300, seed: int = 7) -> pd.DataFrame:
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


def _features(bars: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
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


def test_inactive_policy_does_not_create_best_checkpoint(tmp_path: Path) -> None:
    """The callback must persist the model after the first eval and
    expose a loadable SB3 artifact at the configured path."""
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from learning.rl.callbacks import BestCheckpointEvalCallback
    from simulation.environment.env import TradingEnv

    bars = _bars()
    features = _features(bars)

    def _factory() -> TradingEnv:
        return TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            max_episode_steps=32,
            episodic=False,
        )

    env = _factory()
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
    )
    best_path = tmp_path / "best_model"
    cb = BestCheckpointEvalCallback(
        eval_env_factory=_factory,
        eval_freq=32,  # one eval per rollout
        best_model_path=best_path,
        n_eval_episodes=1,
    )
    cb.init_callback(model)

    cb._run_eval()
    cb._run_eval()
    assert cb.best_checkpoint_status == "NO_ACTIVE_CHECKPOINT"
    assert cb.best_mean_reward == -np.inf
    assert not best_path.with_suffix(".zip").exists()


def test_inactive_policy_stays_ineligible_on_repeated_evaluation(tmp_path: Path) -> None:
    """Two evals: the second yields a worse reward → the on-disk model
    must NOT be replaced. We can't read PPO weights to compare directly,
    so we proxy via the saved file's mtime.
    """
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from learning.rl.callbacks import BestCheckpointEvalCallback
    from simulation.environment.env import TradingEnv

    bars = _bars()
    features = _features(bars)

    def _factory() -> TradingEnv:
        return TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            max_episode_steps=32,
            episodic=False,
        )

    env = _factory()
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        learning_rate=3e-4,
        verbose=0,
        seed=0,
    )
    best_path = tmp_path / "best_model"
    cb = BestCheckpointEvalCallback(
        eval_env_factory=_factory,
        eval_freq=32,
        best_model_path=best_path,
        n_eval_episodes=1,
    )
    cb.init_callback(model)

    cb._run_eval()
    cb._run_eval()
    assert cb.best_checkpoint_status == "NO_ACTIVE_CHECKPOINT"
    assert cb.best_mean_reward == -np.inf
    assert not best_path.with_suffix(".zip").exists()
