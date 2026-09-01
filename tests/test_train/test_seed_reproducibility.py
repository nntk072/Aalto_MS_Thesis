"""Seed-reproducibility test for the PPO training stack.

A long, quiet source of debugging pain: SB3, PyTorch, and Gymnasium each
have independent RNGs. If a caller sets a seed on only some of them, PPO
rollouts will diverge between runs and the rest of the suite will catch
unrelated symptoms (different trade counts, different Sharpe). This test
pins the contract:

- The same seed on PPO + numpy + torch produces identical ``n_steps``
  rollout trade log columns (``type``, ``pnl``, ``equity``).
- The two policies also predict the same action on the same observation.

We deliberately use a tiny n_steps / n_epochs so the test is fast on CPU.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def _make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:

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


def _seed_everything(seed: int) -> None:
    """Seed all RNGs that PPO/SB3/PyTorch touch.

    SB3's PPO seeds its own ``self._generator`` but does *not* seed
    ``torch.manual_seed`` or ``numpy.random`` — those are the caller's
    responsibility. The standard contract is "seed python.random,
    numpy.random, and torch.manual_seed before building the model".
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_same_seed_yields_identical_ppo_rollout() -> None:
    """Two PPO models trained on the same env with the same seed must
    produce identical first-rollout action streams and identical
    predictions on a fresh observation.
    """
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_bars()
    features = _make_features(bars)

    def _build_and_roll() -> tuple[list[int], int, float]:
        _seed_everything(0)
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            max_episode_steps=64,
        )
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            learning_rate=3e-4,
            verbose=0,
            seed=0,
        )
        obs, _ = env.reset(seed=0)
        actions: list[int] = []
        for _ in range(64):
            raw, _ = model.predict(obs, deterministic=True)
            actions.append(int(np.asarray(raw).reshape(-1)[0]))
            obs, _, done, truncated, _ = env.step(actions[-1])
            if done or truncated:
                break
        return actions, actions[-1], float(obs["account"][0])

    # Two independent builds with the same seed.
    a1, last1, eq1 = _build_and_roll()
    a2, last2, eq2 = _build_and_roll()

    assert a1 == a2, f"Action streams differ under same seed: {a1[:5]} vs {a2[:5]}"
    assert last1 == last2
    assert eq1 == pytest.approx(eq2)


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:  # tiny helper
    """Build a minimal feature DataFrame matching the bars index."""

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
