"""Tests for the account-vector normalization in TradingEnv.

The ``seq`` features are z-scored at build time so they sit at ~O(1).
The legacy ``account`` vector embedded raw equity (~1e5) and open PnL,
which dominated the policy's MLP heads and made the time-series signal
effectively invisible. ``normalize_account=True`` (default) rescales the
account vector to the same order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.envs.trading_env import TradingEnv


def _make_bars(n: int = 200, seed: int = 7) -> pd.DataFrame:
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


def _make_features(bars: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
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


def test_normalized_account_has_unit_magnitude() -> None:
    """When normalize_account=True, all account dims should be O(1)."""
    bars = _make_bars()
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
        normalize_account=True,
    )
    obs, _ = env.reset(seed=0)
    acc = obs["account"]
    assert np.all(np.abs(acc) < 10.0), (
        f"Normalized account should be O(1), got max abs = {np.abs(acc).max()}"
    )
    # At reset, equity == initial_balance → norm_equity == log(1) == 0.
    assert acc[0] == 0.0


def test_unnormalized_account_carries_raw_equity() -> None:
    """When normalize_account=False, equity is the raw USD value."""
    bars = _make_bars()
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        initial_balance=100_000.0,
        normalize_account=False,
    )
    obs, _ = env.reset(seed=0)
    acc = obs["account"]
    # Raw equity is 1e5, which would dominate the z-scored seq features.
    assert acc[0] == 100_000.0
    # The vector must still carry raw values (no silent normalization).
    assert acc.shape == (5,)


def test_normalization_default_is_on() -> None:
    """Default behavior must include normalization — it's a strict upgrade
    and the prior off-by-default would silently degrade the policy."""
    bars = _make_bars()
    features = _make_features(bars)
    env = TradingEnv(bars=bars, features=features, obs_window=10, initial_balance=100_000.0)
    obs, _ = env.reset(seed=0)
    # The default normalized equity is log(1)=0, not 1e5.
    assert obs["account"][0] == 0.0
