"""Shared fixtures for the focused TP-mode test modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from simulation.environment.env import TradingEnv


def _make_deterministic_bars(n: int = 60) -> pd.DataFrame:
    """Create deterministic bars for testing."""
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(np.random.default_rng(42).normal(0, 1.0, n))
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0.1, 0.5, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "close": close,
            "tickvol": rng.integers(10, 100, n),
            "volume": rng.integers(1000, 5000, n),
            "vol": 0,
            "spread": 0.6,
            "gap_flag": False,
            "session_id": (pd.Series(idx).dt.floor("D").astype(np.int64) // 86400000000000).values,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Create minimal feature matrix with required columns."""
    n = len(bars)
    idx = bars.index
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "london_high": 20010.0 + rng.uniform(0, 5, n),
            "london_low": 19990.0 + rng.uniform(0, 5, n),
            "asian_high": 20005.0 + rng.uniform(0, 5, n),
            "asian_low": 19995.0 + rng.uniform(0, 5, n),
            "volume_spike": rng.uniform(0.5, 2.0, n),
            "last_swing_high": 20015.0 + rng.uniform(0, 3, n),
            "last_swing_low": 19985.0 + rng.uniform(0, 3, n),
            "atr_5": rng.uniform(10, 20, n),
            "ema_21": 20000.0 + rng.uniform(-10, 10, n),
        },
        index=idx,
    )


def _make_tp_env(**kwargs) -> TradingEnv:
    """Create TradingEnv with TP features enabled."""
    bars = _make_deterministic_bars()
    features = _make_features(bars)

    default_kwargs: dict[str, Any] = {
        "bars": bars,
        "features": features,
        "obs_window": 10,
        "initial_balance": 100_000.0,
        "episodic": True,
        "use_sweep_reward": False,
        "strategy_actions": True,
        "allow_agent_tp_mode": True,
        "tp_mode_defer_threshold": 0.1,
        "allow_multi_tp": True,
        "tp_breakeven_alpha": 0.5,
        "allow_simplex": True,
        "allow_agent_sl_mode": True,
        "sl_mode_defer_threshold": 0.1,
        "agent_direction_control": False,
    }
    default_kwargs.update(kwargs)

    env = TradingEnv(**default_kwargs)
    env.reset(seed=42)
    return env
