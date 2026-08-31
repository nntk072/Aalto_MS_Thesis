"""Tests for the baseline strategy runner (``quant_rl.train.run_baselines``).

Specifically covers the new ``_make_indexed_policy`` and the real indicator
precomputation that the EMA / RSI baselines now use instead of the legacy
returns-heuristic placeholder. The contract under test is:

- ``_make_indexed_policy(actions, obs_window)`` returns a callable that
  emits ``actions.iloc[i]`` for the ``i``-th call, starting at
  ``obs_window`` (so the absolute bar index in ``run_backtest`` lines up
  with the precomputed action series).
- The policy advances its internal index exactly once per call, and
  past-the-end calls fall back to ``0`` (flat) instead of raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.train.run_baselines import _make_indexed_policy


def _make_bars(n: int = 60, trend: float = 0.5) -> pd.DataFrame:
    """Synthetic OHLCV bars with DatetimeIndex — shared with the runner."""
    rng = np.random.default_rng(42)
    close = 20_000.0 + np.cumsum(rng.normal(trend, 2.0, n))
    index = pd.date_range("2025-01-02 16:30", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 1.0, n),
            "high": close + rng.uniform(1.0, 3.0, n),
            "low": close - rng.uniform(1.0, 3.0, n),
            "close": close,
            "volume": rng.uniform(900, 1_100, n),
        },
        index=index,
    )


def test_make_indexed_policy_starts_at_obs_window() -> None:
    """First emitted action must be ``actions.iloc[obs_window]``."""
    actions = pd.Series([0, 0, 0, 1, 1, 0, -1, 0, 0], dtype=int)
    policy = _make_indexed_policy(actions, obs_window=3)
    dummy_obs = np.zeros((1, 1), dtype=np.float32)

    first = policy(dummy_obs)
    assert first == 1  # actions.iloc[3]


def test_make_indexed_policy_advances_one_per_call() -> None:
    """Successive calls must return successive actions from the series."""
    actions = pd.Series([0, 0, 0, 1, -1, 1, 0, -1, 0], dtype=int)
    policy = _make_indexed_policy(actions, obs_window=2)
    dummy_obs = np.zeros((1, 1), dtype=np.float32)

    # First call: actions.iloc[2] = 0 (warmup region).
    # Then the EMA crosses / RSI flips we encoded at indices 3..7.
    emitted = [policy(dummy_obs) for _ in range(6)]
    assert emitted == [0, 1, -1, 1, 0, -1]


def test_make_indexed_policy_past_end_returns_flat() -> None:
    """When the index walks past the end, the policy must yield 0 (flat)."""
    actions = pd.Series([1] * 5, dtype=int)
    policy = _make_indexed_policy(actions, obs_window=2)
    dummy_obs = np.zeros((1, 1), dtype=np.float32)

    for _ in range(3):
        assert policy(dummy_obs) == 1
    # Two more calls land past the end → must fall back to 0, never raise.
    assert policy(dummy_obs) == 0
    assert policy(dummy_obs) == 0


def test_make_indexed_policy_ignores_obs_argument() -> None:
    """The runner never inspects the obs; the closure owns the index."""
    actions = pd.Series([0, 0, 0, 0, 1, 0], dtype=int)
    policy = _make_indexed_policy(actions, obs_window=4)
    sentinel_obs = np.full((10, 5), 99.0, dtype=np.float32)

    assert policy(sentinel_obs) == 1  # still actions.iloc[4]
