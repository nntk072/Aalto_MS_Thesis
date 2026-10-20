from __future__ import annotations

from typing import Any

import numpy as np

from simulation.contracts import SimulationTransition
from simulation.environment.env import TradingEnv
from tests.test_envs.test_trading_env_golden_traces import (
    _make_deterministic_bars,
    _make_features,
)


def _assert_same(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, tuple):
        assert isinstance(right, tuple) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_same(left_item, right_item)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (float, np.floating)) and np.isnan(left):
        assert isinstance(right, (float, np.floating)) and np.isnan(right)
    else:
        assert left == right


def test_native_transition_matches_gymnasium_step() -> None:
    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    native_env = TradingEnv(bars, features, obs_window=10)
    gym_env = TradingEnv(bars, features, obs_window=10)
    _assert_same(native_env.reset(seed=41), gym_env.reset(seed=41))

    native = native_env.transition(0)
    gym_result = gym_env.step(0)

    assert isinstance(native, SimulationTransition)
    _assert_same(native.as_gymnasium(), gym_result)
