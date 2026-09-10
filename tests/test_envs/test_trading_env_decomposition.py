"""Characterization tests for TradingEnv._decode_action decomposition.

Verifies that the extracted _decode_action() method produces identical
results to the original inline action decoding logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.envs.trading_env import TradingEnv


def _make_simple_env(episodic: bool = True) -> TradingEnv:
    """Create a minimal TradingEnv for action decoding tests."""
    n = 100
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.cumsum(np.random.default_rng(0).normal(0, 2, n))
    bars = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tickvol": 50,
            "volume": 2000,
            "vol": 0,
            "spread": 0.6,
            "gap_flag": False,
            "session_id": 0,
        },
        index=idx,
    )
    bars.index.name = "datetime"
    features = pd.DataFrame(
        {
            "london_high": 20500.0,
            "london_low": 19500.0,
            "asian_high": 20300.0,
            "asian_low": 19700.0,
            "volume_spike": 1.0,
            "last_swing_high": close + 5,
            "last_swing_low": close - 5,
        },
        index=idx,
    )
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        episodic=episodic,
    )


class TestDecodeActionDiscrete:
    """Tests for discrete action decoding (default mode)."""

    def test_hold_action(self) -> None:
        """Action 0 = hold."""
        env = _make_simple_env()
        da, rf, rr, tp = env._decode_action(0)
        assert da == 0
        assert rf == env.risk_frac_range[0]
        assert rr == env.rr_ratio_range[0]

    def test_long_action_maps_to_direction_1(self) -> None:
        """Actions 1-9 = enter long."""
        env = _make_simple_env()
        for action in [1, 5, 9]:
            da, _, _, _ = env._decode_action(action)
            assert da == 1

    def test_short_action_maps_to_direction_neg1(self) -> None:
        """Actions 10-18 = enter short."""
        env = _make_simple_env()
        for action in [10, 14, 18]:
            da, _, _, _ = env._decode_action(action)
            assert da == -1

    def test_exit_action_maps_to_hold(self) -> None:
        """Action 19 = exit (mapped to hold in decoding)."""
        env = _make_simple_env()
        da, _, _, _ = env._decode_action(19)
        assert da == 0

    def test_risk_variant_mapping(self) -> None:
        """Actions 1-9 map to 3 risk levels × 3 rr levels."""
        env = _make_simple_env()
        r_lo, r_hi = env.risk_frac_range
        rr_lo, rr_hi = env.rr_ratio_range
        expected_risk = [r_lo, (r_lo + r_hi) / 2, r_hi]
        expected_rr = [rr_lo, (rr_lo + rr_hi) / 2, rr_hi]
        # Action 1: risk_variant=0, rr_variant=0
        da, rf, rr, _ = env._decode_action(1)
        assert da == 1
        assert rf == pytest.approx(expected_risk[0])
        assert rr == pytest.approx(expected_rr[0])
        # Action 5: risk_variant=1, rr_variant=1
        da, rf, rr, _ = env._decode_action(5)
        assert da == 1
        assert rf == pytest.approx(expected_risk[1])
        assert rr == pytest.approx(expected_rr[1])
        # Action 9: risk_variant=2, rr_variant=2
        da, rf, rr, _ = env._decode_action(9)
        assert da == 1
        assert rf == pytest.approx(expected_risk[2])
        assert rr == pytest.approx(expected_rr[2])


class TestDecodeActionContinuous:
    """Tests for continuous action decoding (SAC mode)."""

    def _make_continuous_env(self) -> TradingEnv:
        env = _make_simple_env()
        env.continuous_actions = True
        return env

    def test_positive_action_is_long(self) -> None:
        env = self._make_continuous_env()
        da, rf, _, _ = env._decode_action(np.array([0.5]))
        assert da == 1
        assert rf > 0

    def test_negative_action_is_short(self) -> None:
        env = self._make_continuous_env()
        da, rf, _, _ = env._decode_action(np.array([-0.5]))
        assert da == -1
        assert rf > 0

    def test_zero_action_is_hold(self) -> None:
        env = self._make_continuous_env()
        da, rf, _, _ = env._decode_action(np.array([0.0]))
        assert da == 0
        assert rf == 0.0


class TestDecodeActionStrategy:
    """Tests for strategy (4-D Box) action decoding."""

    def _make_strategy_env(self) -> TradingEnv:
        env = _make_simple_env()
        env.strategy_actions = True
        return env

    def test_strong_positive_direction(self) -> None:
        env = self._make_strategy_env()
        da, _, _, _ = env._decode_action(np.array([0.5, 0.5, 0.5, 0.5]))
        assert da == 1

    def test_strong_negative_direction(self) -> None:
        env = self._make_strategy_env()
        da, _, _, _ = env._decode_action(np.array([-0.5, 0.5, 0.5, 0.5]))
        assert da == -1

    def test_weak_direction_is_hold(self) -> None:
        env = self._make_strategy_env()
        da, _, _, _ = env._decode_action(np.array([0.1, 0.5, 0.5, 0.5]))
        assert da == 0

    def test_tp_mode_selection(self) -> None:
        env = self._make_strategy_env()
        _, _, _, tp = env._decode_action(np.array([0.5, 0.5, 0.5, 0.0]))
        assert tp == "rr"
        _, _, _, tp = env._decode_action(np.array([0.5, 0.5, 0.5, 1.0]))
        assert tp == "previous_day_high_low"
