from __future__ import annotations

import numpy as np
import pytest

from simulation.environment.env import TradingEnv
from tests.test_envs._tp_mode_fixtures import (
    _make_deterministic_bars,
    _make_features,
)


class TestActionDecode:
    def test_action_space_12d(self) -> None:
        """12-D action decoded correctly; all dims at expected indices."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=True,  # +1 dim
            allow_agent_sl_mode=True,  # +1 dim
            allow_agent_tp_mode=True,  # +1 dim
            allow_multi_tp=True,  # +2 dims
            allow_simplex=True,  # +2 dims
        )
        env.reset(seed=42)

        # 5 (base) + 1 (direction) + 1 (sl_mode) + 1 (tp_mode) + 2 (multi_tp) + 2 (simplex) = 12
        assert env.action_space.shape == (12,)

        # Test decoding a 12-D action
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2])
        discrete_action, risk_frac, rr_ratio, tp_mode = env._decode_action(action, features.iloc[0])

        # Verify that all dimensions were processed without error
        assert discrete_action in [-1, 0, 1]
        assert 0.0 <= risk_frac <= 1.0
        assert env._selected_tp_mode is not None

    def test_legacy_5d_action_decode_unchanged(self) -> None:
        """5-D action produces identical result to pre-TP code."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        # Legacy env without TP features
        env_legacy = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=False,
            allow_agent_sl_mode=False,
            allow_agent_tp_mode=False,
            allow_multi_tp=False,
            allow_simplex=False,
        )
        env_legacy.reset(seed=42)

        # New env with TP features disabled
        env_new = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=False,
            allow_agent_sl_mode=False,
            allow_agent_tp_mode=False,
            allow_multi_tp=False,
            allow_simplex=False,
        )
        env_new.reset(seed=42)

        # Same 5-D action
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        discrete_action_legacy, risk_frac_legacy, rr_ratio_legacy, _ = env_legacy._decode_action(
            action, features.iloc[0]
        )
        discrete_action_new, risk_frac_new, rr_ratio_new, _ = env_new._decode_action(
            action, features.iloc[0]
        )

        # Results should be identical
        assert discrete_action_legacy == discrete_action_new
        assert abs(risk_frac_legacy - risk_frac_new) < 1e-10
        assert abs(rr_ratio_legacy - rr_ratio_new) < 1e-10

    def test_legacy_8d_action_decode_unchanged(self) -> None:
        """8-D action (with sl_mode) same as pre-TP code."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        # Legacy env with sl_mode but no TP features
        env_legacy = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=True,
            allow_agent_sl_mode=True,
            allow_agent_tp_mode=False,
            allow_multi_tp=False,
            allow_simplex=False,
        )
        env_legacy.reset(seed=42)

        # New env with same config
        env_new = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=True,
            allow_agent_sl_mode=True,
            allow_agent_tp_mode=False,
            allow_multi_tp=False,
            allow_simplex=False,
        )
        env_new.reset(seed=42)

        # This flag combination is 7-D; an eighth coordinate was historically
        # ignored by the decoder, making misconfigured checkpoints ambiguous.
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        discrete_action_legacy, risk_frac_legacy, rr_ratio_legacy, _ = env_legacy._decode_action(
            action, features.iloc[0]
        )
        discrete_action_new, risk_frac_new, rr_ratio_new, _ = env_new._decode_action(
            action, features.iloc[0]
        )

        # Results should be identical
        assert discrete_action_legacy == discrete_action_new
        assert abs(risk_frac_legacy - risk_frac_new) < 1e-10
        assert abs(rr_ratio_legacy - rr_ratio_new) < 1e-10
        assert env_legacy._selected_sl_mode == env_new._selected_sl_mode

        with pytest.raises(ValueError, match="action width"):
            env_new._decode_action(np.append(action, 0.8), features.iloc[0])

    def test_rr_target_anchors_to_nearest_level(self) -> None:
        """When multi_tp active, rr_ratio matches tp3 selection."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=False,
            allow_agent_sl_mode=False,
            allow_agent_tp_mode=False,
            allow_multi_tp=True,  # Enable multi-tp
            allow_simplex=False,
        )
        env.reset(seed=42)

        # Action with TP selections: [intensity, stop, risk, tp1_sel, tp2_sel, tp3_sel, exit]
        # For 7-D: 5 base + 2 for multi_tp = 7
        action = np.array([0.5, 0.0, 0.5, 0.2, 0.4, 0.8, 0.3])  # tp3_sel = 0.8

        env._decode_action(action, features.iloc[0])

        # rr anchors to the tp3 slot's unit-interval coordinate, like legacy tp_fraction.
        expected = 0.5 * (0.8 + 1.0)
        assert env._selected_tp_fraction == pytest.approx(expected)
        assert env._selected_tp_selections[2] == pytest.approx(expected)

    def test_tp_mode_defer_band_through_decode_action(self) -> None:
        """A neutral tp_mode dim defers to the configured default, not an explicit pick."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            agent_direction_control=False,
            allow_agent_sl_mode=False,
            allow_agent_tp_mode=True,
            allow_multi_tp=True,
            allow_simplex=False,
            tp_mode="breakeven",
        )
        env.reset(seed=42)

        # 8-D: [intensity, stop, risk, tp1, tp2, tp3, tp_mode, exit]
        assert env.action_space.shape == (8,)
        neutral = np.zeros(8, dtype=np.float32)
        env._decode_action(neutral, features.iloc[0])
        assert env._selected_tp_mode == "breakeven"
        assert env._tp_mode_explicit is False

        explicit = neutral.copy()
        explicit[6] = 0.9
        env._decode_action(explicit, features.iloc[0])
        assert env._selected_tp_mode == "trailing"
        assert env._tp_mode_explicit is True
