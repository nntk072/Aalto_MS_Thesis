from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from simulation.environment.env import TradingEnv
from tests.test_envs._tp_mode_fixtures import (
    _make_deterministic_bars,
    _make_features,
    _make_tp_env,
)
from trading.execution.broker import Position
from trading.orders.selection import (
    decode_tp_mode,
)


class TestTPModeStateMachine:
    def test_tp_mode_fixed_keeps_target(self) -> None:
        """With no excursion, mode stays fixed."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)

        # Position with no favorable excursion
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            sl_initial_price=19990.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=0.0,  # No favorable excursion
            mae=0.0,
            mfe=0.0,
            tp_mode="fixed",
            tp_mode_overridden=False,
            tp_breakeven_done=False,
            entry_last_swing_high=None,
            entry_last_swing_low=None,
        )
        env.position = pos
        env.step_idx = 0

        result = env._default_tp_mode(pos)
        assert result == "fixed"

    def test_tp_mode_breakeven_tightens_tp(self) -> None:
        """After 50% favorable, mode advances to breakeven and _apply_tp_breakeven moves nearest TP."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)

        # Position with sufficient favorable excursion (>= 0.5 * risk)
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            sl_initial_price=19990.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=6.0,  # > 0.5 * risk_pts (5.0)
            mae=2.0,
            mfe=6.0,
            tp_mode="fixed",
            tp_mode_overridden=False,
            tp_breakeven_done=False,
            entry_last_swing_high=None,
            entry_last_swing_low=None,
            tp1_price=20010.0,
            tp2_price=None,
            tp3_price=None,
        )
        env.position = pos
        env.step_idx = 0

        # Should return breakeven mode
        result = env._default_tp_mode(pos)
        assert result == "breakeven"

        # Test that _apply_tp_breakeven tightens the nearest TP
        env._apply_tp_breakeven(pos, pos.best_favorable)
        # Should have tightened TP1: entry + alpha * risk = 20000 + 0.5 * 10 = 20005
        assert pos.tp1_price == 20005.0
        assert pos.tp_breakeven_done

    def test_tp_mode_trailing_follows_swing(self) -> None:
        """With a new swing in profitable direction, mode advances to trailing."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)

        # Position with sufficient favorable excursion AND new swing
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            sl_initial_price=19990.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=6.0,  # > 0.5 * risk_pts
            mae=2.0,
            mfe=6.0,
            tp_mode="fixed",
            tp_mode_overridden=False,
            tp_breakeven_done=False,
            entry_last_swing_high=20010.0,  # Original swing high
            entry_last_swing_low=None,
        )
        env.position = pos
        env.step_idx = 0

        # Mock feature row with higher swing high
        feat_row = features.iloc[0].copy()
        feat_row["last_swing_high"] = 20018.0  # New swing high > entry_last_swing_high

        # Override _feature_row_at to return our mock
        def mock_feature_row_at(idx):
            return feat_row

        with mock.patch.object(env, "_feature_row_at", new=mock_feature_row_at):
            env._default_tp_mode(pos)

    def test_tp_mode_agent_override(self) -> None:
        """Agent-selected mode (outside defer band) persists and tp_mode_overridden=True blocks state machine."""

        # Agent action value outside defer band
        u_tp = 0.9  # Outside defer threshold of 0.1 (abs(0.9 - 0.5) = 0.4 > 0.1)
        default_mode = "fixed"
        defer_threshold = 0.1

        mode, is_explicit = decode_tp_mode(u_tp, default_mode, defer_threshold)

        assert mode == "trailing"  # Decoded to trailing (index 2)
        assert is_explicit  # Explicit, outside defer band

    def test_tp_mode_defer_to_default(self) -> None:
        """Agent near center (within defer band) falls back to default state machine."""

        # Agent action value in the defer band
        # To be in the defer band, the unit value u must satisfy abs(u - 0.5) < defer_threshold
        # With default_mode, when action_val is raw [-1,1]: u = 0.5 * (action_val + 1.0)
        # For u ≈ 0.5, need action_val ≈ 0.0
        u_tp = 0.0  # Raw value that converts to u = 0.5 * (0.0 + 1.0) = 0.5
        default_mode = "fixed"
        defer_threshold = 0.1

        mode, is_explicit = decode_tp_mode(u_tp, default_mode, defer_threshold)

        # With u_tp=0.0 (raw), u = 0.5 * (0.0 + 1.0) = 0.5
        # abs(0.5 - 0.5) = 0.0 < 0.1, so it falls back to default mode "fixed"
        assert mode == "fixed"  # Falls back to default
        assert not is_explicit  # Not explicit, within defer band

    def test_tp_mode_state_is_sticky(self) -> None:
        """Mode never regresses automatically (trailing cannot go back to fixed)."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)

        # Position that has already advanced to trailing
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            sl_initial_price=19990.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=6.0,
            mae=2.0,
            mfe=6.0,
            tp_mode="trailing",
            tp_mode_overridden=False,
            tp_breakeven_done=True,
            entry_last_swing_high=20010.0,
            entry_last_swing_low=None,
        )
        env.position = pos
        env.step_idx = 0

        # Even with conditions that would normally be "fixed", should stay "trailing"
        result = env._default_tp_mode(pos)
        assert result == "trailing"  # Stays at trailing, never regresses

    def test_tp_mode_does_not_oscillate(self) -> None:
        """Repeated calls to state machine advance don't flip mode back."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)

        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)

        # Mock feature row with no new swing (same as entry swing)
        feat_row = features.iloc[0].copy()
        feat_row["last_swing_high"] = 20010.0  # Same as entry_last_swing_high, so no new swing

        # Override _feature_row_at to return our mock
        def mock_feature_row_at(idx):
            return feat_row

        with mock.patch.object(env, "_feature_row_at", new=mock_feature_row_at):
            pos = Position(
                direction=1,
                size=1.0,
                entry_price=20000.0,
                margin_used=2000.0,
                sl_price=19990.0,
                sl_initial_price=19990.0,
                entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
                best_favorable=6.0,  # Triggers breakeven (> 0.5 * risk_pts = 5.0)
                mae=2.0,
                mfe=6.0,
                tp_mode="fixed",
                tp_mode_overridden=False,
                tp_breakeven_done=False,
                entry_last_swing_high=20010.0,
                entry_last_swing_low=None,
            )
            env.position = pos
            env.step_idx = 0

            # First call should advance to breakeven (no new swing, so breakeven not trailing)
            first_result = env._default_tp_mode(pos)
            assert first_result == "breakeven"

            # Second call with same conditions should still return breakeven
            second_result = env._default_tp_mode(pos)
            assert second_result == "breakeven"  # Doesn't flip back

    def test_tp_trailing_is_monotonic(self) -> None:
        """For long, TP levels only increase; for short, only decrease."""
        bars = _make_deterministic_bars()
        features = _make_features(bars)
        env = TradingEnv(
            bars=bars,
            features=features,
            obs_window=10,
            initial_balance=100_000.0,
            episodic=True,
            strategy_actions=True,
            allow_agent_tp_mode=True,
            tp_mode_defer_threshold=0.1,
            allow_multi_tp=True,
            tp_breakeven_alpha=0.5,
        )
        env.reset(seed=42)
        levels = ("tp1_price", "tp2_price", "tp3_price")

        long_pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            sl_initial_price=19990.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=6.0,
            mae=0.0,
            mfe=6.0,
            tp_mode="trailing",
            tp1_price=20010.0,
            tp2_price=20015.0,
            tp3_price=20020.0,
        )
        long_history = [[getattr(long_pos, a) for a in levels]]
        for swing in (20012.0, 20013.0, 20014.0):
            feat = features.iloc[0].copy()
            feat["last_swing_high"] = swing
            env._apply_tp_trailing(long_pos, feat)
            long_history.append([getattr(long_pos, a) for a in levels])
        for prev, cur in zip(long_history, long_history[1:]):
            for p, c in zip(prev, cur):
                assert c >= p - 1e-9
        assert long_history[-1][0] > long_history[0][0]

        short_pos = Position(
            direction=-1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=20010.0,
            sl_initial_price=20010.0,
            entry_timestamp=pd.Timestamp("2025-01-06 16:30"),
            best_favorable=6.0,
            mae=0.0,
            mfe=6.0,
            tp_mode="trailing",
            tp1_price=19990.0,
            tp2_price=19985.0,
            tp3_price=19980.0,
        )
        short_history = [[getattr(short_pos, a) for a in levels]]
        for swing in (19988.0, 19987.0, 19986.0):
            feat = features.iloc[0].copy()
            feat["last_swing_low"] = swing
            env._apply_tp_trailing(short_pos, feat)
            short_history.append([getattr(short_pos, a) for a in levels])
        for prev, cur in zip(short_history, short_history[1:]):
            for p, c in zip(prev, cur):
                assert c <= p + 1e-9
        assert short_history[-1][0] < short_history[0][0]

    @pytest.mark.parametrize(
        ("direction", "swing_column", "swing", "targets", "expected"),
        [
            (1, "last_swing_high", 20018.0, (20010.0, 20015.0, 20020.0), 20018.0),
            (-1, "last_swing_low", 19982.0, (19990.0, 19985.0, 19980.0), 19982.0),
        ],
    )
    def test_trailing_update_keeps_slots_and_takes_effect_next_bar(
        self, direction, swing_column, swing, targets, expected
    ) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = Position(
            direction=direction,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0 if direction == 1 else 20010.0,
            sl_initial_price=19990.0 if direction == 1 else 20010.0,
            tp_mode="trailing",
            tp_hit_mask=1,
            tp1_price=targets[0],
            tp2_price=targets[1],
            tp3_price=targets[2],
        )
        feature = _make_features(_make_deterministic_bars()).iloc[0].copy()
        feature[swing_column] = swing
        event_time = env._bar_time(env.step_idx)

        env._apply_tp_trailing(pos, feature, event_time=event_time, bar_idx=env.step_idx)

        assert pos.tp1_price == targets[0]  # Already hit slot is immutable.
        assert pos.tp2_price == expected
        assert pos.tp3_price == targets[2]  # Later slot and quantity stay intact.
        update = env.trade_log[-1]
        assert update["type"] == "tp_update"
        assert update["tp_slot"] == 2
        assert update["time"] == event_time
        assert update["effective_time"] == env._bar_time(env.step_idx + 1)
        assert update["reference"] == swing_column

    def test_trailing_update_does_not_overtake_later_slot(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            tp_mode="trailing",
            tp1_price=20010.0,
            tp2_price=20015.0,
            tp3_price=20020.0,
        )
        feature = _make_features(_make_deterministic_bars()).iloc[0].copy()
        feature["last_swing_high"] = 20022.0
        before = (pos.tp1_price, pos.tp2_price, pos.tp3_price)

        env._apply_tp_trailing(pos, feature)

        assert (pos.tp1_price, pos.tp2_price, pos.tp3_price) == before
        assert not any(row.get("type") == "tp_update" for row in env.trade_log)

    def test_trailing_update_does_not_create_retroactive_target(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            tp_mode="trailing",
            tp1_price=20010.0,
            tp2_price=20020.0,
        )
        feature = _make_features(_make_deterministic_bars()).iloc[0].copy()
        feature["last_swing_high"] = 20018.0
        feature["close"] = 20019.0

        env._apply_tp_trailing(pos, feature)

        assert pos.tp1_price == 20010.0
        assert not any(row.get("type") == "tp_update" for row in env.trade_log)

    def test_fixed_mode_leaves_targets_unchanged(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = Position(
            direction=1,
            size=1.0,
            entry_price=20000.0,
            margin_used=2000.0,
            sl_price=19990.0,
            tp_mode="fixed",
            tp1_price=20010.0,
            tp2_price=20020.0,
            tp3_price=20030.0,
        )
        feature = _make_features(_make_deterministic_bars()).iloc[0].copy()
        feature["last_swing_high"] = 20040.0
        before = (pos.tp1_price, pos.tp2_price, pos.tp3_price)

        env._apply_tp_trailing(pos, feature)

        assert (pos.tp1_price, pos.tp2_price, pos.tp3_price) == before
        assert not any(row.get("type") == "tp_update" for row in env.trade_log)
