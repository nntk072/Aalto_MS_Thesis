"""Tests for TP mode, multi-TP selection, simplex fractions, and ordering invariants."""

from __future__ import annotations

import math
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.broker import Position
from quant_rl.envs.feature_row import BarView
from quant_rl.envs.tp_decoders import (
    DISABLED,
    decode_tp_fractions,
    decode_tp_mode,
    decode_tp_selections,
    validate_tp_levels,
)
from quant_rl.envs.trading_env import TradingEnv


class TestSimplexDecoder:
    def test_simplex_decoder_sums_to_one(self) -> None:
        f1, f2, f3 = decode_tp_fractions(0.0, 0.0)
        assert pytest.approx(f1 + f2 + f3, rel=1e-6) == 1.0
        assert pytest.approx(f1, rel=1e-6) == 1.0 / 3.0
        assert pytest.approx(f2, rel=1e-6) == 1.0 / 3.0
        assert pytest.approx(f3, rel=1e-6) == 1.0 / 3.0

    def test_simplex_renormalize_active_levels(self) -> None:
        fracs = decode_tp_fractions(1.0, -1.0, active_slots=(True, True, True))
        assert len(fracs) == 3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_renormalize_when_tp1_disabled(self) -> None:
        # TP1 disabled, TP2 and TP3 active
        z1, z2 = 0.5, 1.0
        # raw unnormalized fractions:
        m = max(0.5, 1.0, 0.0)
        e1 = math.exp(0.5 - m)
        e2 = math.exp(1.0 - m)
        e3 = math.exp(-m)
        denom = e1 + e2 + e3
        raw_f2 = e2 / denom
        raw_f3 = e3 / denom
        expected_f2 = raw_f2 / (raw_f2 + raw_f3)
        expected_f3 = raw_f3 / (raw_f2 + raw_f3)

        fracs = decode_tp_fractions(z1, z2, active_slots=(False, True, True))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == expected_f2
        assert pytest.approx(fracs[1], rel=1e-6) == expected_f3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_renormalize_when_tp2_disabled(self) -> None:
        # TP2 disabled, TP1 and TP3 active
        z1, z2 = 0.5, 1.0
        m = max(0.5, 1.0, 0.0)
        e1 = math.exp(0.5 - m)
        e2 = math.exp(1.0 - m)
        e3 = math.exp(-m)
        denom = e1 + e2 + e3
        raw_f1 = e1 / denom
        raw_f3 = e3 / denom
        expected_f1 = raw_f1 / (raw_f1 + raw_f3)
        expected_f3 = raw_f3 / (raw_f1 + raw_f3)

        fracs = decode_tp_fractions(z1, z2, active_slots=(True, False, True))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == expected_f1
        assert pytest.approx(fracs[1], rel=1e-6) == expected_f3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0

    def test_simplex_decoder_handles_large_logits(self) -> None:
        fracs = decode_tp_fractions(1000.0, 2000.0, active_slots=(True, True, True))
        assert len(fracs) == 3
        assert pytest.approx(sum(fracs), rel=1e-6) == 1.0
        assert fracs[1] > fracs[0]
        assert fracs[1] > fracs[2]

        fracs_neg = decode_tp_fractions(-1000.0, -2000.0, active_slots=(True, True, True))
        assert len(fracs_neg) == 3
        assert pytest.approx(sum(fracs_neg), rel=1e-6) == 1.0


class TestTPSelection:
    def test_tp_selection_disabled_middle_slot(self) -> None:
        # TP1 active, TP2 disabled, TP3 active
        s1, s2, s3 = decode_tp_selections(0, DISABLED, 2, menu_size=5)
        assert s1 == 0
        assert s2 is None
        assert s3 == 2

    def test_tp_selection_duplicate_resolution(self) -> None:
        # Duplicate 1, 1, 2 -> disable nearer slot (TP1)
        s1, s2, s3 = decode_tp_selections(1, 1, 2, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 2

        # Duplicate 1, 2, 2 -> disable nearer slot (TP2)
        s1, s2, s3 = decode_tp_selections(1, 2, 2, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # Triplicate 1, 1, 1 -> only TP3 remains
        s1, s2, s3 = decode_tp_selections(1, 1, 1, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 1

    def test_tp_selection_invalid_tp3_repair(self) -> None:
        # TP3 is DISABLED, but TP1=1 and TP2=2 are valid.
        # Since TP3 is required, promote furthest candidate (2) to TP3.
        s1, s2, s3 = decode_tp_selections(1, 2, DISABLED, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # TP3 out of bounds clamped to menu_size - 1
        s1, s2, s3 = decode_tp_selections(0, 1, 999, menu_size=5)
        assert s1 == 0
        assert s2 == 1
        assert s3 == 4

    def test_tp_selection_boundary_values(self) -> None:
        # menu_size = 1 -> only index 0 is valid
        s1, s2, s3 = decode_tp_selections(0, 0, 0, menu_size=1)
        assert s1 is None
        assert s2 is None
        assert s3 == 0

        # Boundary continuous values: -1.0 is disabled, 0.0 is index 0, 1.0 is menu_size-1
        s1, s2, s3 = decode_tp_selections(-1.0, 0.0, 1.0, menu_size=4, continuous=True)
        assert s1 is None
        assert s2 == 0
        assert s3 == 3

    def test_tp_selection_all_disabled_falls_back_to_nearest(self) -> None:
        # All DISABLED -> activates nearest valid target as sole TP (TP3)
        s1, s2, s3 = decode_tp_selections(DISABLED, DISABLED, DISABLED, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 0

    def test_tp_selection_preserves_maximum_valid_subset(self) -> None:
        # raw={1, 3, 2}: s2(3) >= s3(2) -> disable s2 -> {1, None, 2}
        s1, s2, s3 = decode_tp_selections(1, 3, 2, menu_size=5)
        assert s1 == 1
        assert s2 is None
        assert s3 == 2

        # raw={2, 1, 3}: s1(2) >= s2(1) -> disable s1 -> {None, 1, 3}
        s1, s2, s3 = decode_tp_selections(2, 1, 3, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 3

        # raw={3, 2, 1}: descending -> {None, None, 1}
        s1, s2, s3 = decode_tp_selections(3, 2, 1, menu_size=5)
        assert s1 is None
        assert s2 is None
        assert s3 == 1

    def test_tp_fraction_mapping_when_tp1_disabled(self) -> None:
        s1, s2, s3 = decode_tp_selections(DISABLED, 1, 3, menu_size=5)
        assert s1 is None
        assert s2 == 1
        assert s3 == 3
        fracs = decode_tp_fractions(0.0, 0.0, active_slots=(s1, s2, s3))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0] + fracs[1], rel=1e-6) == 1.0
        # When z1=z2=0, f1=f2=f3=1/3. Normalizing f2 and f3 gives 0.5 each
        assert pytest.approx(fracs[0], rel=1e-6) == 0.5
        assert pytest.approx(fracs[1], rel=1e-6) == 0.5

    def test_tp_fraction_mapping_when_tp2_disabled(self) -> None:
        s1, s2, s3 = decode_tp_selections(0, DISABLED, 3, menu_size=5)
        assert s1 == 0
        assert s2 is None
        assert s3 == 3
        fracs = decode_tp_fractions(0.0, 0.0, active_slots=(s1, s2, s3))
        assert len(fracs) == 2
        assert pytest.approx(fracs[0], rel=1e-6) == 0.5
        assert pytest.approx(fracs[1], rel=1e-6) == 0.5

    def test_multi_tp_preserves_fraction_slot_mapping(self) -> None:
        # Verify slot identity: fraction for TP1 always couples to TP1,
        # fraction for TP3 always couples to TP3.
        # If z1 has high logit (f1 high) and z2 low:
        z1, z2 = 5.0, -5.0
        fracs_all = decode_tp_fractions(z1, z2, active_slots=(0, 1, 2))
        f1_raw, f2_raw, f3_raw = fracs_all
        assert f1_raw > 0.9  # TP1 gets the large fraction

        # Now suppose TP2 is disabled:
        fracs_sub = decode_tp_fractions(z1, z2, active_slots=(0, None, 2))
        f1_sub, f3_sub = fracs_sub
        # TP1 must still retain the dominant share, not remapped to TP3
        assert f1_sub > 0.9
        assert f3_sub < 0.1


class TestTPValidation:
    def test_validate_tp_levels_long(self) -> None:
        entry = 100.0
        sl = 98.0
        # Valid: SL < Entry <= TP1 < TP2 < TP3
        assert validate_tp_levels(1, entry, sl, [102.0, 104.0, 106.0])
        # Valid with middle disabled: SL < Entry <= TP1 < TP3
        assert validate_tp_levels(1, entry, sl, [102.0, None, 106.0])
        # Invalid: TP below entry
        assert not validate_tp_levels(1, entry, sl, [99.0, 104.0, 106.0])
        # Invalid: non-monotonic
        assert not validate_tp_levels(1, entry, sl, [105.0, 103.0, 106.0])
        # Invalid: TP below SL
        assert not validate_tp_levels(1, entry, sl, [97.0, None, 106.0])

    def test_validate_tp_levels_short(self) -> None:
        entry = 100.0
        sl = 102.0
        # Valid: TP3 < TP2 < TP1 <= Entry < SL
        # In slot order [TP1, TP2, TP3]: [98.0, 96.0, 94.0]
        assert validate_tp_levels(-1, entry, sl, [98.0, 96.0, 94.0])
        # Valid with middle disabled:
        assert validate_tp_levels(-1, entry, sl, [98.0, None, 94.0])
        # Invalid: TP above entry
        assert not validate_tp_levels(-1, entry, sl, [101.0, 96.0, 94.0])
        # Invalid: non-monotonic
        assert not validate_tp_levels(-1, entry, sl, [95.0, 97.0, 94.0])
        # Invalid: TP above SL
        assert not validate_tp_levels(-1, entry, sl, [103.0, None, 94.0])


# Helper functions for TradingEnv tests


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


# Phase 3 — State machine tests
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


class TestTPExecution:
    def test_env_rejects_entry_filling_on_final_ny_bar(self) -> None:
        bars = _make_deterministic_bars().copy()
        bars["session"] = ["ny", "ny", "ny", *(["asian"] * (len(bars) - 3))]
        env = _make_tp_env(
            bars=bars,
            features=_make_features(bars),
            block_overnight=True,
        )
        env.reset(seed=42)
        env.step_idx = 1
        env._ny_pos = 1

        with (
            mock.patch.object(env, "_decode_action", return_value=(1, 0.01, 2.0, "fixed")),
            mock.patch.object(env, "_check_entry_gate", return_value=True),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(1, 0.01)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is None
        assert not any(row.get("type") == "open" for row in env.trade_log)
        assert env._entry_diag.get("rejected_overnight", 0) == 1

    def test_bar_close_entry_does_not_replay_earlier_entry_bar_extrema(self) -> None:
        env = _make_tp_env(block_overnight=False)
        env.reset(seed=42)
        env.step_idx = 20
        bar_time = env._bar_time(env.step_idx)
        bar = env._bar_at(env.step_idx)
        position = env.broker.open_position(
            env.account,
            env._bar_quote(bar),
            1.0,
            1,
        )
        assert position is not None
        position.entry_timestamp = bar_time + pd.Timedelta(minutes=1)
        position.sl_price = float(bar.low) + 0.01
        position.sl_initial_price = position.sl_price
        env.position = position

        with (
            mock.patch.object(env, "_progress_market", return_value=(*env._bar_quote(bar), None)),
            mock.patch.object(env, "_check_guardrails", return_value=(None, False, False, False)),
            mock.patch.object(env, "_apply_peak_trailing_dd", return_value=False),
            mock.patch.object(env, "_decode_action", return_value=(0, 0.0, 2.0, "fixed")),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(0, 0.0)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is position
        assert not any(row.get("type") == "stop_close" for row in env.trade_log)

    def test_env_rejects_entry_without_a_future_fill_bar(self) -> None:
        bars = _make_deterministic_bars()
        env = _make_tp_env(
            bars=bars,
            features=_make_features(bars),
            block_overnight=False,
        )
        env.step_idx = int(env._ny_indices[-1])
        env._ny_pos = len(env._ny_indices) - 1
        with (
            mock.patch.object(env, "_decode_action", return_value=(1, 0.01, 2.0, "fixed")),
            mock.patch.object(env, "_check_entry_gate", return_value=True),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(1, 0.01)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is None
        assert not any(row.get("type") == "open" for row in env.trade_log)
        assert env._entry_diag.get("rejected_terminal", 0) == 1

    def test_strict_eligibility_rejects_missing_entry_gate_inputs(self) -> None:
        env = _make_tp_env(strict_research_eligibility=True)
        with pytest.raises(ValueError, match="entry.gate|entry gate|Entry.gate"):
            env._check_entry_gate(100.0, 1, pd.Series({"volume_spike": 2.0}))

    def test_strict_eligibility_rejects_missing_ny_session(self) -> None:
        bars = _make_deterministic_bars().copy()
        bars["session"] = "asian"
        with pytest.raises(ValueError, match="NY|ny|New York"):
            _make_tp_env(
                bars=bars,
                features=_make_features(bars),
                strict_research_eligibility=True,
            )

    def test_last_ny_bar_is_session_boundary(self) -> None:
        env = _make_tp_env(block_overnight=True)
        env.reset(seed=42)
        env._ny_pos = len(env._ny_indices) - 1
        env.step_idx = int(env._ny_indices[-1])
        assert env._is_ny_session_boundary()

    def test_session_boundary_closes_at_current_quote(self) -> None:
        env = _make_tp_env(block_overnight=True)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        env.position = pos
        bar = BarView(open=100.0, high=101.0, low=99.0, close=100.0, spread=60.0)
        later = BarView(open=105.0, high=106.0, low=104.0, close=105.0, spread=60.0)
        with (
            mock.patch.object(env, "_is_ny_session_boundary", return_value=True),
            mock.patch.object(env, "_bar_at", return_value=later),
        ):
            env._progress_market(
                bar,
                env._feature_row_at(env.step_idx),
                env._bar_time(env.step_idx),
                0,
                env.step_idx + 1,
            )
        assert env.position is None
        assert env.trade_log[-1]["type"] == "eod_close"
        assert env.trade_log[-1]["price"] == pytest.approx(100.0)

    def test_short_ema_exit_pays_ask_spread(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, -1)
        assert pos is not None
        pos.exit_mode = "ema_21"
        pos.stop_distance = 1.0
        pos.best_favorable = 2.0
        env.position = pos
        bar = BarView(open=100.0, high=100.5, low=98.0, close=100.0, spread=60.0)
        with mock.patch.object(env, "_ema21_at", return_value=99.0):
            env._maybe_ema_exit(bar, env.step_idx, env._bar_time(env.step_idx))
        assert env.position is None
        assert env.trade_log[-1]["reason"] == "ema_21_exit"
        assert env.trade_log[-1]["price"] == pytest.approx(100.6)

    def _enter_multi_tp(
        self,
        env: TradingEnv,
        *,
        size: float = 1.0,
        fractions: tuple[float, ...] = (0.3, 0.3, 0.4),
    ) -> Position:
        pos = env.broker.open_position(env.account, (100.0, 100.0), size, 1)
        assert pos is not None
        pos.sl_price = 99.0
        pos.sl_initial_price = 99.0
        pos.tp_initial_size = size
        pos.tp_lot_fractions = fractions
        pos.tp1_price = 101.0
        pos.tp2_price = 102.0
        pos.tp3_price = 103.0
        env.position = pos
        return pos

    @staticmethod
    def _tp_closes(env: TradingEnv) -> list[dict[str, Any]]:
        return [t for t in env.trade_log if t["type"] in ("tp_partial_close", "tp_close")]

    @staticmethod
    def _hit_bar(high: float) -> BarView:
        return BarView(open=high - 0.4, high=high, low=99.5, close=high - 0.2, spread=0.6)

    def test_partial_close_uses_initial_position_size(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)
        assert pos.tp_hit_mask == 1
        assert pos.tp1_price is None
        closes = self._tp_closes(env)
        assert len(closes) == 1
        assert closes[0]["size"] == pytest.approx(0.3)  # 0.3 * tp_initial_size

        env._check_sl_tp(
            BarView(open=101.0, high=102.4, low=100.8, close=102.0, spread=0.6),
            fill_bid=102.0,
            fill_ask=102.6,
        )
        # Still 0.3 of the entry size; 0.3 of the remaining 0.7 would be 0.21.
        assert pos.size == pytest.approx(0.4)
        closes = self._tp_closes(env)
        assert len(closes) == 2
        assert closes[1]["size"] == pytest.approx(0.3)

    def test_cooldown_starts_after_position_completion_not_partial_fill(self) -> None:
        env = _make_tp_env(entry_cooldown_bars=5)
        pos = self._enter_multi_tp(env)
        env._cooldown_until_bar = 0
        env.step_idx = 20

        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)
        assert env._cooldown_until_bar == 0

        env._check_sl_tp(
            BarView(open=100.0, high=100.5, low=98.5, close=99.0, spread=0.6),
            fill_bid=99.0,
            fill_ask=99.6,
        )
        assert env.position is None
        assert env._cooldown_until_bar == 25

    def test_partial_position_closes_at_session_end_as_one_lifecycle(self) -> None:
        env = _make_tp_env(block_overnight=True)
        pos = self._enter_multi_tp(env)
        pos.position_id = "session:partial"
        env.trade_log.append({"type": "open", "position_id": pos.position_id})
        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)

        current = BarView(open=101.0, high=101.5, low=100.5, close=101.2, spread=0.6)
        next_bar = BarView(open=105.0, high=105.5, low=104.5, close=105.2, spread=0.6)
        with (
            mock.patch.object(env, "_is_ny_session_boundary", return_value=True),
            mock.patch.object(env, "_bar_at", return_value=next_bar),
        ):
            env._progress_market(
                current,
                env._feature_row_at(env.step_idx),
                env._bar_time(env.step_idx),
                0,
                env.step_idx + 1,
            )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["size"] for row in fills] == pytest.approx([0.3, 0.7])
        assert [row["position_complete"] for row in fills] == [False, True]
        assert fills[-1]["position_net_pnl"] == pytest.approx(sum(row["net_pnl"] for row in fills))
        assert env.position is None

    def test_partial_close_never_exceeds_initial_size(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        for high in (101.4, 102.4, 103.4):
            env._check_sl_tp(self._hit_bar(high), fill_bid=high, fill_ask=high + 0.6)

        closes = self._tp_closes(env)
        assert len(closes) == 3
        total = sum(t["size"] for t in closes)
        assert total == pytest.approx(1.0)
        assert total <= float(pos.tp_initial_size or pos.size) + 1e-9
        assert env.position is None

    def test_partial_close_rounding_and_final_residual(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        # Thirds round down twice; the final slot must absorb the 0.01 remainder.
        self._enter_multi_tp(env, fractions=(0.333, 0.333, 0.334))

        for high in (101.4, 102.4, 103.4):
            env._check_sl_tp(self._hit_bar(high), fill_bid=high, fill_ask=high + 0.6)

        closes = self._tp_closes(env)
        assert [t["size"] for t in closes] == pytest.approx([0.33, 0.33, 0.34])
        assert sum(t["size"] for t in closes) == pytest.approx(1.0)
        assert env.position is None

    def test_multi_tp_does_not_double_execute(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        bar = BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6)
        env._check_sl_tp(bar, fill_bid=101.0, fill_ask=101.6)
        assert pos.tp_hit_mask == 1

        # A re-armed level beyond the bar must not pay a second time.
        pos.tp1_price = 101.0
        env._check_sl_tp(bar, fill_bid=101.0, fill_ask=101.6)

        assert len(self._tp_closes(env)) == 1
        assert pos.size == pytest.approx(0.7)
        assert pos.tp_hit_mask == 1

    def test_ambiguous_ohlc_uses_stop_before_any_target(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.0, high=102.5, low=98.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["type"] for row in fills] == ["stop_close"]
        assert fills[0]["position_complete"] is True

    @pytest.mark.parametrize(
        ("direction", "stop", "open_price", "high", "low", "expected_sign"),
        [
            (1, 99.0, 97.0, 100.0, 96.0, "below"),
            (-1, 101.0, 103.0, 104.0, 100.0, "above"),
        ],
    )
    def test_gap_through_stop_fills_at_open_or_worse(
        self, direction, stop, open_price, high, low, expected_sign
    ) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, direction)
        assert pos is not None
        pos.position_id = f"gap:{direction}"
        pos.sl_price = stop
        env.position = pos

        env._check_sl_tp(
            BarView(open=open_price, high=high, low=low, close=100.0, spread=0.0),
            fill_bid=100.0,
            fill_ask=100.0,
        )

        fill = next(row for row in env.trade_log if row.get("fill_id"))
        if expected_sign == "below":
            assert fill["price"] <= open_price
        else:
            assert fill["price"] >= open_price

    def test_bar_crossing_multiple_targets_preserves_slot_quantities(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.0, high=104.0, low=99.5, close=103.5, spread=0.6),
            fill_bid=103.5,
            fill_ask=104.1,
        )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["tp_slot"] for row in fills] == [1, 2, 3]
        assert [row["size"] for row in fills] == pytest.approx([0.3, 0.3, 0.4])
        assert [row["position_complete"] for row in fills] == [False, False, True]
        assert len({row["fill_id"] for row in fills}) == 3

    def test_single_target_tick_log_uses_delayed_fill_time(self) -> None:
        from quant_rl.data.ticks import TickBook

        env = _make_tp_env(fill_delay_ms=5_000)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        pos.position_id = "delayed:single"
        pos.entry_timestamp = env._bar_time(env.step_idx)
        pos.sl_price = 99.0
        env.position = pos
        start = pd.Timestamp(pos.entry_timestamp)
        fill_time = start + pd.Timedelta(seconds=20)
        env._tickbook = TickBook(
            np.asarray(
                [start.value, (start + pd.Timedelta(seconds=10)).value, fill_time.value],
                dtype=np.int64,
            ),
            np.asarray([100.0, 98.5, 98.0]),
            np.asarray([100.6, 99.1, 98.6]),
        )

        env._check_sl_tp(
            BarView(open=100.0, high=101.0, low=98.0, close=99.0, spread=0.6),
            fill_bid=99.0,
            fill_ask=99.6,
        )

        assert env.trade_log[-1]["time"] == fill_time

    def test_skipped_bar_replay_records_a_completed_stop_fill(self) -> None:
        env = _make_tp_env(block_overnight=False)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        pos.position_id = "replay:stop"
        pos.entry_timestamp = env._bar_time(env.step_idx - 1)
        pos.sl_price = float(env._bar_at(env.step_idx).low) + 0.1
        pos.sl_initial_price = pos.sl_price
        pos.planned_risk_usd = 1.0
        env.position = pos
        env.trade_log.append({"type": "open", "position_id": pos.position_id})
        skipped_idx = env.step_idx

        env._replay_skipped_bars(skipped_idx, skipped_idx + 1)

        assert env.position is None
        fill = [row for row in env.trade_log if row.get("fill_id")]
        assert len(fill) == 1
        assert fill[0]["position_id"] == "replay:stop"
        assert fill[0]["position_complete"] is True

    def test_tick_tp1_is_partial_and_then_stop_closes_remainder(self) -> None:
        from quant_rl.data.ticks import TickBook

        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)
        start = pd.Timestamp(env._bar_time(env.step_idx)).value
        seconds = np.array([1, 5, 10, 15], dtype=np.int64)
        env._tickbook = TickBook(
            start + seconds * 1_000_000_000,
            np.array([100.2, 101.2, 100.5, 98.8]),
            np.array([100.8, 101.8, 101.1, 99.4]),
        )

        env._check_sl_tp(
            BarView(open=100.2, high=101.2, low=98.8, close=99.1, spread=0.6),
            fill_bid=99.1,
            fill_ask=99.7,
        )
        closes = [r for r in env.trade_log if r.get("fill_id")]
        assert [r["type"] for r in closes] == ["tp_partial_close", "stop_close"]
        assert [r["size"] for r in closes] == pytest.approx([0.3, 0.7])
        assert closes[0]["tp_slot"] == 1
        assert closes[0]["price"] == pytest.approx(101.2)
        assert closes[1]["price"] == pytest.approx(98.8)
        assert pos.tp_hit_mask == 1
        assert env.position is None

    def test_tick_crosses_multiple_targets_in_one_minute(self) -> None:
        from quant_rl.data.ticks import TickBook

        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)
        start = pd.Timestamp(env._bar_time(env.step_idx)).value
        env._tickbook = TickBook(
            start + np.array([1, 7, 10], dtype=np.int64) * 1_000_000_000,
            np.array([100.1, 102.5, 103.3]),
            np.array([100.7, 103.1, 103.9]),
        )
        env._check_sl_tp(
            BarView(open=100.1, high=103.3, low=100.1, close=103.3, spread=0.6),
            fill_bid=103.3,
            fill_ask=103.9,
        )
        closes = self._tp_closes(env)
        assert [r["tp_slot"] for r in closes] == [1, 2, 3]
        assert [r["size"] for r in closes] == pytest.approx([0.3, 0.3, 0.4])
        assert sum(r["size"] for r in closes) == pytest.approx(1.0)
        assert env.position is None


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
