"""Trader-like strategy_actions: context direction, RR, min SL, session caps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from quant_rl.envs.strategies import BaselineStrategy, PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv


def _bars(n: int = 200, session_ids: np.ndarray[Any, Any] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01 16:30", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    # Keep path tight so structural levels stay valid.
    close = np.clip(close, 99.0, 101.0)
    high = close + 0.05
    low = close - 0.05
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    sid = np.zeros(n, dtype=int) if session_ids is None else session_ids
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": sid,
            "session": "ny",
        },
        index=idx,
    )


def _features(bars: pd.DataFrame, *, ctx_dir: float = 1.0) -> pd.DataFrame:
    n = len(bars)
    return pd.DataFrame(
        {
            "asian_high": np.full(n, 102.0),
            "asian_low": np.full(n, 98.0),
            "london_high": np.full(n, 103.0),
            "london_low": np.full(n, 97.0),
            "sweep_high": np.zeros(n),
            "sweep_low": np.zeros(n),
            "po3_manipulation_low": np.full(n, 95.0),
            "po3_manipulation_high": np.full(n, 105.0),
            "po3_manipulation_end": np.zeros(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "price_in_ifvg_bull": np.ones(n),
            "ifvg_bear_low": np.zeros(n),
            "ifvg_bear_high": np.zeros(n),
            "ifvg_bull_active": np.ones(n),
            "ifvg_bear_active": np.zeros(n),
            "last_swing_high": np.full(n, np.nan),
            "last_swing_low": np.full(n, np.nan),
            "htf_day_bias": np.full(n, ctx_dir),
            "context_trade_direction": np.full(n, ctx_dir),
            "manip_reverses_htf": np.zeros(n),
            "atr_5": np.full(n, 1.0),
        },
        index=bars.index,
    )


class TestTraderActionSpace:
    def test_box_is_symmetric_unit(self) -> None:
        bars = _bars()
        env = TradingEnv(
            bars,
            _features(bars),
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.005, 0.01),
            rr_ratio_range=(1.5, 5.0),
        )
        assert isinstance(env.action_space, Box)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0, 1.0])

    def test_baseline_discrete20_unchanged(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(bars, feats, strategy_actions=False, obs_window=10)
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 20
        assert isinstance(env.strategy, BaselineStrategy)


class TestZeroActionOpens:
    """PPO deterministic mean is ~0; that must map to intensity 0.5 and open."""

    def test_zeros_open_long_when_context_and_sl_valid(self) -> None:
        bars = _bars()
        feats = _features(bars, ctx_dir=1.0)
        # Wide structural SL so min_sl_points does not reject.
        feats["po3_manipulation_low"] = np.full(len(bars), 80.0)
        feats["asian_low"] = np.nan
        feats["london_low"] = np.nan
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=5.0,
            min_sl_atr_mult=0.0,
            max_loss_per_trade_usd=1000.0,
        )
        env.reset()
        opened = False
        for _ in range(40):
            env.step(np.zeros(4, dtype=np.float32))
            if env.position is not None:
                opened = True
                assert env.position.direction == 1
                break
        assert opened, "zero action (PPO mean) should open when context+SL valid"


class TestContextDirectionForcesSide:
    def test_long_context_opens_long_only(self) -> None:
        bars = _bars()
        feats = _features(bars, ctx_dir=1.0)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=0.0,
            min_sl_atr_mult=0.0,
            max_entries_per_session=0,
            entry_cooldown_bars=0,
        )
        env.reset()
        # Intensity high; context is long — should open long.
        for _ in range(30):
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            if env.position is not None:
                assert env.position.direction == 1
                break
        else:
            pytest.fail("expected a long open")

    def test_opposite_side_not_openable_via_action(self) -> None:
        """No free long/short dim: action cannot force short when context is long."""
        bars = _bars()
        feats = _features(bars, ctx_dir=1.0)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=0.0,
            min_sl_atr_mult=0.0,
        )
        env.reset()
        # Intensity in Box[-1,1]: a=-0.9 → u=0.05 < 0.1 hold threshold.
        for _ in range(20):
            obs, _, done, truncated, _ = env.step(np.array([-0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            assert env.position is None  # hold (intensity < entry_intensity_threshold)
            if done or truncated:
                break
        _ = obs


class TestRRMapping:
    def test_rr_always_at_least_1_5(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.005, 0.01),
            rr_ratio_range=(1.5, 5.0),
        )
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            _d, _r, rr, _tp = env._decode_action(
                np.array([0.9, 0.0, 0.5, t], dtype=np.float32),
                feats.iloc[10],
            )
            assert rr >= 1.5 - 1e-9


class TestMinSL:
    def test_rejects_two_point_stop_when_min_sl_points_20(self) -> None:
        bars = _bars()
        feats = _features(bars)
        # Only candidate: 2 pts below entry (~100) → 98; min_sl_points=20 rejects.
        feats["po3_manipulation_low"] = np.full(len(bars), 98.0)
        feats["asian_low"] = np.full(len(bars), np.nan)
        feats["london_low"] = np.full(len(bars), np.nan)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=20.0,
            min_sl_atr_mult=0.0,
        )
        env.reset()
        for _ in range(40):
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            assert env.position is None


class TestLotsScaleWithSL:
    def test_lots_increase_when_sl_distance_shrinks(self) -> None:
        bars = _bars()
        feats_wide = _features(bars)
        feats_wide["po3_manipulation_low"] = np.full(len(bars), 80.0)  # 20 pts
        feats_wide["asian_low"] = np.nan
        feats_wide["london_low"] = np.nan

        feats_tight = _features(bars)
        feats_tight["po3_manipulation_low"] = np.full(len(bars), 90.0)  # 10 pts
        feats_tight["asian_low"] = np.nan
        feats_tight["london_low"] = np.nan

        common: dict[str, Any] = dict(
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=5.0,
            min_sl_atr_mult=0.0,
            max_loss_per_trade_usd=1000.0,
            initial_balance=100_000.0,
        )
        env_w = TradingEnv(bars, feats_wide, **common)
        env_t = TradingEnv(bars, feats_tight, **common)
        lots_w = lots_t = None
        for env, sink in ((env_w, "w"), (env_t, "t")):
            env.reset()
            for _ in range(40):
                env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
                if env.position is not None:
                    if sink == "w":
                        lots_w = env.position.size
                    else:
                        lots_t = env.position.size
                    break
        assert lots_w is not None and lots_t is not None
        assert lots_t > lots_w


class TestSessionCaps:
    def test_max_three_entries_per_session(self) -> None:
        n = 300
        bars = _bars(n=n)
        feats = _features(bars)
        feats["po3_manipulation_low"] = np.full(n, 90.0)
        feats["asian_low"] = np.nan
        feats["london_low"] = np.nan
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=1.0,
            min_sl_atr_mult=0.0,
            max_entries_per_session=3,
            entry_cooldown_bars=0,
            block_overnight=False,
            max_episode_steps=None,
        )
        env.reset()
        for _ in range(n - 20):
            if env.position is not None:
                pnl, _fp = env.broker.close_position(
                    env.account,
                    env.position,
                    env._bar_quote(env._bar_at(env.step_idx)),
                )
                env.trade_log.append({"type": "close", "pnl": pnl})
                env.position = None
                env._note_close(pnl)
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            if env._session_entry_counts.get(0, 0) >= 3 and env.position is None:
                # Fourth attempt must not open.
                before = env._session_entry_counts.get(0, 0)
                env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
                assert env.position is None
                assert env._session_entry_counts.get(0, 0) == before == 3
                return
        pytest.fail("never reached 3 session entries")
