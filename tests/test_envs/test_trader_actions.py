"""Trader-like strategy_actions: context direction, RR, min SL, session caps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from quant_rl.envs.po3_reward import PO3Reward
from quant_rl.envs.strategies import BaselineStrategy, DistributionStrategy, PO3IFVGStrategy
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
    is_long = ctx_dir > 0
    return pd.DataFrame(
        {
            "asian_high": np.full(n, 102.0),
            "asian_low": np.full(n, 98.0),
            "london_high": np.full(n, 103.0),
            "london_low": np.full(n, 97.0),
            "sweep_high": np.zeros(n) if is_long else np.ones(n),
            "sweep_low": np.ones(n) if is_long else np.zeros(n),
            "po3_manipulation_low": np.full(n, 95.0),
            "po3_manipulation_high": np.full(n, 105.0),
            "po3_manipulation_end": np.ones(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "price_in_ifvg_bull": np.ones(n) if is_long else np.zeros(n),
            "price_in_ifvg_bear": np.zeros(n) if is_long else np.ones(n),
            "ifvg_retest_bull": np.ones(n) if is_long else np.zeros(n),
            "ifvg_retest_bear": np.zeros(n) if is_long else np.ones(n),
            "ifvg_bear_low": np.zeros(n) if is_long else np.full(n, 99.0),
            "ifvg_bear_high": np.zeros(n) if is_long else np.full(n, 100.0),
            "ifvg_bull_active": np.ones(n) if is_long else np.zeros(n),
            "ifvg_bear_active": np.zeros(n) if is_long else np.ones(n),
            "last_swing_high": np.full(n, np.nan),
            "last_swing_low": np.full(n, np.nan),
            "htf_day_bias": np.full(n, ctx_dir),
            "context_trade_direction": np.full(n, ctx_dir),
            "manip_reverses_htf": np.zeros(n),
            "atr_5": np.full(n, 1.0),
            "prev_day_high": np.full(n, 150.0),
            "prev_day_low": np.full(n, 50.0),
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
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0, -1.0, -1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0, 1.0, 1.0])

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
        # Intensity in Box[-1,1]: a=-0.9 → u=0.05; explicit 0.1 band still holds.
        env.entry_intensity_threshold = 0.1
        for _ in range(20):
            obs, _, done, truncated, _ = env.step(np.array([-0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            assert env.position is None  # hold (intensity < entry_intensity_threshold)
            if done or truncated:
                break
        _ = obs


class TestRRMapping:
    def test_reward_fraction_tracks_the_action(self) -> None:
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
            _d, _r, frac, _tp = env._decode_action(
                np.array([0.9, 0.0, 0.5, t], dtype=np.float32),
                feats.iloc[10],
            )
            assert frac == pytest.approx(0.5 * (t + 1.0))


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
            open_manipulation_bars=0,
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

    def test_unlimited_entries_allow_a_fourth_open(self) -> None:
        n = 200
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
            max_entries_per_session=0,
            open_manipulation_bars=0,
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
            if env._session_entry_counts.get(0, 0) >= 4:
                return
        pytest.fail("session stayed capped below 4 entries")

    def test_stale_distribution_at_the_open_does_not_trade(self) -> None:
        bars = _bars(n=30)
        feats = _features(bars, ctx_dir=1.0)
        feats["po3_manipulation_end"] = 0.0
        feats["po3_distribution"] = 1.0
        feats["po3_distribution_direction"] = 1.0
        feats["sweep_low"] = 0.0
        feats["sweep_high"] = 0.0
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=5,
            open_manipulation_bars=0,
            entry_cooldown_bars=0,
            block_overnight=False,
        )
        env.reset()
        for _ in range(20):
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            assert env.position is None

    def test_post_open_manipulation_end_allows_the_trade(self) -> None:
        n = 40
        bars = _bars(n=n)
        feats = _features(bars, ctx_dir=1.0)
        feats["po3_manipulation_end"] = 0.0
        feats["po3_distribution"] = 1.0
        feats["po3_distribution_direction"] = 1.0
        feats.loc[feats.index[12], "po3_manipulation_end"] = 1.0
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=5,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
            min_sl_points=1.0,
            min_sl_atr_mult=0.0,
            open_manipulation_bars=10,
            entry_cooldown_bars=0,
            block_overnight=False,
            max_episode_steps=None,
        )
        env.reset()
        opened_at = None
        for _ in range(n - 5):
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            if env.position is not None:
                opened_at = env.step_idx
                break
        assert opened_at is not None
        assert opened_at >= 12

    def test_first_ten_minutes_stay_flat(self) -> None:
        bars = _bars(n=20)
        feats = _features(bars)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=5,
            open_manipulation_bars=10,
            entry_cooldown_bars=0,
            block_overnight=False,
        )
        env.reset()
        for _ in range(10):
            env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
            assert env.position is None


def test_risk_at_the_floor_does_not_open() -> None:
    n = 30
    bars = _bars(n=n)
    feats = _features(bars, ctx_dir=1.0)
    feats["po3_manipulation_low"] = np.full(n, 80.0)
    feats["asian_low"] = np.nan
    feats["london_low"] = np.nan
    env = TradingEnv(
        bars,
        feats,
        strategy_actions=True,
        strategy=PO3IFVGStrategy(enforce_gate=False),
        obs_window=5,
        risk_frac_range=(0.0, 0.01),
        risk_floor=0.0005,
        rr_ratio_range=(2.0, 2.0),
        min_sl_points=1.0,
        min_sl_atr_mult=0.0,
        open_manipulation_bars=0,
        entry_cooldown_bars=0,
        block_overnight=False,
    )
    env.reset()
    for _ in range(15):
        env.step(np.array([0.9, 0.0, -1.0, 0.0], dtype=np.float32))
        assert env.position is None


def test_confirmed_entry_is_shaped_once() -> None:
    """A confirmed in-zone entry scores once; later bars of the hold do not."""
    n = 40
    bars = _bars(n=n)
    feats = _features(bars, ctx_dir=1.0)
    feats["po3_distribution"] = 1.0
    feats["po3_manipulation_low"] = np.full(n, 80.0)
    feats["asian_low"] = np.nan
    feats["london_low"] = np.nan
    env = TradingEnv(
        bars,
        feats,
        strategy_actions=True,
        strategy=PO3IFVGStrategy(enforce_gate=False),
        strategy_reward=PO3Reward(
            entry_bonus=0.01,
            manipulation_penalty=0.02,
            invalid_ifvg_penalty=0.01,
            distribution_bonus=0.005,
        ),
        strategy_weight=1.0,
        reward_mode="pnl",
        obs_window=5,
        risk_frac_range=(0.01, 0.01),
        rr_ratio_range=(2.0, 2.0),
        min_sl_points=1.0,
        min_sl_atr_mult=0.0,
        open_manipulation_bars=0,
        entry_cooldown_bars=0,
        block_overnight=False,
        max_episode_steps=None,
    )
    env.reset()
    opened = False
    for _ in range(n - 5):
        _obs, _reward, _done, _trunc, info = env.step(
            np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32)
        )
        shaped = float(info["reward_parts"]["strategy"])
        if env.position is not None and not opened:
            assert shaped > 0.0
            opened = True
            continue
        if opened:
            assert shaped == 0.0
            assert env.position is not None
            return
    pytest.fail("confirmed entry never opened")


def _policy_env(
    feats: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    direction_control: bool = False,
    strategy: Any = None,
) -> TradingEnv:
    return TradingEnv(
        bars,
        feats,
        strategy_actions=True,
        strategy=strategy or PO3IFVGStrategy(enforce_gate=False),
        agent_direction_control=direction_control,
        obs_window=5,
        risk_frac_range=(0.01, 0.01),
        min_sl_points=0.0,
        min_sl_atr_mult=0.0,
        open_manipulation_bars=0,
        entry_cooldown_bars=0,
        block_overnight=False,
        max_episode_steps=None,
    )


def test_sweep_entry_stays_open_while_manipulation_is_active() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 1.0
    feats["po3_manipulation_end"] = 0.0
    env = _policy_env(feats, bars)
    env.reset()
    env.step(np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert env.position is not None
    assert env.position.direction == 1
    assert env.position.sl_price is not None


def test_direction_override_is_refused_until_the_close_through() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 1.0
    feats["po3_manipulation_end"] = 0.0
    env = _policy_env(feats, bars, direction_control=True)
    env.reset()
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([-1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert env.position is None
    assert info["entry_rejection_reason"] == "manipulation_unconfirmed"
    confirmed = _policy_env(feats, bars, direction_control=True)
    confirmed.reset()
    confirmed.step(np.array([1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert confirmed.position is not None
    assert confirmed.position.direction == 1


def test_unconfirmed_leg_does_not_invent_a_side() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 1.0
    feats["po3_manipulation_end"] = 0.0
    feats["sweep_low"] = 0.0
    feats["sweep_high"] = 0.0
    env = _policy_env(feats, bars, direction_control=True)
    env.reset()
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert env.position is None
    assert info["entry_rejection_reason"] == "manipulation_unconfirmed"


def test_manipulation_end_alone_does_not_open_idea1() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 0.0
    feats["po3_manipulation_end"] = 1.0
    feats["po3_distribution"] = 1.0
    feats["po3_distribution_direction"] = 1.0
    feats["sweep_low"] = 0.0
    feats["sweep_high"] = 0.0
    env = _policy_env(feats, bars, direction_control=True)
    env.reset()
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([-1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert env.position is not None
    assert info["entry_rejection_reason"] == ""
    plain = _policy_env(feats, bars)
    plain.reset()
    plain.step(np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert plain.position is None


def test_distribution_strategy_stays_flat_during_the_active_leg() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 1.0
    feats["po3_manipulation_end"] = 0.0
    feats["po3_distribution"] = 0.0
    feats["po3_distribution_after_ifvg"] = 0.0
    feats["sweep_high_level"] = 110.0
    feats["sweep_low_level"] = 90.0
    feats["sweep_high_reclaimed"] = 0.0
    feats["sweep_low_reclaimed"] = 0.0
    feats["bos_up"] = 0.0
    feats["bos_down"] = 0.0
    env = _policy_env(feats, bars, strategy=DistributionStrategy(enforce_gate=True))
    env.reset()
    env.step(np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert env.position is None


def test_short_manipulation_mirrors_the_long_lock() -> None:
    bars = _bars(n=20)
    feats = _features(bars, ctx_dir=-1.0)
    feats["po3_manipulation_active"] = 1.0
    feats["po3_manipulation_end"] = 0.0
    refused = _policy_env(feats, bars, direction_control=True)
    refused.reset()
    _obs, _reward, _done, _trunc, info = refused.step(
        np.array([1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert refused.position is None
    assert info["entry_rejection_reason"] == "manipulation_unconfirmed"
    taken = _policy_env(feats, bars, direction_control=True)
    taken.reset()
    taken.step(np.array([-1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert taken.position is not None
    assert taken.position.direction == -1


def test_close_targets_inside_the_stop_leave_no_order() -> None:
    bars = _bars(n=20)
    feats = _features(bars)
    feats["po3_manipulation_active"] = 0.0
    feats["po3_manipulation_end"] = 1.0
    feats["asian_high"] = 101.0
    feats["london_high"] = 101.0
    feats["prev_day_high"] = 101.0
    feats["last_swing_high"] = np.nan
    feats["ifvg_bear_low"] = np.nan
    feats["ifvg_bear_high"] = np.nan
    env = _policy_env(feats, bars)
    env.reset()
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert env.position is None
    assert info["entry_rejection_reason"] == "no_valid_tp"


def test_a_later_reclaim_does_not_unlock_the_earlier_bar() -> None:
    from quant_rl.features.po3_state import build_po3_state

    idx = pd.date_range("2020-01-02 16:30", periods=8, freq="1min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": [101.0, 100.5, 100.4, 102.0, 102.0, 102.0, 102.0, 102.0],
            "low": [99.0, 97.0, 98.0, 100.0, 101.0, 101.0, 101.0, 101.0],
            "close": [100.0, 99.0, 99.5, 101.5, 101.6, 101.7, 101.8, 101.9],
            "volume": 1,
            "session_id": 0,
            "session": "ny",
        },
        index=idx,
    )
    sweeps = pd.DataFrame(0.0, index=idx, columns=["sweep_high", "sweep_low"])
    sweeps.loc[sweeps.index[1], "sweep_low"] = 1.0
    asian = pd.DataFrame({"asian_high": 103.0, "asian_low": 98.0}, index=idx)
    full = build_po3_state(bars, sweeps, asian)
    early = build_po3_state(bars.iloc[:3], sweeps.iloc[:3], asian.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], early, check_freq=False)
    assert float(early["po3_manipulation_end"].iloc[2]) == 0.0
    feats = _features(bars.iloc[:3])
    feats["po3_manipulation_active"] = early["po3_manipulation_active"].to_numpy()
    feats["po3_manipulation_end"] = early["po3_manipulation_end"].to_numpy()
    feats["po3_manipulation_low"] = early["po3_manipulation_low"].to_numpy()
    feats["po3_manipulation_high"] = early["po3_manipulation_high"].to_numpy()
    env = _policy_env(feats, bars.iloc[:3], direction_control=True)
    env.reset()
    env.step_idx = 2
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([-1.0, 0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert info["entry_rejection_reason"] == "manipulation_unconfirmed"
    assert env.position is None
