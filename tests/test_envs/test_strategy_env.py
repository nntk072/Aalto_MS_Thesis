"""Tests for strategy-mode environment (Agent.md §12, §27, §28, §29)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Dict

from quant_rl.envs.strategies import BaselineStrategy, PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv


def _bars(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01 01:05", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
        },
        index=idx,
    )


def _features(bars: pd.DataFrame) -> pd.DataFrame:
    n = len(bars)
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    return pd.DataFrame(
        {
            "asian_high": np.full(n, float(high.max()) + 1.0),
            "asian_low": np.full(n, float(low.min()) - 1.0),
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
            "last_swing_high": np.full(n, float(high.max()) + 1.0),
            "last_swing_low": np.full(n, float(low.min()) - 1.0),
        },
        index=bars.index,
    )


class TestStrategyActionSpace:
    def test_4d_box_shape_and_bounds(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
        )
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (4,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0, 1.0])

    def test_action_contains_valid_4d(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
        )
        action = np.array([0.8, 0.5, 0.7, 0.0], dtype=np.float32)
        assert env.action_space.contains(action)

    def test_baseline_keeps_legacy_space(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(bars, feats, strategy_actions=False, obs_window=10)
        # Baseline (strategy_actions=False) must NOT use the 4-D Box.
        assert env.action_space.shape != (4,)


class TestStrategyObservation:
    def test_raw_columns_excluded_from_seq(self) -> None:
        bars = _bars()
        feats = _features(bars)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            obs_window=10,
        )
        obs = env.reset()[0]
        assert isinstance(obs, dict)
        assert "seq" in obs
        seq_width = int(obs["seq"].shape[1])
        assert obs["seq"].shape == (10, seq_width)
        assert env.observation_space.contains(obs)


class TestStrategyStructuralSL:
    def test_long_sl_at_manipulation_low(self) -> None:
        """Idea 1 long: SL = manipulation_low (exact mode, Agent.md §27)."""
        bars = _bars()
        feats = _features(bars)
        feats["po3_manipulation_low"] = np.full(len(bars), 100.0)
        feats["po3_manipulation_high"] = np.full(len(bars), 110.0)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            sl_buffer_pts=0.0,
            obs_window=10,
            initial_balance=100_000.0,
            risk_frac_range=(0.01, 0.01),
            rr_ratio_range=(2.0, 2.0),
        )
        env.reset()
        # Step until a long entry is filled: strong long action each bar.
        obs, _ = env.reset()
        filled = False
        for _ in range(len(bars) - 5):
            action = np.array([0.9, 0.5, 0.5, 0.0], dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)
            if env.position is not None:
                assert env.position.sl_price == pytest.approx(100.0)
                filled = True
                break
            if done or truncated:
                break
        assert filled, "expected a long position to be opened with SL=100.0"

    def test_invalid_geometry_rejected(self) -> None:
        """Manipulation_low >= entry must reject, not create a backwards SL."""
        bars = _bars()
        feats = _features(bars)
        # Set the manipulation low above every price -> invalid long geometry.
        feats["po3_manipulation_low"] = np.full(len(bars), 999.0)
        env = TradingEnv(
            bars,
            feats,
            strategy_actions=True,
            strategy=PO3IFVGStrategy(enforce_gate=False),
            sl_buffer_pts=0.0,
            obs_window=10,
        )
        env.reset()
        opened = False
        for _ in range(20):
            action = np.array([0.9, 0.5, 0.5, 0.0], dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)
            if env.position is not None:
                opened = True
                break
        assert not opened, "invalid geometry must not open a position"
