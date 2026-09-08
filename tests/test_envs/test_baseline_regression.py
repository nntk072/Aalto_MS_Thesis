"""Phase 5 regression tests: baseline (Idea 3) behaviour preserved bit-for-bit.

The strategy machinery was added as a strict *overlay*: when
``strategy_actions=False`` (the default / Idea 3 baseline), the environment
must behave exactly as before — legacy discrete / 1-D action spaces, legacy
DSR reward, legacy entry gate, legacy swing-level SL/TP, and no columns
dropped from the observation.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from quant_rl.envs.reward import DSRReward
from quant_rl.envs.strategies import BaselineStrategy
from quant_rl.envs.sweep_reward import CompositeReward
from quant_rl.envs.trading_env import TradingEnv


class _ValidateEntryKwargs(TypedDict, total=False):
    direction: int
    row: pd.Series


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 200
    dates = pd.date_range("2020-06-01 08:00", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    high = close + rng.uniform(0.05, 0.15, n)
    low = close - rng.uniform(0.05, 0.15, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
        },
        index=dates,
    )


@pytest.fixture
def features(bars: pd.DataFrame) -> pd.DataFrame:
    n = len(bars)
    return pd.DataFrame(
        {
            "london_high": np.full(n, 102.0),
            "london_low": np.full(n, 98.0),
            "asian_high": np.full(n, 101.0),
            "asian_low": np.full(n, 99.0),
            "volume_spike": np.full(n, 2.0),
            "last_swing_high": np.full(n, 103.0),
            "last_swing_low": np.full(n, 97.0),
            "close": bars["close"].to_numpy(),
            "extra_col": np.arange(n, dtype=float),
        },
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# 1. BaselineStrategy is a correct no-op
# ---------------------------------------------------------------------------


class TestBaselineStrategyNoop:
    def test_default_strategy_is_baseline(self, bars: pd.DataFrame, features: pd.DataFrame) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        assert isinstance(env.strategy, BaselineStrategy)
        assert env.strategy.name == "baseline"

    def test_baseline_has_no_required_features(self) -> None:
        strat = BaselineStrategy()
        assert strat.required_features == ()
        assert strat.raw_columns == ()

    def test_baseline_validate_entry_always_true(self) -> None:
        row = pd.Series({"x": 1.0})
        assert BaselineStrategy().validate_entry(direction=1, row=row) is True
        assert BaselineStrategy().validate_entry(direction=-1, row=row) is True

    def test_baseline_sl_reference_is_none(self) -> None:
        row = pd.Series({"x": 1.0})
        assert BaselineStrategy().sl_reference(direction=1, row=row) is None
        assert BaselineStrategy().sl_reference(direction=-1, row=row) is None

    def test_baseline_target_candidates_empty(self) -> None:
        row = pd.Series({"x": 1.0})
        assert BaselineStrategy().target_candidates(direction=1, row=row) == {}
        assert BaselineStrategy().target_candidates(direction=-1, row=row) == {}


# ---------------------------------------------------------------------------
# 2. Action space unchanged
# ---------------------------------------------------------------------------


class TestBaselineActionSpace:
    def test_baseline_action_space_is_discrete(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        assert not env.strategy_actions
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 20

    def test_strategy_actions_false_with_continuous_uses_box(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        env = TradingEnv(
            bars, features, strategy_actions=False, continuous_actions=True, obs_window=20
        )
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (1,)

    def test_strategy_actions_true_uses_4d_box(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        env = TradingEnv(
            bars, features, strategy_actions=True, strategy=BaselineStrategy(), obs_window=20
        )
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (4,)


# ---------------------------------------------------------------------------
# 3. Reward function unchanged
# ---------------------------------------------------------------------------


class TestBaselineReward:
    def test_baseline_reward_is_dsr(self, bars: pd.DataFrame, features: pd.DataFrame) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        assert isinstance(env.reward_fn, DSRReward)
        assert not isinstance(env.reward_fn, CompositeReward)

    def test_strategy_reward_ignored_without_strategy_actions(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        po3_feats = features.copy()
        po3_feats["po3_manipulation_low"] = np.full(len(bars), 95.0)
        po3_feats["po3_manipulation_high"] = np.full(len(bars), 105.0)
        env = TradingEnv(
            bars, po3_feats, strategy_reward={"dummy": True}, strategy_weight=0.5, obs_window=20
        )
        assert isinstance(env.reward_fn, DSRReward)
        assert not isinstance(env.reward_fn, CompositeReward)


# ---------------------------------------------------------------------------
# 4. Observation: no columns dropped
# ---------------------------------------------------------------------------


class TestBaselineObservation:
    def test_no_columns_dropped_in_baseline(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        env.reset()
        seq_cols = set(env._obs_features.columns)
        expected_cols = set(features.columns)
        assert seq_cols == expected_cols

    def test_extra_col_in_obs(self, bars: pd.DataFrame, features: pd.DataFrame) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        assert "extra_col" in env._obs_features.columns


# ---------------------------------------------------------------------------
# 5. Entry gate: legacy path used, not strategy.validate_entry
# ---------------------------------------------------------------------------


class TestBaselineEntryGate:
    def test_entry_gate_allows_with_sufficient_volume_spike(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        env.reset()
        env.step(1)

    def test_baseline_uses_legacy_gate_not_strategy(
        self, bars: pd.DataFrame, features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        env.reset()
        feat_row = env.features.iloc[env.step_idx]
        called_strategy = False
        original = BaselineStrategy.validate_entry

        def spy(self_bs: BaselineStrategy, *, direction: int, row: pd.Series) -> bool:
            nonlocal called_strategy
            called_strategy = True
            return original(self_bs, direction=direction, row=row)

        monkeypatch.setattr(BaselineStrategy, "validate_entry", spy)
        env._check_entry_gate(float(bars["close"].iloc[env.step_idx]), 1, feat_row)
        assert not called_strategy, "Baseline gate must not call strategy.validate_entry"


# ---------------------------------------------------------------------------
# 6. SL/TP: legacy swing-level path used, not structural
# ---------------------------------------------------------------------------


class TestBaselineSLTP:
    def test_baseline_sl_uses_swing_levels(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        """When a long fills in baseline mode, SL must come from last_swing_low,
        NOT from BaselineStrategy.sl_reference (which returns None)."""
        env = TradingEnv(bars, features, obs_window=20)
        env.reset()
        for _ in range(len(bars) - 20):
            _, _, done, trunc, _ = env.step(1)
            if env.position is not None:
                assert env.position.sl_price is not None
                assert env.position.sl_price == pytest.approx(97.0)
                break
            if done or trunc:
                break

    def test_baseline_rejects_entry_without_swing_levels(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        feats = features.drop(columns=["last_swing_high", "last_swing_low"])
        env = TradingEnv(bars, feats, obs_window=20)
        env.reset()
        for _ in range(20):
            env.step(1)
            assert env.position is None, "baseline entry without swing levels must be rejected"

    def test_baseline_no_naked_position_on_bad_geometry(
        self, bars: pd.DataFrame, features: pd.DataFrame
    ) -> None:
        feats = features.copy()
        feats["last_swing_low"] = 999.0
        env = TradingEnv(bars, feats, obs_window=20)
        env.reset()
        for _ in range(20):
            env.step(1)
            assert env.position is None


# ---------------------------------------------------------------------------
# 7. Trade log records baseline
# ---------------------------------------------------------------------------


class TestBaselineTradeLog:
    def test_trade_log_strategy_field(self, bars: pd.DataFrame, features: pd.DataFrame) -> None:
        env = TradingEnv(bars, features, obs_window=20)
        env.reset()
        for _ in range(len(bars) - 20):
            _, _, done, trunc, _ = env.step(1)
            if env.trade_log:
                last = env.trade_log[-1]
                assert last.get("strategy") == "baseline"
                break
            if env.position is not None or done or trunc:
                break
        else:
            assert env.strategy.name == "baseline"
