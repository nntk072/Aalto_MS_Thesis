"""Tests for continuous action space and strategy entry eligibility."""

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from simulation.environment.env import TradingEnv


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Create sample bars for testing."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="1min")
    data = {
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(100, 110, n),
        "low": np.random.uniform(95, 100, n),
        "close": np.random.uniform(95, 110, n),
        "volume": np.random.randint(1000, 5000, n),
        "session_id": [0] * n,
    }
    bars = pd.DataFrame(data, index=dates)
    # Make high >= open, close and low <= open, close for valid OHLC
    bars["high"] = bars[["open", "close", "high"]].max(axis=1)
    bars["low"] = bars[["open", "close", "low"]].min(axis=1)
    return bars


class TestContinuousActionSpace:
    """Tests for continuous action space functionality."""

    @pytest.fixture
    def sample_features(self, sample_bars: pd.DataFrame) -> pd.DataFrame:
        """Create sample features."""
        # Add required columns for entry gate
        features = pd.DataFrame(index=sample_bars.index)
        features["london_high"] = 105.0
        features["london_low"] = 95.0
        features["asian_high"] = 103.0
        features["asian_low"] = 97.0
        features["volume_spike"] = np.random.uniform(0.5, 2.5, len(sample_bars))
        return features

    def test_continuous_action_space(
        self, sample_bars: pd.DataFrame, sample_features: pd.DataFrame
    ) -> None:
        """Test that continuous action space is Box(-1, 1)."""
        env = TradingEnv(
            sample_bars,
            sample_features,
            continuous_actions=True,
            obs_window=10,
        )

        # Check action space is Box(-1, 1)
        assert env.action_space.shape == (1,)
        assert isinstance(env.action_space, Box)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_discrete_action_space(
        self, sample_bars: pd.DataFrame, sample_features: pd.DataFrame
    ) -> None:
        """Test that discrete action space is Discrete(20)."""
        env = TradingEnv(
            sample_bars,
            sample_features,
            continuous_actions=False,
            obs_window=10,
        )

        # Check action space is Discrete(20)
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 20

    def test_continuous_action_mapping(
        self, sample_bars: pd.DataFrame, sample_features: pd.DataFrame
    ) -> None:
        """Test that continuous actions are mapped correctly."""
        env = TradingEnv(
            sample_bars,
            sample_features,
            continuous_actions=True,
            max_risk_frac=0.01,
            obs_window=10,
        )
        env.reset()

        for action_val in [0.5, -0.5, 0.0]:
            action = np.array([action_val])
            obs, _, _, _, info = env.step(action)
            assert np.isfinite(obs["seq"]).all()
            assert "position" in info

    @pytest.mark.parametrize("direction", [1, -1])
    @pytest.mark.parametrize("already_beyond", [True, False])
    def test_session_levels_do_not_gate_entries(
        self, sample_bars: pd.DataFrame, direction: int, already_beyond: bool
    ) -> None:
        """Asian/London levels inform order placement, not entry eligibility."""
        sample_bars["close"] = 100.0
        high = 99.0 if direction == 1 and already_beyond else 101.0
        low = 101.0 if direction == -1 and already_beyond else 99.0
        features = pd.DataFrame(
            {
                "london_high": np.full(len(sample_bars), high),
                "london_low": np.full(len(sample_bars), low),
                "asian_high": np.full(len(sample_bars), high),
                "asian_low": np.full(len(sample_bars), low),
                "volume_spike": np.full(len(sample_bars), 2.0),
            },
            index=sample_bars.index,
        )
        env = TradingEnv(sample_bars, features, continuous_actions=True, obs_window=10)
        env.reset()

        env.step(np.array([0.5 * direction]))

        assert env._entry_diag["rejected_gate"] == 0

    def test_hold_always_allowed(
        self, sample_bars: pd.DataFrame, sample_features: pd.DataFrame
    ) -> None:
        """Test that hold action is always allowed."""
        env = TradingEnv(
            sample_bars,
            sample_features,
            continuous_actions=True,
            obs_window=10,
        )
        env.reset()

        # Hold action should always be allowed
        action = np.array([0.0])
        obs, _, _, _, _ = env.step(action)

        # Position should remain None
        assert env.position is None

    def test_exit_always_allowed(
        self, sample_bars: pd.DataFrame, sample_features: pd.DataFrame
    ) -> None:
        """Test that exit is always allowed."""
        env = TradingEnv(
            sample_bars,
            sample_features,
            continuous_actions=False,  # Use discrete for exit test
            obs_window=10,
        )
        env.reset()

        # Enter a position first (action 1 = enter_long)
        action = 1
        obs, _, _, _, _ = env.step(action)

        # Exit should be allowed
        action = 19  # exit
        obs, _, _, _, info = env.step(action)


class TestRiskFallback:
    """Tests for the no-naked-position fallback."""

    def test_no_naked_position_without_swing_levels(self, sample_bars: pd.DataFrame) -> None:
        """Entry without swing levels must be rejected, not opened naked at 1.0 lots.

        The old fallback opened lots=1.0 with sl_price=None/tp_price=None when
        no swing levels were available, bypassing the entire risk framework.
        """
        # No stop or session-level candidates are available.
        features = pd.DataFrame(
            {"dummy_feature": np.zeros(len(sample_bars))}, index=sample_bars.index
        )

        env = TradingEnv(sample_bars, features, continuous_actions=False, obs_window=10)
        env.reset()

        # Entry is allowed; risk geometry still requires a stop.
        sample_bars.loc[sample_bars.index[10], "close"] = 105.0

        obs, _, _, _, _ = env.step(1)  # enter_long

        # Phase A fix: no swing levels → entry rejected (treated as hold)
        assert env.position is None
        assert env.trade_log == []
