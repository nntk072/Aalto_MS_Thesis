"""Tests for continuous action space and entry gate."""

import numpy as np
import pandas as pd
import pytest

from quant_rl.envs.trading_env import TradingEnv


class TestContinuousActionSpace:
    """Tests for continuous action space functionality."""

    @pytest.fixture
    def sample_bars(self) -> pd.DataFrame:
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

    def test_entry_gate_long_allowed(self, sample_bars: pd.DataFrame) -> None:
        """Test that long entry is allowed when conditions are met."""
        # Create features with long conditions met
        features = pd.DataFrame(index=sample_bars.index)
        features["london_high"] = 100.0
        features["london_low"] = 90.0
        features["asian_high"] = 98.0
        features["asian_low"] = 92.0
        features["volume_spike"] = 2.0  # > 1.5

        env = TradingEnv(
            sample_bars,
            features,
            continuous_actions=True,
            obs_window=10,
        )
        env.reset()

        # Set price above London High
        sample_bars.loc[sample_bars.index[10], "close"] = 105.0

        # Long action should be allowed
        action = np.array([0.5])
        obs, _, _, _, info = env.step(action)

        assert env.position is not None or info.get("position") is not None

    def test_entry_gate_long_blocked(self, sample_bars: pd.DataFrame) -> None:
        """Test that long entry is blocked when conditions are not met."""
        # Create features with long conditions NOT met
        features = pd.DataFrame(index=sample_bars.index)
        features["london_high"] = 100.0
        features["london_low"] = 90.0
        features["asian_high"] = 98.0
        features["asian_low"] = 92.0
        features["volume_spike"] = 1.0  # < 1.5

        env = TradingEnv(
            sample_bars,
            features,
            continuous_actions=True,
            obs_window=10,
        )
        env.reset()

        # Set price below London High
        sample_bars.loc[sample_bars.index[10], "close"] = 95.0

        # Long action should be blocked (forced to hold)
        action = np.array([0.5])
        obs, _, _, _, _ = env.step(action)

        # Position should be None (no position opened)
        assert env.position is None

    def test_entry_gate_short_allowed(self, sample_bars: pd.DataFrame) -> None:
        """Test that short entry is allowed when conditions are met."""
        # Create features with short conditions met
        features = pd.DataFrame(index=sample_bars.index)
        features["london_high"] = 100.0
        features["london_low"] = 90.0
        features["asian_high"] = 98.0
        features["asian_low"] = 92.0
        features["volume_spike"] = 2.0  # > 1.5

        env = TradingEnv(
            sample_bars,
            features,
            continuous_actions=True,
            obs_window=10,
        )
        env.reset()

        # Set price below London Low
        sample_bars.loc[sample_bars.index[10], "close"] = 85.0

        # Short action should be allowed
        action = np.array([-0.5])
        obs, _, _, _, info = env.step(action)

        assert env.position is not None or info.get("position") is not None

    def test_entry_gate_short_blocked(self, sample_bars: pd.DataFrame) -> None:
        """Test that short entry is blocked when conditions are not met."""
        # Create features with short conditions NOT met
        features = pd.DataFrame(index=sample_bars.index)
        features["london_high"] = 100.0
        features["london_low"] = 90.0
        features["asian_high"] = 98.0
        features["asian_low"] = 92.0
        features["volume_spike"] = 1.0  # < 1.5

        env = TradingEnv(
            sample_bars,
            features,
            continuous_actions=True,
            obs_window=10,
        )
        env.reset()

        # Set price above London Low
        sample_bars.loc[sample_bars.index[10], "close"] = 95.0

        # Short action should be blocked (forced to hold)
        action = np.array([-0.5])
        obs, _, _, _, _ = env.step(action)

        # Position should be None (no position opened)
        assert env.position is None

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

        assert env.position is None or info.get("position") is None
