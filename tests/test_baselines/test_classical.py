"""Tests for classical baseline strategies on synthetic bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.baselines import (
    BaseStrategy,
    BuyAndHoldStrategy,
    EMAMACDRSIStrategy,
    MultiLevelBreakoutStrategy,
    SessionMomentumStrategy,
)


def _make_bars(n: int = 60, trend: float = 0.5) -> pd.DataFrame:
    """Synthetic uptrending OHLCV bars with volume and a DatetimeIndex."""
    rng = np.random.default_rng(42)
    close = 20_000.0 + np.cumsum(rng.normal(trend, 2.0, n))
    index = pd.date_range("2025-01-02 16:30", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 1.0, n),
            "high": close + rng.uniform(1.0, 3.0, n),
            "low": close - rng.uniform(1.0, 3.0, n),
            "close": close,
            "volume": rng.uniform(900, 1_100, n),
        },
        index=index,
    )


@pytest.mark.unit
class TestBuyAndHold:
    def test_emits_constant_long_fraction(self) -> None:
        # Arrange
        strategy = BuyAndHoldStrategy(n_bars=5)

        # Act
        actions = [strategy.act({}) for _ in range(5)]

        # Assert
        assert actions == [1.0] * 5

    def test_size_is_clamped_to_unit_range(self) -> None:
        # Act
        strategy = BuyAndHoldStrategy(n_bars=1, size=5.0)

        # Assert
        assert strategy.act({}) == 1.0

    def test_intraday_enters_once_per_session(self) -> None:
        idx = pd.DatetimeIndex(
            list(pd.date_range("2025-01-06 16:30", periods=3, freq="1min", tz="Etc/GMT-3"))
            + list(pd.date_range("2025-01-07 16:30", periods=3, freq="1min", tz="Etc/GMT-3"))
        )
        bars = pd.DataFrame({"close": 100.0}, index=idx)
        strategy = BuyAndHoldStrategy(bars=bars, mode="intraday")
        actions = [strategy.act({}) for _ in range(len(bars))]
        assert actions == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def test_swing_enters_once(self) -> None:
        idx = pd.date_range("2025-01-06 16:30", periods=4, freq="1min", tz="Etc/GMT-3")
        bars = pd.DataFrame({"close": 100.0}, index=idx)
        strategy = BuyAndHoldStrategy(bars=bars, mode="swing")
        actions = [strategy.act({}) for _ in range(len(bars))]
        assert actions == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
class TestEMAMACDRSIStrategy:
    def test_actions_within_unit_bounds(self) -> None:
        # Arrange
        strategy = EMAMACDRSIStrategy(_make_bars())

        # Act
        actions = [strategy.act({}) for _ in range(50)]

        # Assert
        assert all(-1.0 <= a <= 1.0 for a in actions)
        assert any(a != 0.0 for a in actions)

    def test_is_a_base_strategy(self) -> None:
        assert isinstance(EMAMACDRSIStrategy(_make_bars()), BaseStrategy)


@pytest.mark.unit
class TestSessionMomentumStrategy:
    def test_second_session_takes_sign_of_the_first(self) -> None:
        # Day 1 rises, so day 2 is long. Day 2 falls, so day 3 is short.
        idx = pd.DatetimeIndex(
            list(pd.date_range("2025-01-06 16:30", periods=3, freq="1min", tz="Etc/GMT-3"))
            + list(pd.date_range("2025-01-07 16:30", periods=3, freq="1min", tz="Etc/GMT-3"))
            + list(pd.date_range("2025-01-08 16:30", periods=3, freq="1min", tz="Etc/GMT-3"))
        )
        close = [100.0, 101.0, 110.0, 110.0, 105.0, 100.0, 100.0, 100.0, 100.0]
        bars = pd.DataFrame({"close": close}, index=idx)
        strategy = SessionMomentumStrategy(bars)
        actions = [strategy.act({}) for _ in range(len(bars))]
        assert actions[:3] == [0.0, 0.0, 0.0]
        assert actions[3:6] == [1.0, 1.0, 1.0]
        assert actions[6:9] == [-1.0, -1.0, -1.0]


@pytest.mark.unit
class TestMultiLevelBreakout:
    def _bars_with_levels(self, n: int = 40) -> pd.DataFrame:
        bars = _make_bars(n)
        # Force one strong breakout bar above the Asian high
        bars.loc[bars.index[-3], "close"] = bars["close"].iloc[-4] + 100.0
        return bars

    def test_actions_within_unit_bounds_and_finite(self) -> None:
        # Arrange
        strategy = MultiLevelBreakoutStrategy(self._bars_with_levels())

        # Act
        actions = [strategy.act({}) for _ in range(30)]

        # Assert
        assert all(np.isfinite(a) and -1.0 <= a <= 1.0 for a in actions)

    def test_no_volume_column_still_runs(self) -> None:
        # Arrange
        bars = self._bars_with_levels().drop(columns=["volume"])
        strategy = MultiLevelBreakoutStrategy(bars)

        # Act — spike treated as always satisfied
        actions = [strategy.act({}) for _ in range(10)]

        # Assert
        assert len(actions) == 10

    def test_reset_replays_same_signals(self) -> None:
        # Arrange
        strategy = MultiLevelBreakoutStrategy(self._bars_with_levels())
        first_pass = [strategy.act({}) for _ in range(20)]

        # Act
        strategy.reset()
        second_pass = [strategy.act({}) for _ in range(20)]

        # Assert
        assert first_pass == second_pass
