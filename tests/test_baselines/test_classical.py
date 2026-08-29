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
