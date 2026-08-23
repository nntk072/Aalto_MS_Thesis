"""Tests for liquidity level detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.structure import detect_session_levels


@pytest.fixture
def asian_session_bars() -> pd.DataFrame:
    """Create sample bars for Asian session."""
    dates = pd.date_range("2025-01-01 01:05", periods=100, freq="5min")
    np.random.seed(42)
    high = np.cumsum(np.random.randn(100) * 0.1) + 100
    low = high - np.abs(np.random.randn(100) * 0.05)
    close = (high + low) / 2
    volume = np.random.randint(1000, 5000, 100)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def full_day_bars() -> pd.DataFrame:
    """Create sample bars for full day (Asian + London + NY)."""
    dates = pd.date_range("2025-01-01 01:05", periods=300, freq="5min")
    np.random.seed(42)
    high = np.cumsum(np.random.randn(300) * 0.1) + 100
    low = high - np.abs(np.random.randn(300) * 0.05)
    close = (high + low) / 2
    volume = np.random.randint(1000, 5000, 300)
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestDetectSessionLevels:
    def test_asian_levels_detected(self, asian_session_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(asian_session_bars)
        assert "asian_high" in levels.columns
        assert "asian_low" in levels.columns
        assert "london_high" in levels.columns
        assert "london_low" in levels.columns
        assert "prev_day_close" in levels.columns

    def test_london_levels_detected(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert "london_high" in levels.columns
        assert "london_low" in levels.columns

    def test_forward_fill_to_ny(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert "asian_high" in levels.columns
        assert "asian_low" in levels.columns

    def test_prev_day_close(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert "prev_day_close" in levels.columns
        assert levels["prev_day_close"].isna().sum() > 0  # First bar has no prev close

    def test_output_shape(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert isinstance(levels, pd.DataFrame)
        assert len(levels) == len(full_day_bars)
        assert levels.index.equals(full_day_bars.index)
