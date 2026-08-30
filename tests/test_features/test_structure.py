"""Tests for liquidity level detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.structure import detect_session_levels, get_session


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


class TestGetSession:
    """Tests for session tagging function."""

    def test_asian_session(self) -> None:
        """Test timestamps within Asian session."""
        # 01:05 to 08:59 should be Asia
        assert get_session("2025-01-01 01:05:00+03:00") == "asia"
        assert get_session("2025-01-01 05:30:00+03:00") == "asia"
        assert get_session("2025-01-01 08:59:00+03:00") == "asia"

    def test_london_session(self) -> None:
        """Test timestamps within London session."""
        # 09:00 to 16:29 should be London
        assert get_session("2025-01-01 09:00:00+03:00") == "london"
        assert get_session("2025-01-01 12:30:00+03:00") == "london"
        assert get_session("2025-01-01 16:29:00+03:00") == "london"

    def test_ny_session(self) -> None:
        """Test timestamps within NY session."""
        # 16:30 to 23:49 should be NY
        assert get_session("2025-01-01 16:30:00+03:00") == "ny"
        assert get_session("2025-01-01 20:00:00+03:00") == "ny"
        assert get_session("2025-01-01 23:49:00+03:00") == "ny"

    def test_ny_overnight(self) -> None:
        """Test overnight hours (23:50-01:05) count as NY."""
        assert get_session("2025-01-01 23:50:00+03:00") == "ny"
        assert get_session("2025-01-01 00:00:00+03:00") == "ny"
        assert get_session("2025-01-01 01:00:00+03:00") == "ny"

    def test_naive_timestamp(self) -> None:
        """Test that naive timestamps are localized to default tz."""
        ts = pd.Timestamp("2025-01-01 10:00:00")  # naive
        assert get_session(ts) == "london"

    def test_different_timezones(self) -> None:
        """Test conversion from different timezones."""
        # 10:00 UTC = 13:00 UTC+3 (London)
        ts_utc = pd.Timestamp("2025-01-01 10:00:00+00:00")
        assert get_session(ts_utc) == "london"

    def test_string_input(self) -> None:
        """Test string timestamp input."""
        assert get_session("2025-01-01 14:00:00+03:00") == "london"
        assert get_session("2025-01-01 18:00:00+03:00") == "ny"
