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
        assert "prev_day_high" in levels.columns
        assert "prev_day_low" in levels.columns

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

    def test_prev_day_high_low(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert "prev_day_high" in levels.columns
        assert "prev_day_low" in levels.columns
        idx_dt = pd.DatetimeIndex(full_day_bars.index)
        # First day bars should have NaN prev_day_high/low (no prior completed day)
        first_day_mask = idx_dt.date == idx_dt[0].date()
        assert levels.loc[first_day_mask, "prev_day_high"].isna().all()
        assert levels.loc[first_day_mask, "prev_day_low"].isna().all()
        # Second day bars should have finite values from day 1
        second_day = idx_dt[288].date()
        second_day_mask = idx_dt.date == second_day
        assert levels.loc[second_day_mask, "prev_day_high"].notna().any()
        assert levels.loc[second_day_mask, "prev_day_low"].notna().any()

    def test_output_shape(self, full_day_bars: pd.DataFrame) -> None:
        levels = detect_session_levels(full_day_bars)
        assert isinstance(levels, pd.DataFrame)
        assert len(levels) == len(full_day_bars)
        assert levels.index.equals(full_day_bars.index)


def test_detect_session_levels_on_pipeline_output() -> None:
    idx = pd.date_range("2025-01-06 01:05", periods=22 * 60, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.arange(len(idx), dtype=float) * 0.01
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )
    levels = detect_session_levels(df)
    ny = idx[idx.strftime("%H:%M") == "16:30"]
    assert len(ny)
    assert pd.notna(levels.loc[ny[0], "asian_high"])
    assert pd.notna(levels.loc[ny[0], "london_high"])


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
        assert get_session("2025-01-01 16:30:00+03:00") == "ny"
        assert get_session("2025-01-01 20:00:00+03:00") == "ny"
        assert get_session("2025-01-01 23:00:00+03:00") == "ny"

    def test_get_session_closed_label(self) -> None:
        """Broker gap is closed, not ny."""
        assert get_session("2025-01-01 23:01:00+03:00") == "closed"
        assert get_session("2025-01-01 23:50:00+03:00") == "closed"
        assert get_session("2025-01-01 00:00:00+03:00") == "closed"
        assert get_session("2025-01-01 01:00:00+03:00") == "closed"

    def test_ny_overnight(self) -> None:
        """Overnight hours (23:01-01:04) are closed."""
        assert get_session("2025-01-01 23:50:00+03:00") == "closed"
        assert get_session("2025-01-01 00:00:00+03:00") == "closed"
        assert get_session("2025-01-01 01:00:00+03:00") == "closed"

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
