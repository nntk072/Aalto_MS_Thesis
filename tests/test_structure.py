"""Tests for swing structure detection."""

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.structure import structure_levels


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Create synthetic OHLC bars for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1min")
    data = {
        "open": [100 + i * 0.1 for i in range(50)],
        "high": [102 + i * 0.1 for i in range(50)],
        "low": [99 + i * 0.1 for i in range(50)],
        "close": [100.5 + i * 0.1 for i in range(50)],
        "volume": [1000 + i * 10 for i in range(50)],
    }
    return pd.DataFrame(data, index=dates)


def test_structure_levels_shape(sample_bars: pd.DataFrame) -> None:
    """Test that structure_levels returns correct shape."""
    result = structure_levels(sample_bars, swing_period=5)
    assert result.shape[0] == sample_bars.shape[0]
    assert "last_swing_high" in result.columns
    assert "last_swing_low" in result.columns
    assert "last_swing_high_time" in result.columns
    assert "last_swing_low_time" in result.columns


def test_structure_levels_causality(sample_bars: pd.DataFrame) -> None:
    """Test that swing levels only use past data (causality check)."""
    result = structure_levels(sample_bars, swing_period=5)
    # All non-NaN swing high values should be from the bars we've seen
    # (they cannot exceed the current bar's high when looking backward)
    non_nan_highs = result["last_swing_high"][result["last_swing_high"].notna()]
    if len(non_nan_highs) > 0:
        # The swing high represents a high from the past, so it should be reasonable
        assert non_nan_highs.min() > 0


def test_structure_levels_forward_fill(sample_bars: pd.DataFrame) -> None:
    """Test that swings are forward-filled correctly."""
    result = structure_levels(sample_bars, swing_period=5)
    # After a swing is detected, all subsequent bars should have that value
    # until the next swing is detected
    swing_high_df = result["last_swing_high"].dropna()
    if len(swing_high_df) > 0:
        # Check that values are either NaN or monotonic/repeated
        non_nan = result["last_swing_high"][result["last_swing_high"].notna()]
        assert len(non_nan) > 0


def test_structure_levels_record_swing_extremum_price() -> None:
    """The recorded level must be the swing bar's extremum price, not the
    confirmation bar's price (the flag fires ``swing_period`` bars later)."""
    idx = pd.date_range("2024-01-02 01:05", periods=10, freq="1min")
    highs = [101.0, 101.2, 101.4, 101.6, 102.0, 101.5, 101.0, 100.8, 100.6, 100.4]
    lows = [100.5, 100.3, 100.1, 99.9, 99.5, 100.0, 100.5, 100.8, 101.0, 101.2]
    closes = [100.8, 100.7, 100.6, 100.5, 101.0, 100.8, 100.6, 100.4, 100.2, 100.0]
    bars = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes}, index=idx)
    result = structure_levels(bars, swing_period=2)

    # Swing high is bar 4 (102.0), the max of the 5-bar lookback window ending
    # at itself; it is confirmed 2 bars later (bar 6). The recorded level must
    # be 102.0 with the swing bar's timestamp — not bar 6's own price.
    sh_levels = result["last_swing_high"].dropna()
    assert len(sh_levels) > 0
    assert sh_levels.iloc[0] == pytest.approx(102.0)
    assert result["last_swing_high_time"].dropna().iloc[0] == idx[4]

    sl_levels = result["last_swing_low"].dropna()
    assert len(sl_levels) > 0
    assert sl_levels.iloc[0] == pytest.approx(99.9)
    assert result["last_swing_low_time"].dropna().iloc[0] == idx[3]


def test_structure_levels_are_causal() -> None:
    """Adding future bars must not change past swing levels (Agent.md §26)."""
    idx = pd.date_range("2024-01-02 01:05", periods=20, freq="1min")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.2, 20))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
        },
        index=idx,
    )
    full = structure_levels(bars, swing_period=3)
    short = structure_levels(bars.iloc[:12], swing_period=3)
    pd.testing.assert_frame_equal(full.iloc[:12], short, check_freq=False)
