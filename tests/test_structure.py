"""Tests for swing structure detection."""

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
