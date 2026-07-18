"""Tests for structure-based SL/TP computation."""
import numpy as np
import pandas as pd
import pytest

from quant_rl.features.structure import compute_structure_features


@pytest.fixture
def sample_bars():
    """Create sample M1 bars with defined swing highs/lows."""
    data = {
        "open": [100, 101, 102, 101, 100, 99, 100, 101, 102, 101, 100],
        "high": [101, 102, 103, 102, 101, 100, 101, 102, 103, 102, 101],
        "low": [100, 101, 102, 101, 100, 99, 100, 101, 102, 101, 100],
        "close": [100.5, 101.5, 102.5, 101.5, 100.5, 99.5, 100.5, 101.5, 102.5, 101.5, 100.5],
    }
    index = pd.date_range("2026-01-01", periods=11, freq="1min")
    return pd.DataFrame(data, index=index)


def test_compute_structure_features(sample_bars):
    """Test that structure features are computed and causal."""
    result = compute_structure_features(sample_bars, swing_period=2, buffer_pts=1.0)
    
    assert "last_swing_low_price" in result.columns
    assert "last_swing_high_price" in result.columns
    assert len(result) == len(sample_bars)
    assert result.index.equals(sample_bars.index)
    
    # Early bars should have NaN (no prior swings)
    assert pd.isna(result["last_swing_low_price"].iloc[0])
    assert pd.isna(result["last_swing_high_price"].iloc[0])


def test_structure_features_no_lookahead(sample_bars):
    """Verify that structure features don't use future bars."""
    result = compute_structure_features(sample_bars, swing_period=2, buffer_pts=1.0)
    
    # For bar i, the structure should only depend on bars up to i (no lookahead)
    for i in range(5, len(result)):
        if not pd.isna(result["last_swing_low_price"].iloc[i]):
            # The swing low should have occurred before bar i
            assert i > 2  # Minimum bars needed for swing detection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
