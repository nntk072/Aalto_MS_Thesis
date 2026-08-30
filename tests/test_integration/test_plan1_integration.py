"""Integration tests for PLAN 1."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_rl.features.build import build_features
from quant_rl.features.indicators import atr, volume_spike
from quant_rl.features.structure import detect_session_levels


class TestPlan1Integration:
    @pytest.fixture
    def full_day_data(self) -> pd.DataFrame:
        """Create sample full-day data with volume column."""
        dates = pd.date_range("2025-01-01 01:05", periods=500, freq="1min")
        data = {
            "open": [100 + i * 0.1 for i in range(500)],
            "high": [102 + i * 0.1 for i in range(500)],
            "low": [98 + i * 0.1 for i in range(500)],
            "close": [100 + i * 0.1 for i in range(500)],
            "volume": [1000 + i * 10 for i in range(500)],
        }
        return pd.DataFrame(data, index=dates)

    def test_liquidity_levels_in_features(self, full_day_data: pd.DataFrame) -> None:
        """Test that liquidity levels are present in feature pipeline."""
        features = build_features(full_day_data)
        assert "asian_high" in features.columns
        assert "asian_low" in features.columns
        assert "london_high" in features.columns
        assert "london_low" in features.columns
        assert "prev_day_close" in features.columns
        assert features["asian_high"].notna().any()
        assert features["volume_spike"].notna().any()
        assert features["atr_5"].notna().any()

    def test_volume_spike_in_features(self, full_day_data: pd.DataFrame) -> None:
        """Test that volume_spike is present in feature pipeline."""
        features = build_features(full_day_data)
        assert "volume_spike" in features.columns
        assert features["volume_spike"].notna().any()

    def test_atr_in_features(self, full_day_data: pd.DataFrame) -> None:
        """Test that atr_5 is present in feature pipeline."""
        features = build_features(full_day_data)
        assert "atr_5" in features.columns
        assert features["atr_5"].notna().any()

    def test_detect_session_levels_returns_expected_columns(
        self, full_day_data: pd.DataFrame
    ) -> None:
        """Test that detect_session_levels returns expected columns."""
        levels = detect_session_levels(full_day_data)
        assert "asian_high" in levels.columns
        assert "asian_low" in levels.columns
        assert "london_high" in levels.columns
        assert "london_low" in levels.columns
        assert "prev_day_close" in levels.columns

    def test_volume_spike_function(self) -> None:
        """Test volume_spike function directly."""
        volume = pd.Series([1000, 1100, 1200, 1500, 2000, 1800, 1600, 1400, 1300, 1200])
        spike = volume_spike(volume, window=3)
        assert len(spike) == len(volume)
        # Window [1800, 1600, 1400] -> median=1600, ratio=1400/1600=0.875
        assert spike.iloc[7] == pytest.approx(0.875, rel=1e-3)
        # Window [1200, 1500, 2000] -> median=1500, ratio=2000/1500=1.333
        assert spike.iloc[4] > 1.0

    def test_atr_function(self, full_day_data: pd.DataFrame) -> None:
        """Test atr function directly."""
        atr_series = atr(full_day_data, period=14)
        assert len(atr_series) == len(full_day_data)
        assert atr_series.notna().sum() > 0
