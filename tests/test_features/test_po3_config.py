"""Tests for PO3/IFVG detection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.po3_config import (
    EntryConfig,
    FVGConfig,
    IFVGConfig,
    detect_entry_trigger,
    detect_fvg,
    detect_ifvg_confirmation,
)


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Create sample OHLC bars for testing."""
    dates = pd.date_range("2025-01-01", periods=100, freq="min")
    np.random.seed(42)
    
    # Create some price movement
    close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.1), index=dates)
    high = close + np.abs(np.random.randn(100) * 0.05)
    low = close - np.abs(np.random.randn(100) * 0.05)
    open_price = close.shift(1).fillna(close.iloc[0])
    
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close},
        index=dates,
    )


class TestDetectFVG:
    """Tests for FVG detection."""

    def test_detect_fvg_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that FVG detection returns expected columns."""
        result = detect_fvg(sample_bars)
        assert "fvg_bullish" in result.columns
        assert "fvg_bearish" in result.columns
        assert "fvg_bullish_low" in result.columns
        assert "fvg_bearish_high" in result.columns

    def test_fvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as input."""
        result = detect_fvg(sample_bars)
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_fvg_with_custom_config(self, sample_bars: pd.DataFrame) -> None:
        """Test FVG detection with custom minimum imbalance."""
        config = FVGConfig(min_imbalance_pts=0.5)
        result = detect_fvg(sample_bars, config=config)
        assert isinstance(result, pd.DataFrame)

    def test_fvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that FVG signals are binary (0 or 1)."""
        result = detect_fvg(sample_bars)
        assert set(result["fvg_bullish"].unique()).issubset({0, 1})
        assert set(result["fvg_bearish"].unique()).issubset({0, 1})


class TestDetectIFVGConfirmation:
    """Tests for IFVG confirmation detection."""

    def test_ifvg_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that IFVG confirmation returns expected columns."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        assert "ifvg_bullish_confirmed" in ifvg_result.columns
        assert "ifvg_bearish_confirmed" in ifvg_result.columns

    def test_ifvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as input."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        assert len(ifvg_result) == len(sample_bars)

    def test_ifvg_with_custom_config(self, sample_bars: pd.DataFrame) -> None:
        """Test IFVG confirmation with custom threshold."""
        fvg_result = detect_fvg(sample_bars)
        config = IFVGConfig(close_through_threshold=0.3)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result, config=config)
        assert isinstance(ifvg_result, pd.DataFrame)

    def test_ifvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that IFVG signals are binary (0 or 1)."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        assert set(ifvg_result["ifvg_bullish_confirmed"].unique()).issubset({0, 1})
        assert set(ifvg_result["ifvg_bearish_confirmed"].unique()).issubset({0, 1})


class TestDetectEntryTrigger:
    """Tests for entry trigger detection."""

    def test_entry_trigger_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that entry trigger returns expected columns."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result)
        assert "entry_long" in entry_result.columns
        assert "entry_short" in entry_result.columns
        assert "entry_trigger_type" in entry_result.columns

    def test_entry_trigger_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as input."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result)
        assert len(entry_result) == len(sample_bars)

    def test_entry_trigger_with_retest_type(self, sample_bars: pd.DataFrame) -> None:
        """Test entry trigger with retest type."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result, entry_type='retest')
        assert isinstance(entry_result, pd.DataFrame)

    def test_entry_trigger_with_close_through_type(self, sample_bars: pd.DataFrame) -> None:
        """Test entry trigger with close_through type."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result, entry_type='close_through')
        assert isinstance(entry_result, pd.DataFrame)

    def test_entry_trigger_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that entry signals are binary (0 or 1)."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result)
        assert set(entry_result["entry_long"].unique()).issubset({0, 1})
        assert set(entry_result["entry_short"].unique()).issubset({0, 1})

    def test_entry_trigger_with_custom_config(self, sample_bars: pd.DataFrame) -> None:
        """Test entry trigger with custom configuration."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        config = EntryConfig(retest_threshold=0.05)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result, config=config)
        assert isinstance(entry_result, pd.DataFrame)
