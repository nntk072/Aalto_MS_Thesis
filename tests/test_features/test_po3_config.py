"""Tests for PO3/IFVG detection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.po3_config import (
    detect_po3_entries,
    detect_ltf_ifvg,
    EntryConfig,
    FVGConfig,
    IFVGConfig,
    detect_entry_trigger,
    detect_fvg,
    detect_htf_fvg,
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
    volume = pd.Series(np.random.randint(100, 1000, size=100), index=dates)
    
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
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


class TestDetectHTFFVG:
    """Tests for HTF FVG detection."""

    def test_htf_fvg_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that HTF FVG detection returns expected columns."""
        result = detect_htf_fvg(sample_bars, htf='M5')
        assert "htf_fvg_bullish" in result.columns
        assert "htf_fvg_bearish" in result.columns
        assert "htf_fvg_bullish_low" in result.columns
        assert "htf_fvg_bearish_high" in result.columns

    def test_htf_fvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as M1 input."""
        result = detect_htf_fvg(sample_bars, htf='M5')
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_htf_fvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that HTF FVG signals are binary (0 or 1)."""
        result = detect_htf_fvg(sample_bars, htf='M5')
        assert set(result["htf_fvg_bullish"].unique()).issubset({0, 1})
        assert set(result["htf_fvg_bearish"].unique()).issubset({0, 1})

    def test_htf_fvg_with_custom_config(self, sample_bars: pd.DataFrame) -> None:
        """Test HTF FVG detection with custom configuration."""
        config = FVGConfig(min_imbalance_pts=0.5)
        result = detect_htf_fvg(sample_bars, htf='M15', config=config)
        assert isinstance(result, pd.DataFrame)

    def test_htf_fvg_with_different_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test HTF FVG detection with different timeframes."""
        for tf in ['M5', 'M15', 'M30']:
            result = detect_htf_fvg(sample_bars, htf=tf)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_bars)


class TestDetectLTFIFVG:
    """Tests for LTF IFVG confirmation detection."""

    def test_ltf_ifvg_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that LTF IFVG detection returns expected columns."""
        result = detect_ltf_ifvg(sample_bars, primary_tf='M5')
        assert "ltf_ifvg_bullish_confirmed" in result.columns
        assert "ltf_ifvg_bearish_confirmed" in result.columns
        assert "ltf_ifvg_bullish_low" in result.columns
        assert "ltf_ifvg_bearish_high" in result.columns

    def test_ltf_ifvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as M1 input."""
        result = detect_ltf_ifvg(sample_bars, primary_tf='M5')
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_ltf_ifvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that LTF IFVG signals are binary (0 or 1)."""
        result = detect_ltf_ifvg(sample_bars, primary_tf='M5')
        assert set(result["ltf_ifvg_bullish_confirmed"].unique()).issubset({0, 1})
        assert set(result["ltf_ifvg_bearish_confirmed"].unique()).issubset({0, 1})

    def test_ltf_ifvg_with_custom_configs(self, sample_bars: pd.DataFrame) -> None:
        """Test LTF IFVG detection with custom configurations."""
        fvg_config = FVGConfig(min_imbalance_pts=0.5)
        ifvg_config = IFVGConfig(close_through_threshold=0.3)
        result = detect_ltf_ifvg(sample_bars, primary_tf='M15', fvg_config=fvg_config, ifvg_config=ifvg_config)
        assert isinstance(result, pd.DataFrame)

    def test_ltf_ifvg_with_different_primary_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test LTF IFVG detection with different primary timeframes."""
        for tf in ['M5', 'M15', 'M30']:
            result = detect_ltf_ifvg(sample_bars, primary_tf=tf)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_bars)


class TestDetectPO3Entries:
    """Tests for unified PO3 entry detection."""

    def test_po3_entries_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that unified PO3 entry detection returns expected columns."""
        result = detect_po3_entries(sample_bars)
        # Check HTF FVG columns
        assert "htf_fvg_bullish" in result.columns
        assert "htf_fvg_bearish" in result.columns
        # Check LTF IFVG columns
        assert "ltf_ifvg_bullish_confirmed" in result.columns
        assert "ltf_ifvg_bearish_confirmed" in result.columns
        # Check entry columns
        assert "entry_long" in result.columns
        assert "entry_short" in result.columns
        assert "entry_trigger_type" in result.columns

    def test_po3_entries_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as M1 input."""
        result = detect_po3_entries(sample_bars)
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_po3_entries_with_different_entry_types(self, sample_bars: pd.DataFrame) -> None:
        """Test unified PO3 entry detection with different entry types."""
        for entry_type in ['retest', 'close_through']:
            result = detect_po3_entries(sample_bars, entry_type=entry_type)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_bars)

    def test_po3_entries_with_custom_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test unified PO3 entry detection with custom timeframes."""
        result = detect_po3_entries(sample_bars, htf='M30', primary_tf='M15')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_bars)

    def test_po3_entries_with_custom_configs(self, sample_bars: pd.DataFrame) -> None:
        """Test unified PO3 entry detection with custom configurations."""
        fvg_config = FVGConfig(min_imbalance_pts=0.5)
        ifvg_config = IFVGConfig(close_through_threshold=0.3)
        entry_config = EntryConfig(retest_threshold=0.05)
        result = detect_po3_entries(
            sample_bars,
            fvg_config=fvg_config,
            ifvg_config=ifvg_config,
            entry_config=entry_config,
        )
        assert isinstance(result, pd.DataFrame)
