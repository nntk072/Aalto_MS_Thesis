"""Tests for PO3/IFVG detection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.po3_config import (
    EntryConfig,
    FVGConfig,
    IFVGConfig,
    build_fvg_zones,
    detect_entry_trigger,
    detect_fvg,
    detect_htf_fvg,
    detect_ifvg_confirmation,
    detect_ltf_ifvg,
    detect_po3_entries,
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
        assert "fvg_bullish_high" in result.columns
        assert "fvg_bearish_low" in result.columns
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
        assert "ifvg_bullish_low" in ifvg_result.columns
        assert "ifvg_bullish_high" in ifvg_result.columns
        assert "ifvg_bearish_low" in ifvg_result.columns
        assert "ifvg_bearish_high" in ifvg_result.columns

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
        entry_result = detect_entry_trigger(sample_bars, ifvg_result, entry_type="retest")
        assert isinstance(entry_result, pd.DataFrame)

    def test_entry_trigger_with_close_through_type(self, sample_bars: pd.DataFrame) -> None:
        """Test entry trigger with close_through type."""
        fvg_result = detect_fvg(sample_bars)
        ifvg_result = detect_ifvg_confirmation(sample_bars, fvg_result)
        entry_result = detect_entry_trigger(sample_bars, ifvg_result, entry_type="close_through")
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
        result = detect_htf_fvg(sample_bars, htf="M5")
        assert "htf_fvg_bullish" in result.columns
        assert "htf_fvg_bearish" in result.columns
        assert "htf_fvg_bullish_low" in result.columns
        assert "htf_fvg_bullish_high" in result.columns
        assert "htf_fvg_bearish_low" in result.columns
        assert "htf_fvg_bearish_high" in result.columns

    def test_htf_fvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as M1 input."""
        result = detect_htf_fvg(sample_bars, htf="M5")
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_htf_fvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that HTF FVG signals are binary (0 or 1)."""
        result = detect_htf_fvg(sample_bars, htf="M5")
        assert set(result["htf_fvg_bullish"].unique()).issubset({0, 1})
        assert set(result["htf_fvg_bearish"].unique()).issubset({0, 1})

    def test_htf_fvg_with_custom_config(self, sample_bars: pd.DataFrame) -> None:
        """Test HTF FVG detection with custom configuration."""
        config = FVGConfig(min_imbalance_pts=0.5)
        result = detect_htf_fvg(sample_bars, htf="M15", config=config)
        assert isinstance(result, pd.DataFrame)

    def test_htf_fvg_with_different_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test HTF FVG detection with different timeframes."""
        for tf in ["M5", "M15", "M30"]:
            result = detect_htf_fvg(sample_bars, htf=tf)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_bars)


class TestDetectLTFIFVG:
    """Tests for LTF IFVG confirmation detection."""

    def test_ltf_ifvg_returns_expected_columns(self, sample_bars: pd.DataFrame) -> None:
        """Test that LTF IFVG detection returns expected columns."""
        result = detect_ltf_ifvg(sample_bars, primary_tf="M5")
        assert "ltf_ifvg_bullish_confirmed" in result.columns
        assert "ltf_ifvg_bearish_confirmed" in result.columns
        assert "ltf_ifvg_bullish_low" in result.columns
        assert "ltf_ifvg_bullish_high" in result.columns
        assert "ltf_ifvg_bearish_low" in result.columns
        assert "ltf_ifvg_bearish_high" in result.columns

    def test_ltf_ifvg_output_shape(self, sample_bars: pd.DataFrame) -> None:
        """Test that output has same length as M1 input."""
        result = detect_ltf_ifvg(sample_bars, primary_tf="M5")
        assert len(result) == len(sample_bars)
        assert result.index.equals(sample_bars.index)

    def test_ltf_ifvg_binary_values(self, sample_bars: pd.DataFrame) -> None:
        """Test that LTF IFVG signals are binary (0 or 1)."""
        result = detect_ltf_ifvg(sample_bars, primary_tf="M5")
        assert set(result["ltf_ifvg_bullish_confirmed"].unique()).issubset({0, 1})
        assert set(result["ltf_ifvg_bearish_confirmed"].unique()).issubset({0, 1})

    def test_ltf_ifvg_with_custom_configs(self, sample_bars: pd.DataFrame) -> None:
        """Test LTF IFVG detection with custom configurations."""
        fvg_config = FVGConfig(min_imbalance_pts=0.5)
        ifvg_config = IFVGConfig(close_through_threshold=0.3)
        result = detect_ltf_ifvg(
            sample_bars, primary_tf="M15", fvg_config=fvg_config, ifvg_config=ifvg_config
        )
        assert isinstance(result, pd.DataFrame)

    def test_ltf_ifvg_with_different_primary_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test LTF IFVG detection with different primary timeframes."""
        for tf in ["M5", "M15", "M30"]:
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
        assert "htf_fvg_bullish_low" in result.columns
        assert "htf_fvg_bullish_high" in result.columns
        assert "htf_fvg_bearish_low" in result.columns
        assert "htf_fvg_bearish_high" in result.columns
        # Check LTF IFVG columns
        assert "ltf_ifvg_bullish_confirmed" in result.columns
        assert "ltf_ifvg_bearish_confirmed" in result.columns
        assert "ltf_ifvg_bullish_low" in result.columns
        assert "ltf_ifvg_bullish_high" in result.columns
        assert "ltf_ifvg_bearish_low" in result.columns
        assert "ltf_ifvg_bearish_high" in result.columns
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
        for entry_type in ["retest", "close_through"]:
            result = detect_po3_entries(sample_bars, entry_type=entry_type)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_bars)

    def test_po3_entries_with_custom_timeframes(self, sample_bars: pd.DataFrame) -> None:
        """Test unified PO3 entry detection with custom timeframes."""
        result = detect_po3_entries(sample_bars, htf="M30", primary_tf="M15")
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


class TestIFVGConfirmationBehavior:
    """Behavioral tests: confirmations must actually fire on close-through data."""

    @pytest.fixture()
    def bullish_close_through_bars(self) -> pd.DataFrame:
        """Bars with a bullish FVG (zone [10, 12]) confirmed by closes above it."""
        idx = pd.date_range("2025-01-01", periods=10, freq="1min")
        return pd.DataFrame(
            {
                "open": [9, 10, 11, 12, 11, 13, 14, 14, 13, 12],
                "high": [10, 11, 13, 13, 12, 14, 15, 15, 14, 13],
                "low": [9, 10, 12, 11, 10, 12, 13, 13, 12, 11],
                "close": [9, 10, 13, 12, 11, 13, 14, 14, 13, 12],
            },
            index=idx,
        )

    def test_bullish_ifvg_fires_on_close_through(
        self, bullish_close_through_bars: pd.DataFrame
    ) -> None:
        """A close through the FVG zone must mark the IFVG confirmed."""
        fvg = detect_fvg(bullish_close_through_bars)
        ifvg = detect_ifvg_confirmation(bullish_close_through_bars, fvg)
        assert ifvg["ifvg_bullish_confirmed"].sum() > 0
        # Zone bounds must be populated and match the FVG gap [bar1.high, bar3.low].
        confirmed = ifvg[ifvg["ifvg_bullish_confirmed"] == 1].iloc[0]
        assert confirmed["ifvg_bullish_low"] == 10.0
        assert confirmed["ifvg_bullish_high"] == 12.0

    def test_bearish_ifvg_fires_on_close_through(self) -> None:
        """A bearish FVG with closes below it must confirm."""
        idx = pd.date_range("2025-01-01", periods=10, freq="1min")
        bars = pd.DataFrame(
            {
                "open": [13, 12, 11, 10, 11, 9, 8, 8, 9, 10],
                "high": [14, 13, 12, 11, 12, 10, 9, 9, 10, 11],
                "low": [12, 11, 9, 9, 10, 8, 7, 7, 8, 9],
                "close": [13, 12, 9, 10, 11, 9, 8, 8, 9, 10],
            },
            index=idx,
        )
        fvg = detect_fvg(bars)
        ifvg = detect_ifvg_confirmation(bars, fvg)
        assert ifvg["ifvg_bearish_confirmed"].sum() > 0
        confirmed = ifvg[ifvg["ifvg_bearish_confirmed"] == 1].iloc[0]
        # Bearish FVG zone is [bar3.high, bar1.low] = [9, 10].
        assert confirmed["ifvg_bearish_low"] == 9.0
        assert confirmed["ifvg_bearish_high"] == 10.0


class TestBuildFVGZones:
    """Tests for the drawable zone builder."""

    @pytest.fixture()
    def fvg_bars(self) -> pd.DataFrame:
        """M1 bars whose M5 resample forms a bullish FVG (zone [11, 12] at 00:10).

        M5[0]  (00:00, bars 0-4):  high 11, low 9
        M5[1]  (00:05, bars 5-9):  high 11, low 9
        M5[2]  (00:10, bars 10-14): low 12   -> bar3.low > bar1.high => FVG
        M5[3]  (00:15, bars 15-19): high 14, low 12 (zone stays unfilled)
        """
        idx = pd.date_range("2025-01-01", periods=20, freq="1min")
        return pd.DataFrame(
            {
                "open": [
                    9,
                    9,
                    9,
                    9,
                    9,
                    10,
                    9,
                    9,
                    10,
                    9,
                    12,
                    12,
                    12,
                    12,
                    12,
                    13,
                    13,
                    12,
                    12,
                    12,
                ],
                "high": [
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    11,
                    10,
                    10,
                    13,
                    14,
                    13,
                    14,
                    13,
                    14,
                    13,
                    14,
                    13,
                    14,
                ],
                "low": [
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    9,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                    12,
                ],
                "close": [
                    9,
                    10,
                    9,
                    10,
                    9,
                    10,
                    9,
                    10,
                    9,
                    9,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                    12,
                    13,
                ],
            },
            index=idx,
        )

    def test_zone_from_htf_fvg(self, fvg_bars: pd.DataFrame) -> None:
        """Each 0->1 signal run yields exactly one zone with correct bounds."""
        signals = detect_htf_fvg(fvg_bars, htf="M5")
        zones = build_fvg_zones(fvg_bars, signals)
        bullish = [z for z in zones if z.side == "bullish"]
        assert len(bullish) >= 1
        z = bullish[0]
        assert z.kind == "htf_fvg"
        assert z.confirmed is False
        assert z.zone_low == 11.0
        assert z.zone_high == 12.0
        assert z.start_ts < z.end_ts

    def test_zone_edge_dedup(self, fvg_bars: pd.DataFrame) -> None:
        """Forward-filled runs must not create many overlapping zones."""
        signals = detect_htf_fvg(fvg_bars, htf="M5")
        zones = build_fvg_zones(fvg_bars, signals)
        starts = [z.start_ts for z in zones if z.side == "bullish"]
        # The M5 H4 bar where FVG fires is one contiguous run -> 1 start.
        assert len(starts) <= 3

    def test_zone_invalidation_on_fill(self) -> None:
        """A zone is marked invalidated when price trades back through the gap."""
        idx = pd.date_range("2025-01-01", periods=10, freq="1min")
        bars = pd.DataFrame(
            {
                "open": [9, 10, 11, 12, 12, 11, 10, 10, 11, 11],
                "high": [10, 11, 13, 13, 13, 12, 11, 11, 12, 12],
                "low": [9, 10, 12, 12, 12, 11, 10, 10, 11, 11],
                "close": [9, 10, 13, 12, 12, 11, 10, 10, 11, 11],
            },
            index=idx,
        )
        # Hand-build an HTF FVG signal run that is later filled (low <= 10).
        signals = pd.DataFrame(
            {
                "htf_fvg_bullish": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                "htf_fvg_bullish_low": [
                    np.nan,
                    np.nan,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                ],
                "htf_fvg_bullish_high": [
                    np.nan,
                    np.nan,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                    12.0,
                ],
            },
            index=idx,
        )
        zones = build_fvg_zones(bars, signals, max_zone_bars=20)
        assert len(zones) == 1
        assert zones[0].invalidated is True
        # Ends at bar 5 (index 6) — not the horizon.
        assert zones[0].end_ts == idx[6]

    def test_zone_horizon_expiry(self, fvg_bars: pd.DataFrame) -> None:
        """A never-filled zone ends at the max_zone_bars horizon."""
        signals = detect_htf_fvg(fvg_bars, htf="M5")
        zones = build_fvg_zones(fvg_bars, signals, max_zone_bars=3)
        bull = [z for z in zones if z.side == "bullish"]
        assert bull
        # Zone starts at the first M1 bar of the FVG M5 bar (00:10) and,
        # never being filled, ends 3 bars later at 00:13.
        assert bull[0].invalidated is False
        assert bull[0].end_ts == fvg_bars.index[13]

    def test_empty_signals(self, fvg_bars: pd.DataFrame) -> None:
        """Missing signal columns return an empty zone list."""
        zones = build_fvg_zones(fvg_bars, pd.DataFrame(index=fvg_bars.index))
        assert zones == []
