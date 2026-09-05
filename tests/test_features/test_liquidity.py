"""Tests for causal liquidity sweep and BOS detection (Agent.md §5, §19, §25)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.liquidity import detect_bos, detect_liquidity_sweeps
from quant_rl.features.structure import structure_levels


def make_bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (high, low, close) triples on a 1-min index."""
    idx = pd.date_range("2024-01-02 01:05", periods=len(rows), freq="1min")
    high = [r[0] for r in rows]
    low = [r[1] for r in rows]
    close = [r[2] for r in rows]
    open_ = [close[0]] + close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# (high, low, close) path: rise to 102.2, V-dip to 99.0, recover.
V_PATH = [
    (100.5, 100.0, 100.2),
    (100.8, 100.2, 100.6),
    (101.5, 100.6, 101.2),
    (101.8, 101.0, 101.5),
    (102.0, 101.2, 101.8),
    (102.2, 101.5, 102.0),
    (101.8, 101.0, 101.2),
    (100.8, 100.0, 100.2),
    (99.8, 99.0, 99.2),
    (100.5, 99.5, 100.3),
    (101.2, 100.5, 101.0),
    (101.6, 101.0, 101.4),
    (101.8, 101.2, 101.5),
    (102.0, 101.4, 101.8),
    (102.2, 101.6, 102.0),
]


class TestSweepLow:
    def test_sweep_low_detection(self) -> None:
        bars = make_bars(V_PATH)
        # Sweep bar: dips below confirmed swing low 99.0, closes back above it.
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 101.8, "low": 98.5, "close": 100.0}
        feats = detect_liquidity_sweeps(bars, swing_period=2)

        assert feats["sweep_low"].iloc[12] == 1
        assert feats["sweep_low_level"].iloc[12] == pytest.approx(99.0)
        assert feats["sweep_low_reclaimed"].iloc[12] == 1
        assert feats["sweep_low_age"].iloc[12] == 0

    def test_no_sweep_before_confirmation(self) -> None:
        bars = make_bars(V_PATH)
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 101.8, "low": 98.5, "close": 100.0}
        feats = detect_liquidity_sweeps(bars, swing_period=2)

        assert feats["sweep_low"].iloc[:12].sum() == 0
        assert feats["sweep_high"].iloc[:12].sum() == 0

    def test_sweep_low_without_same_bar_reclaim(self) -> None:
        bars = make_bars(V_PATH)
        # Takes the low 99.0 but closes below it -> no same-bar reclaim.
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 101.6, "low": 98.5, "close": 98.8}
        feats = detect_liquidity_sweeps(bars, swing_period=2)

        assert feats["sweep_low"].iloc[12] == 1
        assert feats["sweep_low_reclaimed"].iloc[12] == 0
        # Next bar closes back above the level -> reclaim fires then.
        bars.loc[bars.index[13]] = {"open": 98.8, "high": 100.5, "low": 98.6, "close": 100.2}
        feats2 = detect_liquidity_sweeps(bars, swing_period=2)
        assert feats2["sweep_low_reclaimed"].iloc[13] == 1
        assert feats2["sweep_low_age"].iloc[13] == 1

    def test_sweep_distance_filter(self) -> None:
        bars = make_bars(V_PATH)
        # Deep breakdown 3+ ATR below the level is a breakout, not a sweep.
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 101.6, "low": 93.0, "close": 93.5}
        feats = detect_liquidity_sweeps(bars, swing_period=2, max_sweep_distance_atr=2.0)
        assert feats["sweep_low"].iloc[12] == 0


class TestSweepHigh:
    def test_sweep_high_detection(self) -> None:
        bars = make_bars(V_PATH)
        # Sweep bar: takes confirmed swing high 102.2, closes back below it.
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 102.5, "low": 101.0, "close": 101.5}
        feats = detect_liquidity_sweeps(bars, swing_period=2)

        assert feats["sweep_high"].iloc[12] == 1
        assert feats["sweep_high_level"].iloc[12] == pytest.approx(102.2)
        assert feats["sweep_high_reclaimed"].iloc[12] == 1
        assert feats["sweep_high_age"].iloc[12] == 0

    def test_continuation_does_not_refire(self) -> None:
        bars = make_bars(V_PATH)
        # Bar 12 takes 102.2 and closes above it (setup stays active).
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 102.5, "low": 101.0, "close": 102.4}
        # Bar 13 stays above the same level -> continuation, not a new sweep;
        # it closes back below, so the reclaim fires there.
        bars.loc[bars.index[13]] = {"open": 102.4, "high": 102.6, "low": 101.2, "close": 101.8}
        feats = detect_liquidity_sweeps(bars, swing_period=2)

        assert feats["sweep_high"].iloc[13] == 0
        assert feats["sweep_high_age"].iloc[13] == 1
        assert feats["sweep_high_level"].iloc[13] == pytest.approx(102.2)
        assert feats["sweep_high_reclaimed"].iloc[13] == 1


class TestBos:
    def test_bos_up_from_confirmed_swing(self) -> None:
        bars = make_bars(V_PATH)
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 102.8, "low": 101.2, "close": 102.6}
        structure = structure_levels(bars, swing_period=2)
        bos = detect_bos(bars, structure)

        assert bos["bos_up"].iloc[12] == 1
        assert bos["bos_up_level"].iloc[12] == pytest.approx(102.2)

    def test_wick_above_is_not_bos(self) -> None:
        bars = make_bars(V_PATH)
        # High pokes above 102.2 but close stays below -> no close-through BOS.
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 102.5, "low": 101.0, "close": 101.5}
        structure = structure_levels(bars, swing_period=2)
        bos = detect_bos(bars, structure)

        assert bos["bos_up"].iloc[12] == 0
        assert np.isnan(bos["bos_up_level"].iloc[12])

    def test_bos_down_from_confirmed_swing(self) -> None:
        bars = make_bars(V_PATH)
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 101.6, "low": 98.5, "close": 98.2}
        structure = structure_levels(bars, swing_period=2)
        bos = detect_bos(bars, structure)

        assert bos["bos_down"].iloc[12] == 1
        assert bos["bos_down_level"].iloc[12] == pytest.approx(99.0)


class TestCausality:
    def test_sweep_does_not_use_future_data(self) -> None:
        """Agent.md §26 pattern: adding future bars must not change the past."""
        bars = make_bars(V_PATH)
        bars.loc[bars.index[12]] = {"open": 101.4, "high": 102.5, "low": 98.5, "close": 100.0}
        bars.loc[bars.index[13]] = {"open": 100.0, "high": 100.6, "low": 99.5, "close": 100.2}
        bars.loc[bars.index[14]] = {"open": 100.2, "high": 101.0, "low": 100.0, "close": 100.8}

        cols = [
            "sweep_high",
            "sweep_low",
            "sweep_high_level",
            "sweep_low_level",
            "sweep_high_reclaimed",
            "sweep_low_reclaimed",
            "sweep_high_age",
            "sweep_low_age",
        ]
        full = detect_liquidity_sweeps(bars, swing_period=2)
        short = detect_liquidity_sweeps(bars.iloc[:12], swing_period=2)
        pd.testing.assert_frame_equal(full[cols].iloc[:12], short[cols], check_freq=False)

        bos_full = detect_bos(bars, structure_levels(bars, swing_period=2))
        bos_short = detect_bos(bars.iloc[:12], structure_levels(bars.iloc[:12], swing_period=2))
        pd.testing.assert_frame_equal(bos_full.iloc[:12], bos_short, check_freq=False)
