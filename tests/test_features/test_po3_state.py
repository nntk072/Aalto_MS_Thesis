"""Tests for PO3 manipulation/distribution state and IFVG zones (Agent.md §6, §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.po3_state import build_ifvg_zone_features, build_po3_state

N = 20


def make_bars(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (high, low, close) triples on a 1-min index."""
    idx = pd.date_range("2024-01-02 01:05", periods=len(rows), freq="1min")
    high = [r[0] for r in rows]
    low = [r[1] for r in rows]
    close = [r[2] for r in rows]
    open_ = [close[0]] + close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def empty_sweeps(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=index,
        columns=[
            "sweep_high",
            "sweep_low",
            "sweep_high_level",
            "sweep_low_level",
            "sweep_high_reclaimed",
            "sweep_low_reclaimed",
            "sweep_high_age",
            "sweep_low_age",
        ],
    )


def flat_asian(index: pd.DatetimeIndex, high: float, low: float) -> pd.DataFrame:
    return pd.DataFrame({"asian_high": high, "asian_low": low}, index=index)


# Sideways range, sweep low at bar 10, manipulation leg, rally.
LONG_PATH = [(101.5, 100.5, 101.0)] * 10 + [
    (100.8, 98.0, 100.0),  # bar 10: sweep low wick
    (101.0, 97.5, 100.2),  # bar 11: manipulation low 97.5
    (101.2, 100.0, 100.8),  # bar 12
    (101.6, 100.8, 101.4),  # bar 13: close 101.4 above manip high -> end
    (102.0, 101.2, 101.8),  # bar 14: distribution
    (102.2, 101.4, 102.0),
]


def long_sweeps(index: pd.DatetimeIndex) -> pd.DataFrame:
    sw = empty_sweeps(index)
    sw.loc[index[10], "sweep_low"] = 1.0
    sw.loc[index[10], "sweep_low_level"] = 98.5
    return sw


class TestPO3StateLong:
    def test_manipulation_starts_on_sweep(self) -> None:
        bars = make_bars(LONG_PATH)
        state = build_po3_state(bars, long_sweeps(bars.index), flat_asian(bars.index, 103.0, 99.0))

        assert state["po3_manipulation_active"].iloc[10] == 1
        assert state["po3_manipulation_low"].iloc[10] == pytest.approx(98.0)
        assert state["po3_manipulation_active"].iloc[:10].sum() == 0

    def test_manipulation_end_and_distribution(self) -> None:
        bars = make_bars(LONG_PATH)
        state = build_po3_state(bars, long_sweeps(bars.index), flat_asian(bars.index, 103.0, 99.0))

        # Bar 13 closes above the manipulation-leg high -> manipulation ends.
        assert state["po3_manipulation_end"].iloc[13] == 1
        assert state["po3_manipulation_end"].iloc[:13].sum() == 0
        assert state["po3_distribution"].iloc[13] == 1
        assert state["po3_distribution_direction"].iloc[13] == 1
        assert state["po3_manipulation_active"].iloc[13] == 0

    def test_extremes_persist_for_sl(self) -> None:
        bars = make_bars(LONG_PATH)
        state = build_po3_state(bars, long_sweeps(bars.index), flat_asian(bars.index, 103.0, 99.0))

        tail = state.iloc[13:]
        assert np.allclose(tail["po3_manipulation_low"].to_numpy(dtype=float), 97.5)
        assert (tail["po3_distribution"] == 1).all()
        assert (tail["po3_distribution_direction"] == 1).all()


class TestPO3StateShort:
    def test_short_manipulation_end_and_distribution(self) -> None:
        bars = make_bars(LONG_PATH)
        sw = empty_sweeps(bars.index)
        sw.loc[bars.index[10], "sweep_high"] = 1.0
        sw.loc[bars.index[10], "sweep_high_level"] = 101.8
        # Sweep-high wick to 102.5, base holds above 100.0, then breakdown.
        bars.loc[bars.index[10]] = {"open": 101.0, "high": 102.5, "low": 100.5, "close": 101.0}
        bars.loc[bars.index[11]] = {"open": 101.0, "high": 101.2, "low": 100.0, "close": 100.6}
        bars.loc[bars.index[12]] = {"open": 100.4, "high": 100.4, "low": 99.0, "close": 99.5}
        state = build_po3_state(bars, sw, flat_asian(bars.index, 103.0, 99.0))

        assert state["po3_manipulation_active"].iloc[10] == 1
        assert state["po3_manipulation_high"].iloc[11] == pytest.approx(102.5)
        assert state["po3_manipulation_end"].iloc[12] == 1
        assert state["po3_distribution_direction"].iloc[12] == -1
        tail = state.iloc[12:]
        assert np.allclose(tail["po3_manipulation_high"].to_numpy(dtype=float), 102.5)


class TestPO3StateGuards:
    def test_no_state_without_asian_context(self) -> None:
        bars = make_bars(LONG_PATH)
        no_asian = pd.DataFrame({"asian_high": np.nan, "asian_low": np.nan}, index=bars.index)
        state = build_po3_state(bars, long_sweeps(bars.index), no_asian)

        assert state["po3_manipulation_active"].sum() == 0
        assert state["po3_distribution"].sum() == 0

    def test_opposite_sweep_flips_direction(self) -> None:
        bars = make_bars(LONG_PATH)
        sw = long_sweeps(bars.index)
        sw.loc[bars.index[14], "sweep_high"] = 1.0
        sw.loc[bars.index[14], "sweep_high_level"] = 102.5
        state = build_po3_state(bars, sw, flat_asian(bars.index, 103.0, 99.0))

        # A fresh opposite sweep re-arms a short manipulation setup.
        assert state["po3_manipulation_active"].iloc[14] == 1
        assert state["po3_distribution_direction"].iloc[14] == 0

    def test_causality(self) -> None:
        bars = make_bars(LONG_PATH)
        sw = long_sweeps(bars.index)
        asian = flat_asian(bars.index, 103.0, 99.0)
        full = build_po3_state(bars, sw, asian)
        short = build_po3_state(bars.iloc[:12], sw.iloc[:12], asian.iloc[:12])
        pd.testing.assert_frame_equal(full.iloc[:12], short, check_freq=False)


class TestIFVGZones:
    def build_frame(self) -> pd.DataFrame:
        # Single bullish FVG: bar1 high 100.5 (bar 1), bar3 low 100.9 (bar 3),
        # confirmed same bar (close-through), retest into zone at bar 8.
        rows = [
            (100.0, 99.0, 99.5),
            (100.5, 99.5, 100.0),
            (100.5, 99.8, 100.0),
            (102.5, 100.9, 102.0),  # FVG zone 100.5-100.9 + close-through
            (103.0, 100.4, 102.5),  # low stays <= 100.5: no second FVG
            (103.5, 102.0, 103.0),
            (103.2, 102.0, 102.5),
            (103.0, 101.5, 101.8),
            (103.0, 100.6, 100.8),  # retest: close inside the zone
            (102.5, 101.0, 101.5),
        ]
        return make_bars(rows)

    def test_bullish_zone_active_and_in_zone(self) -> None:
        bars = self.build_frame()
        zones = build_ifvg_zone_features(bars, max_age_bars=50)

        confirmed = zones["ifvg_bull_active"] == 1
        assert confirmed.any()
        start = int(np.argmax(confirmed.to_numpy()))
        zone_low = zones["ifvg_bull_low"].iloc[start]
        zone_high = zones["ifvg_bull_high"].iloc[start]
        assert zone_low == pytest.approx(100.5)
        assert zone_high == pytest.approx(100.9)
        # Retest bar closes inside the zone bounds.
        assert zones["price_in_ifvg_bull"].iloc[8] == 1
        assert zones["ifvg_bull_distance_atr"].iloc[start] >= 0

    def test_zone_expires(self) -> None:
        bars = self.build_frame()
        zones = build_ifvg_zone_features(bars, max_age_bars=2)

        confirmed = zones["ifvg_bull_active"] == 1
        start = int(np.argmax(confirmed.to_numpy()))
        # Lifetime is start <= t < start + max_age_bars.
        assert zones["ifvg_bull_active"].iloc[start + 1] == 1
        assert zones["ifvg_bull_active"].iloc[start + 2] == 0

    def test_bearish_zone(self) -> None:
        # Bearish FVG (bar3 high 99.0 < bar1 low 99.5), confirmed same bar,
        # later retest back up into the zone (98.5-99.0 after re-confirmation).
        rows = [
            (101.0, 100.0, 100.5),
            (100.5, 99.5, 100.0),
            (100.2, 99.0, 99.5),
            (99.0, 97.5, 98.0),  # bearish FVG zone 99.0-99.5 + close-through
            (98.5, 97.0, 97.5),
            (97.8, 96.0, 96.5),
            (97.0, 95.8, 96.2),
            (96.5, 95.5, 96.0),
            (99.5, 98.2, 98.7),  # retest: close inside the active zone
            (99.0, 98.0, 98.5),
        ]
        bars = make_bars(rows)
        zones = build_ifvg_zone_features(bars, max_age_bars=50)

        confirmed = zones["ifvg_bear_active"] == 1
        assert confirmed.any()
        start = int(np.argmax(confirmed.to_numpy()))
        assert zones["ifvg_bear_low"].iloc[start] == pytest.approx(99.0)
        assert zones["ifvg_bear_high"].iloc[start] == pytest.approx(99.5)
        assert zones["price_in_ifvg_bear"].iloc[8] == 1

    def test_causality(self) -> None:
        bars = self.build_frame()
        full = build_ifvg_zone_features(bars, max_age_bars=50)
        short = build_ifvg_zone_features(bars.iloc[:8], max_age_bars=50)
        pd.testing.assert_frame_equal(full.iloc[:8], short, check_freq=False)
