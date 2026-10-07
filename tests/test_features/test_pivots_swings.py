"""Pivot confirmation, ATR swings, and structure_levels wrapper contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.features.structure import (
    detect_pivots,
    detect_swings,
    structure_levels,
    swing_features,
)
from quant_rl.features.swings import classify_structure, retained_swing_levels


def _pivot_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 01:05", periods=10, freq="1min")
    highs = [101.0, 101.2, 101.4, 101.6, 102.0, 101.5, 101.0, 100.8, 100.6, 100.4]
    lows = [100.5, 100.3, 100.1, 99.9, 99.5, 100.0, 100.5, 100.8, 101.0, 101.2]
    closes = [100.8, 100.7, 100.6, 100.5, 101.0, 100.8, 100.6, 100.4, 100.2, 100.0]
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes}, index=idx)


def test_pivot_confirmation_has_no_lookahead() -> None:
    bars = _pivot_bars()
    piv = detect_pivots(bars, left=2, right=2)
    assert bool(piv["pivot_high_event"].iloc[6])
    assert not bool(piv["pivot_high_event"].iloc[4])
    assert piv["pivot_high_location"].iloc[6] == pytest.approx(4)
    assert piv["pivot_high_price"].iloc[6] == pytest.approx(101.0)
    assert piv["pivot_high_extreme"].iloc[6] == pytest.approx(102.0)


def test_swings_only_visible_after_confirmation() -> None:
    bars = _pivot_bars()
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    assert np.isnan(swings["swing_high_extreme"].iloc[4])
    assert swings["swing_high_extreme"].iloc[6] == pytest.approx(102.0)


def test_structure_levels_wrapper_columns() -> None:
    bars = _pivot_bars()
    result = structure_levels(bars, swing_period=2)
    assert "last_swing_high" in result.columns
    assert "last_swing_low" in result.columns


def test_swing_features_are_causal() -> None:
    bars = _pivot_bars()
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    feat = swing_features(bars, swings, classify_structure(swings))
    trunc = bars.iloc[:8]
    piv_t = detect_pivots(trunc, left=2, right=2)
    sw_t = detect_swings(trunc, piv_t, atr_mult=0.0)
    feat_t = swing_features(trunc, sw_t, classify_structure(sw_t))
    overlap = feat_t.index[:6]
    pd.testing.assert_series_equal(
        feat.loc[overlap, "last_swing_side"],
        feat_t.loc[overlap, "last_swing_side"],
        check_names=False,
    )


def test_features_invariant_to_future_data() -> None:
    bars = _pivot_bars()
    full = structure_levels(bars, swing_period=2)
    trunc = structure_levels(bars.iloc[:8], swing_period=2)
    pd.testing.assert_series_equal(
        full["last_swing_high"].iloc[:6],
        trunc["last_swing_high"].iloc[:6],
        check_names=False,
    )


def test_retained_levels_appear_only_after_confirmation() -> None:
    bars = _pivot_bars()
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    retained = retained_swing_levels(swings, bars)
    # The high at bar 4 is confirmed at bar 6; it must be absent before then.
    assert retained["n_retained_highs"].iloc[4] == 0
    assert retained["n_retained_highs"].iloc[6] == 1
    assert retained["retained_high_1"].iloc[6] == pytest.approx(102.0)


def test_retained_levels_are_removed_on_sweep() -> None:
    n = 40
    idx = pd.date_range("2024-01-02 01:05", periods=n, freq="1min")
    close = np.full(n, 100.0)
    # V-shaped swing low confirmed at bar 9, then price breaks below it.
    close[6] = 99.0
    close[7] = 98.0
    close[8] = 99.0
    close[9] = 100.0
    close[20:] = 97.0
    bars = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    retained = retained_swing_levels(swings, bars)
    # Before the sweep the low is retained; after, it is gone.
    assert retained["n_retained_lows"].iloc[15] == 1
    assert retained["n_retained_lows"].iloc[25] == 0


def test_swept_event_preserved_independently_of_retained_set() -> None:
    """A swept level must be distinguishable from one that never existed."""
    n = 40
    idx = pd.date_range("2024-01-02 01:05", periods=n, freq="1min")
    close = np.full(n, 100.0)
    close[6] = 99.0
    close[7] = 98.0
    close[8] = 99.0
    close[9] = 100.0
    close[20:] = 97.0
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    retained = retained_swing_levels(swings, bars)
    # A level was retained and later swept: the swept event column records it.
    assert retained["n_retained_lows"].iloc[15] == 1
    assert retained["n_retained_lows"].iloc[25] == 0
    assert np.isfinite(retained["swept_low"].iloc[20])
    assert retained["swept_low_loc"].iloc[20] == pytest.approx(7.0)


def test_retained_levels_incremental_equals_full_run() -> None:
    """Running incrementally must reproduce the full-dataset retained state.

    Catches accidental vectorized lookahead: any future-bar dependence would
    make the two differ on the overlap.
    """
    bars = _pivot_bars()
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    full = retained_swing_levels(swings, bars)

    trunc = bars.iloc[:8]
    piv_t = detect_pivots(trunc, left=2, right=2)
    sw_t = detect_swings(trunc, piv_t, atr_mult=0.0)
    part = retained_swing_levels(sw_t, trunc)

    overlap = part.index[:6]
    pd.testing.assert_series_equal(
        full["n_retained_highs"].loc[overlap],
        part["n_retained_highs"].loc[overlap],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        full["n_retained_lows"].loc[overlap],
        part["n_retained_lows"].loc[overlap],
        check_names=False,
    )


def test_last_swing_columns_unchanged_by_retained_levels() -> None:
    """Backward compat: last_swing_* keeps its legacy meaning exactly."""
    bars = _pivot_bars()
    legacy = structure_levels(bars, swing_period=2)
    assert "last_swing_high" in legacy.columns
    assert "last_swing_low" in legacy.columns
    assert "retained_high_1" in legacy.columns
    assert "retained_low_1" in legacy.columns
    # last_swing_* must still equal the accepted zigzag extreme.
    piv = detect_pivots(bars, left=2, right=2)
    swings = detect_swings(bars, piv, atr_mult=0.0)
    pd.testing.assert_series_equal(
        legacy["last_swing_high"],
        swings["swing_high_extreme"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        legacy["last_swing_low"],
        swings["swing_low_extreme"],
        check_names=False,
    )


def test_bars_since_last_swing_scales_by_obs_window() -> None:
    n = 61
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min")
    close = np.full(n, 100.0)
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close},
        index=idx,
    )
    high_event = np.zeros(n, dtype=bool)
    high_event[0] = True
    swings = pd.DataFrame(
        {
            "swing_high_event": high_event,
            "swing_low_event": np.zeros(n, dtype=bool),
            "swing_high_extreme": close,
            "swing_low_extreme": close,
            "swing_high_price": close,
            "swing_low_price": close,
        },
        index=idx,
    )
    feat = swing_features(
        bars,
        swings,
        classify_structure(swings),
        atr=pd.Series(1.0, index=idx),
        obs_window=60,
    )
    since = feat["bars_since_last_swing"]
    assert since.iloc[0] == pytest.approx(0.0)
    assert since.iloc[30] == pytest.approx(0.5)
    assert since.iloc[60] == pytest.approx(1.0)
    assert float(since.max()) <= 1.0
