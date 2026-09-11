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
from quant_rl.features.swings import classify_structure


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
