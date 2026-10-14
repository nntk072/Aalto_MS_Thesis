"""Revised PO3 phase ordering and setup-association tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.features.po3_ifvg_state import build_po3_ifvg_state


def _inputs(direction: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2025-01-06 10:00", periods=7, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "high": [101.0, 101.0, 100.5, 101.5, 102.0, 102.5, 103.0],
            "low": [100.0, 100.0, 98.5, 99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 100.2, 98.8, 99.2, 101.0, 102.0, 102.5],
        },
        index=idx,
    )
    sweeps = pd.DataFrame(
        {
            "sweep_low": [0, 0, float(direction == 1), 0, 0, 0, 0],
            "sweep_high": [0, 0, float(direction == -1), 0, 0, 0, 0],
            "sweep_low_level": [np.nan, np.nan, 99.0, np.nan, np.nan, np.nan, np.nan],
            "sweep_high_level": [np.nan, np.nan, 101.0, np.nan, np.nan, np.nan, np.nan],
            "sweep_low_ref_tf_code": [0, 0, 5, 0, 0, 0, 0],
            "sweep_high_ref_tf_code": [0, 0, 15, 0, 0, 0, 0],
            "sweep_low_pivot_time_ns": [np.nan, np.nan, idx[0].value, *([np.nan] * 4)],
            "sweep_high_pivot_time_ns": [np.nan, np.nan, idx[0].value, *([np.nan] * 4)],
            "sweep_low_confirmed_at_ns": [np.nan, np.nan, idx[1].value, *([np.nan] * 4)],
            "sweep_high_confirmed_at_ns": [np.nan, np.nan, idx[1].value, *([np.nan] * 4)],
        },
        index=idx,
    )
    zones = pd.DataFrame(
        {
            "ifvg_bull_confirmed": [0, 0, 0, 0, float(direction == 1), 0, 0],
            "ifvg_bear_confirmed": [0, 0, 0, 0, float(direction == -1), 0, 0],
            "ifvg_bull_fvg_origin_bar": [np.nan, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan],
            "ifvg_bear_fvg_origin_bar": [np.nan, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan],
            "ifvg_bull_low": [np.nan, np.nan, np.nan, np.nan, 99.5, np.nan, np.nan],
            "ifvg_bull_high": [np.nan, np.nan, np.nan, np.nan, 100.0, np.nan, np.nan],
            "ifvg_bear_low": [np.nan, np.nan, np.nan, np.nan, 100.0, np.nan, np.nan],
            "ifvg_bear_high": [np.nan, np.nan, np.nan, np.nan, 100.5, np.nan, np.nan],
        },
        index=idx,
    )
    return bars, sweeps, zones


def test_reclaim_waits_for_associated_ifvg_then_starts_distribution() -> None:
    bars, sweeps, zones = _inputs()
    state = build_po3_ifvg_state(bars, sweeps, zones)

    assert state["po3_manipulation_active"].iloc[2] == 1
    assert state["po3_manipulation_end"].iloc[3] == 1
    assert state["po3_waiting_ifvg"].iloc[3] == 1
    assert state["po3_distribution"].iloc[3] == 0
    assert state["po3_ifvg_confirmed"].iloc[4] == 1
    assert state["po3_distribution"].iloc[4] == 1
    assert state["po3_distribution_after_ifvg"].iloc[4] == 0
    assert state["po3_distribution_after_ifvg"].iloc[5] == 1
    assert state["po3_distribution_started"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert state["po3_setup_id"].iloc[5] == 1
    assert state["po3_sweep_reference_tf_code"].iloc[5] == 5
    assert state["po3_ifvg_fvg_origin_bar"].iloc[5] == 3


def test_same_candle_reclaim_is_allowed_but_distribution_waits_for_ifvg() -> None:
    bars, sweeps, zones = _inputs()
    bars.loc[bars.index[2], "close"] = 99.2
    state = build_po3_ifvg_state(bars, sweeps, zones)

    assert state["po3_manipulation_end"].iloc[2] == 1
    assert state["po3_waiting_ifvg"].iloc[2] == 1
    assert state["po3_distribution"].iloc[2:4].sum() == 0


def test_fvg_origin_before_setup_is_not_accepted() -> None:
    bars, sweeps, zones = _inputs()
    zones.loc[bars.index[4], "ifvg_bull_fvg_origin_bar"] = 1.0
    state = build_po3_ifvg_state(bars, sweeps, zones)

    assert state["po3_ifvg_confirmed"].sum() == 0
    assert state["po3_distribution"].sum() == 0


def test_ifvg_confirmed_before_reclaim_is_not_carried_forward() -> None:
    bars, sweeps, zones = _inputs()
    zones.loc[bars.index[2], "ifvg_bull_confirmed"] = 1.0
    zones.loc[bars.index[2], "ifvg_bull_fvg_origin_bar"] = 2.0
    zones.loc[bars.index[2], "ifvg_bull_low"] = 99.5
    zones.loc[bars.index[2], "ifvg_bull_high"] = 100.0
    zones.loc[bars.index[4], "ifvg_bull_confirmed"] = 0.0
    state = build_po3_ifvg_state(bars, sweeps, zones)

    assert state["po3_manipulation_end"].iloc[3] == 1
    assert state["po3_ifvg_confirmed"].sum() == 0
    assert state["po3_distribution"].sum() == 0


def test_revised_state_is_prefix_causal() -> None:
    bars, sweeps, zones = _inputs()
    full = build_po3_ifvg_state(bars, sweeps, zones)
    prefix = build_po3_ifvg_state(bars.iloc[:5], sweeps.iloc[:5], zones.iloc[:5])

    pd.testing.assert_frame_equal(full.iloc[:5], prefix, check_freq=False)


def test_short_setup_mirrors_reference_reclaim_and_ifvg_direction() -> None:
    bars, sweeps, zones = _inputs(direction=-1)
    bars.loc[bars.index[2], "close"] = 101.2
    bars.loc[bars.index[3], "close"] = 100.8
    state = build_po3_ifvg_state(bars, sweeps, zones)

    assert state["po3_manipulation_end"].iloc[3] == 1
    assert state["po3_ifvg_confirmed"].iloc[4] == 1
    assert state["po3_distribution_direction"].iloc[4] == -1
