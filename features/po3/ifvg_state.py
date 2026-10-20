"""Versioned PO3 manipulation/reclaim/IFVG phase state machine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_po3_ifvg_state(
    bars: pd.DataFrame,
    sweeps: pd.DataFrame,
    ifvg_zones: pd.DataFrame,
    *,
    max_ifvg_age_bars: int = 50,
) -> pd.DataFrame:
    """Build the revised causal phase sequence for Idea 1/2.

    A sweep starts manipulation. Its exact swept reference determines the
    first reclaim close. A matching IFVG confirmation is eligible only after
    reclaim and only when that FVG originated on or after the setup sweep.
    Distribution begins on the confirmation row. The later-ready flag starts
    on the next row. Reset/rearm remains the legacy policy: a fresh sweep
    replaces the current setup; otherwise distribution persists.
    """
    if max_ifvg_age_bars < 0:
        raise ValueError("max_ifvg_age_bars must be non-negative")
    n = len(bars)
    idx = bars.index
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    sweep_low = sweeps["sweep_low"].to_numpy(dtype=float)
    sweep_high = sweeps["sweep_high"].to_numpy(dtype=float)
    ref_low = sweeps["sweep_low_level"].to_numpy(dtype=float)
    ref_high = sweeps["sweep_high_level"].to_numpy(dtype=float)
    low_tf = sweeps.get("sweep_low_ref_tf_code", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    high_tf = sweeps.get("sweep_high_ref_tf_code", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    low_pivot_time = sweeps.get("sweep_low_pivot_time_ns", pd.Series(np.nan, index=idx)).to_numpy(
        dtype=float
    )
    high_pivot_time = sweeps.get("sweep_high_pivot_time_ns", pd.Series(np.nan, index=idx)).to_numpy(
        dtype=float
    )
    low_confirmed_at = sweeps.get(
        "sweep_low_confirmed_at_ns", pd.Series(np.nan, index=idx)
    ).to_numpy(dtype=float)
    high_confirmed_at = sweeps.get(
        "sweep_high_confirmed_at_ns", pd.Series(np.nan, index=idx)
    ).to_numpy(dtype=float)
    bull_confirm = ifvg_zones["ifvg_bull_confirmed"].to_numpy(dtype=float)
    bear_confirm = ifvg_zones["ifvg_bear_confirmed"].to_numpy(dtype=float)
    bull_origin = ifvg_zones["ifvg_bull_fvg_origin_bar"].to_numpy(dtype=float)
    bear_origin = ifvg_zones["ifvg_bear_fvg_origin_bar"].to_numpy(dtype=float)
    bull_confirmation_tf_code = ifvg_zones.get(
        "_ifvg_bull_confirmation_tf_code", pd.Series(1.0, index=idx)
    ).to_numpy(dtype=float)
    bear_confirmation_tf_code = ifvg_zones.get(
        "_ifvg_bear_confirmation_tf_code", pd.Series(1.0, index=idx)
    ).to_numpy(dtype=float)

    output = pd.DataFrame(
        {
            "po3_manipulation_active": np.zeros(n),
            "po3_manipulation_high": np.full(n, np.nan),
            "po3_manipulation_low": np.full(n, np.nan),
            "po3_manipulation_end": np.zeros(n),
            "po3_manipulation_direction": np.zeros(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "po3_waiting_ifvg": np.zeros(n),
            "po3_ifvg_confirmed": np.zeros(n),
            "po3_distribution_started": np.zeros(n),
            "po3_distribution_after_ifvg": np.zeros(n),
            "po3_setup_id": np.zeros(n),
            "po3_sweep_reference": np.full(n, np.nan),
            "po3_sweep_reference_tf_code": np.zeros(n),
            "po3_sweep_pivot_time_ns": np.full(n, np.nan),
            "po3_sweep_confirmed_at_ns": np.full(n, np.nan),
            "po3_sweep_time_ns": np.full(n, np.nan),
            "po3_reclaim_time_ns": np.full(n, np.nan),
            "po3_ifvg_time_ns": np.full(n, np.nan),
            "po3_ifvg_confirmation_tf_code": np.zeros(n),
            "po3_ifvg_fvg_origin_bar": np.full(n, np.nan),
            "po3_ifvg_fvg_origin_time_ns": np.full(n, np.nan),
            "po3_ifvg_low": np.full(n, np.nan),
            "po3_ifvg_high": np.full(n, np.nan),
            "po3_ifvg_through_zone_extreme": np.full(n, np.nan),
            "po3_sweep_conflict": np.zeros(n),
        },
        index=idx,
    )
    column_index = {str(name): i for i, name in enumerate(output.columns)}

    # state: 0 idle, 1/2 manipulation long/short, 3/4 waiting long/short,
    # 5/6 distribution long/short.
    state = 0
    direction = 0
    setup_id = 0
    sweep_index = -1
    sweep_reference = np.nan
    reference_tf = 0.0
    pivot_time_ns = confirmed_at_ns = sweep_time_ns = np.nan
    reclaim_time_ns = ifvg_time_ns = np.nan
    run_high = run_low = np.nan
    ifvg_origin_bar = np.nan
    ifvg_origin_time_ns = np.nan
    zone_low = zone_high = np.nan
    through_zone_extreme = np.nan
    ifvg_index = -1

    for t in range(n):
        low_event = sweep_low[t] > 0.0
        high_event = sweep_high[t] > 0.0
        if low_event and high_event:
            output.iat[t, column_index["po3_sweep_conflict"]] = 1.0
        elif low_event or high_event:
            new_direction = 1 if low_event else -1
            state = 1 if new_direction == 1 else 2
            direction = new_direction
            setup_id += 1
            sweep_index = t
            sweep_reference = ref_low[t] if direction == 1 else ref_high[t]
            reference_tf = low_tf[t] if direction == 1 else high_tf[t]
            pivot_time_ns = low_pivot_time[t] if direction == 1 else high_pivot_time[t]
            confirmed_at_ns = low_confirmed_at[t] if direction == 1 else high_confirmed_at[t]
            sweep_time_ns = pd.Timestamp(idx[t]).value
            reclaim_time_ns = ifvg_time_ns = np.nan
            ifvg_index = -1
            run_high, run_low = high[t], low[t]
            ifvg_origin_bar = ifvg_origin_time_ns = zone_low = zone_high = np.nan
            through_zone_extreme = np.nan

        if state in (1, 2):
            run_high = max(run_high, high[t])
            run_low = min(run_low, low[t])
            reclaimed = close[t] > sweep_reference if direction == 1 else close[t] < sweep_reference
            if reclaimed:
                state = 3 if direction == 1 else 4
                reclaim_time_ns = pd.Timestamp(idx[t]).value
                output.iat[t, column_index["po3_manipulation_end"]] = 1.0

        if state in (1, 2):
            output.iat[t, column_index["po3_manipulation_active"]] = 1.0

        if state == 3 and direction == 1 and bull_confirm[t] > 0.0:
            origin = bull_origin[t]
            if np.isfinite(origin) and sweep_index <= int(origin) <= t:
                if t - int(origin) <= max_ifvg_age_bars:
                    state = 5
                    ifvg_origin_bar = origin
                    ifvg_origin_time_ns = pd.Timestamp(idx[int(origin)]).value
                    zone_low = float(ifvg_zones["ifvg_bull_low"].iloc[t])
                    zone_high = float(ifvg_zones["ifvg_bull_high"].iloc[t])
                    if low[t] <= zone_high and high[t] >= zone_low:
                        through_zone_extreme = low[t]
                    ifvg_time_ns = pd.Timestamp(idx[t]).value
                    ifvg_index = t
                    output.iat[t, column_index["po3_ifvg_confirmed"]] = 1.0
                    output.iat[t, column_index["po3_distribution_started"]] = 1.0
                    output.iat[t, column_index["po3_ifvg_confirmation_tf_code"]] = (
                        bull_confirmation_tf_code[t]
                    )
        elif state == 4 and direction == -1 and bear_confirm[t] > 0.0:
            origin = bear_origin[t]
            if np.isfinite(origin) and sweep_index <= int(origin) <= t:
                if t - int(origin) <= max_ifvg_age_bars:
                    state = 6
                    ifvg_origin_bar = origin
                    ifvg_origin_time_ns = pd.Timestamp(idx[int(origin)]).value
                    zone_low = float(ifvg_zones["ifvg_bear_low"].iloc[t])
                    zone_high = float(ifvg_zones["ifvg_bear_high"].iloc[t])
                    if high[t] >= zone_low and low[t] <= zone_high:
                        through_zone_extreme = high[t]
                    ifvg_time_ns = pd.Timestamp(idx[t]).value
                    ifvg_index = t
                    output.iat[t, column_index["po3_ifvg_confirmed"]] = 1.0
                    output.iat[t, column_index["po3_distribution_started"]] = 1.0
                    output.iat[t, column_index["po3_ifvg_confirmation_tf_code"]] = (
                        bear_confirmation_tf_code[t]
                    )

        if state in (3, 4):
            output.iat[t, column_index["po3_waiting_ifvg"]] = 1.0

        if state in (5, 6) and low[t] <= zone_high and high[t] >= zone_low:
            excursion = low[t] if direction == 1 else high[t]
            if not np.isfinite(through_zone_extreme):
                through_zone_extreme = excursion
            elif direction == 1:
                through_zone_extreme = min(through_zone_extreme, excursion)
            else:
                through_zone_extreme = max(through_zone_extreme, excursion)

        if state in (5, 6):
            output.iat[t, column_index["po3_distribution"]] = 1.0
            output.iat[t, column_index["po3_distribution_direction"]] = float(direction)
            if t > ifvg_index:
                output.iat[t, column_index["po3_distribution_after_ifvg"]] = 1.0

        if setup_id:
            output.iat[t, column_index["po3_setup_id"]] = float(setup_id)
            output.iat[t, column_index["po3_manipulation_high"]] = run_high
            output.iat[t, column_index["po3_manipulation_low"]] = run_low
            output.iat[t, column_index["po3_manipulation_direction"]] = (
                -1.0 if direction == 1 else 1.0
            )
            output.iat[t, column_index["po3_sweep_reference"]] = sweep_reference
            output.iat[t, column_index["po3_sweep_reference_tf_code"]] = reference_tf
            output.iat[t, column_index["po3_sweep_pivot_time_ns"]] = pivot_time_ns
            output.iat[t, column_index["po3_sweep_confirmed_at_ns"]] = confirmed_at_ns
            output.iat[t, column_index["po3_sweep_time_ns"]] = sweep_time_ns
            output.iat[t, column_index["po3_reclaim_time_ns"]] = reclaim_time_ns
            output.iat[t, column_index["po3_ifvg_time_ns"]] = ifvg_time_ns
            output.iat[t, column_index["po3_ifvg_fvg_origin_bar"]] = ifvg_origin_bar
            output.iat[t, column_index["po3_ifvg_fvg_origin_time_ns"]] = ifvg_origin_time_ns
            output.iat[t, column_index["po3_ifvg_low"]] = zone_low
            output.iat[t, column_index["po3_ifvg_high"]] = zone_high
            output.iat[t, column_index["po3_ifvg_through_zone_extreme"]] = through_zone_extreme

    return output
