"""Observation feature schema and raw-metadata exclusion rules."""

from __future__ import annotations

import pandas as pd

# Raw price-level suffixes the environment must exclude from the model
# sequence (Agent.md §11): the M1 convention is ``strategy.raw_columns``
# (exact names, e.g. ``last_swing_high``). MTF blocks reuse the same stems
# with a ``{TF}_`` prefix (``M5_last_swing_high``), so the env matches
# ``{TF}_{stem}`` for every stem below. ATR-normalised distances
# (``*_dist_to_swing_*_atr``, ``*_distance_atr``) and binary flags stay in.
MTF_RAW_SUFFIXES = (
    "last_swing_high",
    "last_swing_low",
    *(
        f"swing_{side}_candidate_{slot}{suffix}"
        for side in ("high", "low")
        for slot in range(1, 4)
        for suffix in ("", "_pivot_time_ms", "_confirmed_at_ms", "_age_bars")
    ),
    "sweep_high_level",
    "sweep_low_level",
    "sweep_high_pivot_time_ns",
    "sweep_low_pivot_time_ns",
    "sweep_high_confirmed_at_ns",
    "sweep_low_confirmed_at_ns",
    "bos_up_level",
    "bos_down_level",
    "mss_up_level",
    "mss_down_level",
    "po3_manipulation_high",
    "po3_manipulation_low",
    "po3_setup_id",
    "po3_sweep_reference",
    "po3_sweep_pivot_time_ns",
    "po3_sweep_confirmed_at_ns",
    "po3_sweep_time_ns",
    "po3_reclaim_time_ns",
    "po3_ifvg_time_ns",
    "po3_ifvg_fvg_origin_bar",
    "po3_ifvg_fvg_origin_time_ns",
    "po3_ifvg_low",
    "po3_ifvg_high",
    "po3_ifvg_through_zone_extreme",
    "ifvg_bull_low",
    "ifvg_bull_high",
    "ifvg_bear_low",
    "ifvg_bear_high",
    "ifvg_bull_origin",
    "ifvg_bear_origin",
    "ifvg_bull_fvg_origin_bar",
    "ifvg_bear_fvg_origin_bar",
    "fvg_bull_low",
    "fvg_bull_high",
    "fvg_bear_low",
    "fvg_bear_high",
    "fvg_bull_origin",
    "fvg_bear_origin",
    "asian_high",
    "asian_low",
    "london_high",
    "london_low",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
    "ctx_asia_high",
    "ctx_asia_low",
    "ctx_london_high",
    "ctx_london_low",
    "ctx_prev_day_high",
    "ctx_prev_day_low",
    "ctx_prev_week_high",
    "ctx_prev_week_low",
    "previous_session_high",
    "previous_session_low",
    "previous_session_midpoint",
    "previous_session_completed_at_ns",
    "decision_close",
)

# Absolute timestamps and row/setup identifiers are retained in the feature
# frame for causal audits and strategy state, but are not model features. In
# particular, Unix nanoseconds become ~1e18 float32 inputs and leak calendar
# position into the policy without expressing a useful duration. Relative age
# and elapsed-bar features remain eligible for the model.
OBSERVATION_METADATA_SUFFIXES = (
    "_time_ns",
    "_at_ns",
    "_time_ms",
    "_at_ms",
    "_origin_bar",
    "_setup_id",
)

# Unprefixed raw price stems always dropped from model ``seq`` (baseline included).
OBS_RAW_PRICE_COLUMNS = MTF_RAW_SUFFIXES


def select_legacy_obs_columns(
    features: pd.DataFrame,
    raw_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Reproduce the pre-sequence-schema observation filter for explicit legacy replay."""
    drop_cols = [c for c in raw_columns if c in features.columns]
    mtf_raw = {
        c
        for c in features.columns
        if any(str(c).endswith(f"_{stem}") for stem in MTF_RAW_SUFFIXES)
        and str(c) not in raw_columns
    }
    kept = features.drop(columns=drop_cols + sorted(mtf_raw))
    return kept.select_dtypes(include="number")


def select_obs_columns(
    features: pd.DataFrame,
    raw_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Select model columns, excluding raw prices and absolute metadata."""
    legacy = select_legacy_obs_columns(features, raw_columns)
    metadata_cols = [
        str(column)
        for column in legacy.columns
        if str(column).endswith(OBSERVATION_METADATA_SUFFIXES)
    ]
    return legacy.drop(columns=metadata_cols)
