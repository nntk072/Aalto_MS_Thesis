"""Causal M1 liquidity sweeps of confirmed higher-timeframe pivots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd

from ..data.resample import resample
from .indicators import atr
from .swings import TIMEFRAME_CONFIG, detect_pivots

_TF_MINUTES = {"M5": 5, "M10": 10, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_TF_CODE = dict(_TF_MINUTES)


@dataclass
class _PivotReference:
    side: str
    timeframe: str
    level: float
    pivot_time: pd.Timestamp
    confirmed_at: pd.Timestamp
    overshoot: float = float("nan")
    qualifies: bool = False

    def __getitem__(self, key: str) -> Any:
        """Keep the small mapping-style inspection API used by audit tests."""
        return getattr(self, key)


def _coerce_reference(reference: _PivotReference | dict[str, object]) -> _PivotReference:
    """Normalize test/audit mapping inputs into a typed internal record."""
    if isinstance(reference, _PivotReference):
        return reference
    return _PivotReference(
        side=str(reference["side"]),
        timeframe=str(reference["timeframe"]),
        level=float(cast(float, reference["level"])),
        pivot_time=pd.Timestamp(cast(pd.Timestamp, reference["pivot_time"])),
        confirmed_at=pd.Timestamp(cast(pd.Timestamp, reference["confirmed_at"])),
        overshoot=float(cast(float, reference.get("overshoot", float("nan")))),
        qualifies=bool(reference.get("qualifies", False)),
    )


def _pivot_references(bars: pd.DataFrame, timeframe: str) -> list[_PivotReference]:
    """Return pivot references with the instant their full confirming bar ends."""
    minutes = _TF_MINUTES[timeframe]
    tf = resample(bars, timeframe)  # type: ignore[arg-type]
    if tf.empty:
        return []

    # The trailing resampled bar may be partial. It must not participate in
    # right-hand confirmation, even if it happens to contain an apparent pivot.
    last_m1_end = pd.Timestamp(bars.index[-1]) + pd.Timedelta(minutes=1)
    bar_ends = pd.Series(tf.index, index=tf.index, dtype="object").shift(-1)
    if timeframe == "D1":
        final_bar_end = tf.index[-1] + pd.DateOffset(days=1)
    else:
        final_bar_end = tf.index[-1] + pd.Timedelta(minutes=minutes)
    bar_ends.iloc[-1] = final_bar_end
    complete = pd.DatetimeIndex(bar_ends) <= last_m1_end
    tf = tf.loc[complete]
    bar_ends = bar_ends.loc[complete]
    if tf.empty:
        return []

    config = TIMEFRAME_CONFIG[timeframe]
    pivots = detect_pivots(tf, left=int(config["left"]), right=int(config["right"]))
    refs: list[_PivotReference] = []
    for side, event_col, price_col, loc_col in (
        ("high", "pivot_high_event", "pivot_high_extreme", "pivot_high_location"),
        ("low", "pivot_low_event", "pivot_low_extreme", "pivot_low_location"),
    ):
        for i in np.flatnonzero(pivots[event_col].to_numpy(dtype=bool)):
            location = int(pivots[loc_col].iloc[i])
            pivot_time = pd.Timestamp(tf.index[location])
            confirmed_at = pd.Timestamp(bar_ends.iloc[i])
            refs.append(
                _PivotReference(
                    side=side,
                    timeframe=timeframe,
                    level=float(pivots[price_col].iloc[i]),
                    pivot_time=pivot_time,
                    confirmed_at=confirmed_at,
                )
            )
    return refs


@overload
def detect_confirmed_htf_sweeps(
    bars: pd.DataFrame,
    *,
    timeframes: tuple[str, ...] = ("M5", "M15", "H1"),
    max_sweep_distance_atr: float = 2.0,
    atr_period: int = 5,
    return_audit: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def detect_confirmed_htf_sweeps(
    bars: pd.DataFrame,
    *,
    timeframes: tuple[str, ...] = ("M5", "M15", "H1"),
    max_sweep_distance_atr: float = 2.0,
    atr_period: int = 5,
    return_audit: Literal[True],
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def detect_confirmed_htf_sweeps(
    bars: pd.DataFrame,
    *,
    timeframes: tuple[str, ...] = ("M5", "M15", "H1"),
    max_sweep_distance_atr: float = 2.0,
    atr_period: int = 5,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Detect M1 sweeps of confirmed standard higher-timeframe pivots.

    HTF bars are left-labeled intervals. A pivot confirmed on an HTF bar is
    available only at that interval's end; the first eligible M1 bar opens
    at or after that instant. References are consumed when first crossed,
    including crosses too deep to qualify as sweeps.

    If a candle crosses multiple references on one side, select the finest
    timeframe (M5, M10, M15, M30, H1, H4, then D1), then the reference nearest
    the wick, then the latest pivot, then the lower price as the stable tie-break.
    With ``return_audit=True``, return every crossed reference and its
    qualification/selection status as a second frame.
    """
    invalid = set(timeframes) - set(_TF_MINUTES)
    if invalid:
        raise ValueError(f"Unsupported sweep-reference timeframe(s): {sorted(invalid)}")
    if len(timeframes) != len(set(timeframes)):
        raise ValueError("Sweep-reference timeframes must be unique")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("M1 bars must have a unique, increasing DatetimeIndex")

    references = [
        _coerce_reference(ref) for tf in timeframes for ref in _pivot_references(bars, tf)
    ]
    priority = {tf: i for i, tf in enumerate(_TF_MINUTES)}
    references.sort(key=lambda r: (r.confirmed_at, priority[r.timeframe], r.pivot_time))

    n = len(bars)
    atr_values = atr(bars, period=atr_period).to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    features = pd.DataFrame(
        {
            "sweep_high": np.zeros(n),
            "sweep_low": np.zeros(n),
            "sweep_high_level": np.full(n, np.nan),
            "sweep_low_level": np.full(n, np.nan),
            "sweep_high_reclaimed": np.zeros(n),
            "sweep_low_reclaimed": np.zeros(n),
            "sweep_high_age": np.full(n, np.nan),
            "sweep_low_age": np.full(n, np.nan),
            "sweep_high_ref_tf_code": np.zeros(n),
            "sweep_low_ref_tf_code": np.zeros(n),
            "sweep_high_pivot_time_ns": np.full(n, np.nan),
            "sweep_low_pivot_time_ns": np.full(n, np.nan),
            "sweep_high_confirmed_at_ns": np.full(n, np.nan),
            "sweep_low_confirmed_at_ns": np.full(n, np.nan),
            "sweep_high_candidate_count": np.zeros(n),
            "sweep_low_candidate_count": np.zeros(n),
        },
        index=bars.index,
    )
    audit_rows: list[dict[str, object]] = []
    pending = list(references)
    active: list[_PivotReference] = []
    feature_cols = {str(name): i for i, name in enumerate(features.columns)}

    for i, timestamp in enumerate(bars.index):
        while pending and pending[0].confirmed_at <= timestamp:
            new_ref = pending.pop(0)
            active.append(new_ref)
            same_group = [
                ref
                for ref in active
                if ref.side == new_ref.side and ref.timeframe == new_ref.timeframe
            ]
            if len(same_group) > 8:
                oldest = min(same_group, key=lambda ref: ref.pivot_time)
                active.remove(oldest)

        for side in ("high", "low"):
            attack = highs[i] if side == "high" else lows[i]
            crossed: list[_PivotReference] = []
            for ref in active:
                if ref.side == side and (
                    (side == "high" and attack > ref.level)
                    or (side == "low" and attack < ref.level)
                ):
                    ref.overshoot = attack - ref.level if side == "high" else ref.level - attack
                    ref.qualifies = bool(
                        np.isfinite(atr_values[i])
                        and ref.overshoot <= max_sweep_distance_atr * atr_values[i]
                    )
                    crossed.append(ref)

            if not crossed:
                continue
            # Every crossed level is no longer an unswept liquidity reference.
            crossed_ids = {id(ref) for ref in crossed}
            active = [ref for ref in active if id(ref) not in crossed_ids]
            eligible = [ref for ref in crossed if ref.qualifies]
            for ref in crossed:
                audit_rows.append(
                    {
                        "sweep_time": timestamp,
                        "side": side,
                        "reference_timeframe": ref.timeframe,
                        "reference_price": ref.level,
                        "pivot_time": ref.pivot_time,
                        "confirmed_at": ref.confirmed_at,
                        "overshoot": ref.overshoot,
                        "eligible": ref.qualifies,
                        "selected": False,
                    }
                )
            if not eligible:
                continue

            selected = min(
                eligible,
                key=lambda ref: (
                    priority[ref.timeframe],
                    ref.overshoot,
                    -ref.pivot_time.value,
                    ref.level,
                ),
            )
            column = f"sweep_{side}"
            features.iat[i, feature_cols[column]] = 1.0
            features.iat[i, feature_cols[f"sweep_{side}_level"]] = selected.level
            features.iat[i, feature_cols[f"sweep_{side}_ref_tf_code"]] = _TF_CODE[
                selected.timeframe
            ]
            features.iat[i, feature_cols[f"sweep_{side}_pivot_time_ns"]] = selected.pivot_time.value
            features.iat[i, feature_cols[f"sweep_{side}_confirmed_at_ns"]] = (
                selected.confirmed_at.value
            )
            features.iat[i, feature_cols[f"sweep_{side}_candidate_count"]] = len(eligible)
            for row in reversed(audit_rows):
                if row["sweep_time"] != timestamp or row["side"] != side:
                    break
                if (
                    row["reference_timeframe"] == selected.timeframe
                    and row["reference_price"] == selected.level
                ):
                    row["selected"] = True
                    break

    if not return_audit:
        return features
    return features, pd.DataFrame(
        audit_rows,
        columns=[
            "sweep_time",
            "side",
            "reference_timeframe",
            "reference_price",
            "pivot_time",
            "confirmed_at",
            "overshoot",
            "eligible",
            "selected",
        ],
    )
