"""Structural stop candidate validation and provenance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.orders.targets import (
    collapse_levels,
    deviation_candidates,
    finite_level,
    normalize_timeframe,
)


def _flag(row: Any, name: str) -> bool:
    value = finite_level(row, name)
    return value is not None and value > 0.0


def manipulation_state(row: pd.Series | Any) -> str:
    """Return the PO3 manipulation state that filters structural stops."""
    if _flag(row, "po3_manipulation_end"):
        return "done"
    if _flag(row, "po3_manipulation_active"):
        return "against"
    return "none"


def stop_buffer(buffer_pts: float) -> float:
    """Return the configured structural stop buffer, clamped at zero."""
    return max(float(buffer_pts), 0.0)


def filter_sl_candidates(
    *,
    direction: int,
    entry_price: float,
    candidates: list[tuple[str, float]],
    min_dist: float,
) -> list[tuple[str, float]]:
    """Keep SL levels on the correct side and at least ``min_dist`` away."""
    valid: list[tuple[str, float]] = []
    for name, level in candidates:
        if direction == 1 and level < entry_price:
            dist = entry_price - level
        elif direction == -1 and level > entry_price:
            dist = level - entry_price
        else:
            continue
        if dist + 1e-12 >= min_dist:
            valid.append((name, level))
    return valid


def stop_menu(
    strategy: Any,
    *,
    direction: int,
    entry_price: float,
    row: Any,
    min_dist: float,
    buffer: float,
    manipulation: str,
    htf: tuple[str, ...] = ("M5", "M15", "H1"),
    price_increment: float | None = None,
) -> tuple[list[tuple[str, float, float]], list[dict[str, Any]]]:
    """Build eligible structural stops and an audit trail for rejected levels."""
    early_audit: list[dict[str, Any]] = []
    candidate_meta: dict[str, dict[str, Any]] = {}
    candidate_schema_active = False
    raw = list(strategy.sl_candidates(direction=direction, row=row))
    if not raw and hasattr(strategy, "sl_reference"):
        ref = strategy.sl_reference(direction=direction, row=row)
        if ref is not None and np.isfinite(float(ref)):
            raw = [("sl_reference", float(ref))]
    swing = "last_swing_low" if direction == 1 else "last_swing_high"
    for tf in htf:
        tf_up = normalize_timeframe(str(tf))
        candidate_side = "low" if direction == 1 else "high"
        candidate_prefix = f"{tf_up}_swing_{candidate_side}_candidate_"
        candidate_columns = getattr(row, "index", ())
        has_candidate_schema = any(
            f"{candidate_prefix}{slot}" in candidate_columns for slot in range(1, 4)
        )
        if has_candidate_schema:
            candidate_schema_active = True
            decision_at = getattr(row, "name", None)
            decision_ms = getattr(row, "decision_at_ms", None)
            if decision_ms is None and decision_at is not None and not pd.isna(decision_at):
                decision_ms = int(pd.Timestamp(decision_at).value // 1_000_000)
            if decision_ms is not None:
                decision_ms = int(decision_ms)
            for slot in range(1, 4):
                name = f"{candidate_prefix}{slot}"
                level = finite_level(row, name)
                if level is None:
                    continue
                pivot_ms = finite_level(row, f"{name}_pivot_time_ms")
                confirmed_ms = finite_level(row, f"{name}_confirmed_at_ms")
                age_bars = finite_level(row, f"{name}_age_bars")
                reason = ""
                if decision_ms is None:
                    reason = "missing_decision_time"
                elif pivot_ms is None or confirmed_ms is None:
                    reason = "missing_provenance"
                elif age_bars is None or age_bars < 0:
                    reason = "invalid_age"
                elif pivot_ms > confirmed_ms or confirmed_ms > decision_ms:
                    reason = "unconfirmed_or_future_reference"
                if reason:
                    early_audit.append(
                        {
                            "name": name,
                            "reference_price": level,
                            "stop_price": float("nan"),
                            "distance": float("nan"),
                            "status": "rejected",
                            "reason": reason,
                            "timeframe": tf_up,
                            "pivot_time_ms": pivot_ms,
                            "confirmed_at_ms": confirmed_ms,
                            "age_bars": age_bars,
                        }
                    )
                    continue
                raw.append((name, level))
                candidate_meta[name] = {
                    "timeframe": tf_up,
                    "pivot_time_ms": pivot_ms,
                    "confirmed_at_ms": confirmed_ms,
                    "age_bars": age_bars,
                }
        else:
            level = finite_level(row, f"{tf_up}_{swing}")
            if level is not None:
                raw.append((f"{tf_up}_{swing}", level))
    if direction == 1:
        side_names = [
            "london_low",
            "asian_low",
            "prev_day_low",
            "ctx_prev_day_low",
            "ctx_prev_week_low",
            "yesterday_low",
            "lastweek_low",
        ]
    else:
        side_names = [
            "london_high",
            "asian_high",
            "prev_day_high",
            "ctx_prev_day_high",
            "ctx_prev_week_high",
            "yesterday_high",
            "lastweek_high",
        ]
    for name in side_names:
        level = finite_level(row, name)
        if level is not None:
            raw.append((name, level))
    raw.extend(deviation_candidates(row))
    smt_tfs = [normalize_timeframe(str(tf)) for tf in htf]
    smt_bullish = _flag(row, "smt_bullish") or any(
        _flag(row, f"{tf}_smt_bullish") for tf in smt_tfs
    )
    smt_bearish = _flag(row, "smt_bearish") or any(
        _flag(row, f"{tf}_smt_bearish") for tf in smt_tfs
    )
    if (direction == 1 and smt_bullish) or (direction == -1 and smt_bearish):
        raw = [("smt_swing", px) if name == swing else (name, px) for name, px in raw]
    before_filter = list(raw)
    if (
        manipulation == "against"
        and getattr(strategy, "manipulation_filter", "against") == "against"
    ):
        raw = [(name, px) for name, px in raw if "ifvg" not in name and "fvg" not in name]
    elif (
        manipulation == "against"
        and getattr(strategy, "manipulation_filter", "against") == "follow"
    ):
        raw = [(name, px) for name, px in raw if "ifvg" in name or "fvg" in name]
    menu: list[tuple[str, float, float]] = []
    audit: list[dict[str, Any]] = list(early_audit)
    filtered_names = {name for name, _ in before_filter} - {name for name, _ in raw}
    for name in sorted(filtered_names):
        level = next(float(px) for candidate, px in before_filter if candidate == name)
        audit.append(
            {
                "name": name,
                "reference_price": level,
                "stop_price": float("nan"),
                "distance": float("nan"),
                "status": "rejected",
                "reason": "manipulation_filter",
            }
        )
    collapsed = collapse_levels(
        [(name, float(price)) for name, price in raw],
        price_increment=price_increment if candidate_schema_active else None,
    )
    for name, level in collapsed:
        stop = level - buffer if direction == 1 else level + buffer
        if (direction == 1 and stop >= entry_price) or (direction == -1 and stop <= entry_price):
            reason = "wrong_side"
        elif abs(entry_price - stop) + 1e-12 < min_dist:
            reason = "below_min_distance"
        else:
            distance = abs(entry_price - stop)
            menu.append((name, float(stop), distance))
            audit.append(
                {
                    "name": name,
                    "reference_price": level,
                    "stop_price": stop,
                    "distance": distance,
                    "status": "eligible",
                    "reason": "",
                }
            )
            continue
        audit.append(
            {
                "name": name,
                "reference_price": level,
                "stop_price": stop,
                "distance": abs(entry_price - stop),
                "status": "rejected",
                "reason": reason,
            }
        )
    for item in audit:
        names = str(item.get("name", "")).split("|")
        provenance = [candidate_meta[name] for name in names if name in candidate_meta]
        if provenance:
            item["provenance"] = provenance
    menu.sort(key=lambda item: item[2])
    return menu, audit


__all__ = ["filter_sl_candidates", "manipulation_state", "stop_buffer", "stop_menu"]
