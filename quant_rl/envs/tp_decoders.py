"""Deterministic decoders and validators for TP modes, multi-TP selections, and fractions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

DISABLED: None = None
_TP_MODES = ("fixed", "breakeven", "trailing")


def decode_tp_mode(
    action_val: float,
    default_mode: str = "fixed",
    defer_threshold: float = 0.1,
    is_unit: bool = False,
) -> tuple[str, bool]:
    """Decode TP mode from continuous action.

    Args:
        action_val: Action value in [-1, 1] (or [0, 1] if is_unit=True).
        default_mode: Fallback mode when within defer threshold.
        defer_threshold: Band around neutral (0.5 for unit, 0.0 for raw).
        is_unit: If True, action_val is in [0, 1]. If False, action_val is in [-1, 1].

    Returns:
        tuple of (effective_mode, is_explicit).
    """
    val = float(action_val)
    if is_unit:
        u = float(np.clip(val, 0.0, 1.0))
    else:
        raw = float(np.clip(val, -1.0, 1.0))
        u = 0.5 * (raw + 1.0)

    if abs(u - 0.5) < float(defer_threshold):
        return default_mode, False

    mode_idx = min(2, int(u * 3))
    return _TP_MODES[mode_idx], True


def _decode_single_slot(val: Any, menu_size: int, continuous: bool | None = None) -> int | None:
    if val is None or val is DISABLED:
        return None
    if isinstance(val, (int, np.integer)):
        if val < 0:
            return None
        return min(int(val), menu_size - 1)
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if np.isnan(fval) or fval < 0.0:
            return None
        is_cont = continuous if continuous is not None else (fval <= 1.0)
        if is_cont:
            return min(int(fval * menu_size), menu_size - 1)
        return min(int(fval), menu_size - 1)
    return None


def decode_tp_selections(
    tp1_raw: Any,
    tp2_raw: Any,
    tp3_raw: Any,
    menu_size: int,
    continuous: bool | None = None,
) -> tuple[int | None, int | None, int | None]:
    """Decode raw TP selections into deterministic, slot-preserving menu indices.

    Preserves slot identity and maximum valid ordered subset.

    Args:
        tp1_raw: Raw selection for slot TP1 (int index, float continuous, or sentinel).
        tp2_raw: Raw selection for slot TP2.
        tp3_raw: Raw selection for slot TP3.
        menu_size: Number of available targets in target menu.
        continuous: If True, floats in [0, 1] are treated as continuous box coordinates.

    Returns:
        tuple of (tp1_idx, tp2_idx, tp3_idx) where each is int or None (DISABLED).
    """
    if menu_size <= 0:
        return None, None, None

    s1 = _decode_single_slot(tp1_raw, menu_size, continuous)
    s2 = _decode_single_slot(tp2_raw, menu_size, continuous)
    s3 = _decode_single_slot(tp3_raw, menu_size, continuous)

    # 1. All-DISABLED fallback: activate nearest valid target as sole TP (TP3)
    if s1 is None and s2 is None and s3 is None:
        return None, None, 0

    # 2. Maximum-valid-subset repair & TP3 requirement:
    # If TP3 is disabled, promote the furthest valid candidate to TP3
    if s3 is None:
        candidates = [(slot, val) for slot, val in [(1, s1), (2, s2)] if val is not None]
        if candidates:
            # Pick highest menu index (break tie by preferring slot 2)
            promoted_slot, promoted_val = max(candidates, key=lambda item: (item[1], item[0]))
            s3 = promoted_val
            if promoted_slot == 1:
                s1 = None
                # Check if s2 can remain active and strictly precede s3
                if s2 is not None and s2 >= s3:
                    s2 = None
            else:
                s2 = None
                # Check if s1 can remain active and strictly precede s3
                if s1 is not None and s1 >= s3:
                    s1 = None

    # 3. Remove duplicates: disable nearer slot
    if s1 is not None and s2 is not None and s1 == s2:
        s1 = None
    if s2 is not None and s3 is not None and s2 == s3:
        s2 = None
    if s1 is not None and s3 is not None and s1 == s3:
        s1 = None

    # 4. Enforce monotonic index ordering: idx(tp1) < idx(tp2) < idx(tp3)
    if s1 is not None and s2 is not None and s1 >= s2:
        s1 = None
    if s2 is not None and s3 is not None and s2 >= s3:
        s2 = None
    if s1 is not None and s3 is not None and s1 >= s3:
        s1 = None

    # 5. Ensure TP3 remains active
    if s3 is None:
        s3 = 0

    return s1, s2, s3


def decode_tp_fractions(
    z1: float,
    z2: float,
    active_slots: Sequence[Any] | None = None,
) -> tuple[float, ...]:
    """Decode 2-D simplex logits into normalized fractions for active TP slots.

    Args:
        z1: First simplex logit.
        z2: Second simplex logit.
        active_slots: Optional sequence of 3 slot states (index or bool) indicating
            which of (TP1, TP2, TP3) are active.

    Returns:
        tuple of normalized fractions summing to 1.0 for the active slots.
    """
    m = max(float(z1), float(z2), 0.0)
    e1 = math.exp(float(z1) - m)
    e2 = math.exp(float(z2) - m)
    e3 = math.exp(-m)
    denom = e1 + e2 + e3
    f1 = e1 / denom
    f2 = e2 / denom
    f3 = e3 / denom

    if active_slots is None:
        return (f1, f2, f3)

    mask = [s is not None and s is not False for s in active_slots]
    if len(mask) != 3:
        raise ValueError(f"active_slots must have length 3, got {len(active_slots)}")

    active_fractions: list[float] = []
    if mask[0]:
        active_fractions.append(f1)
    if mask[1]:
        active_fractions.append(f2)
    if mask[2]:
        active_fractions.append(f3)

    if not active_fractions:
        return (1.0,)

    total = sum(active_fractions)
    if total <= 1e-12:
        return tuple(1.0 / len(active_fractions) for _ in active_fractions)

    return tuple(float(f / total) for f in active_fractions)


def validate_tp_levels(
    direction: int,
    entry_price: float,
    sl_price: float | None,
    tp_prices: Sequence[float | None],
    raise_exc: bool = False,
) -> bool:
    """Validate TP ordering and SL/TP crossing protection invariants.

    Long:  SL < Entry <= TP1 < TP2 < TP3
    Short: TP3 < TP2 < TP1 <= Entry < SL

    Args:
        direction: Trade direction (+1 long, -1 short).
        entry_price: Entry price.
        sl_price: Stop loss price or None.
        tp_prices: Sequence of (tp1_price, tp2_price, tp3_price) or active levels.
        raise_exc: Whether to raise ValueError on invalid ordering.

    Returns:
        True if all invariants hold, False otherwise.
    """
    active = [float(p) for p in tp_prices if p is not None]

    if direction == 1:
        if sl_price is not None and float(sl_price) >= float(entry_price):
            if raise_exc:
                raise ValueError(f"Long SL {sl_price} must be < entry {entry_price}")
            return False
        for p in active:
            if p < float(entry_price):
                if raise_exc:
                    raise ValueError(f"Long TP {p} must be >= entry {entry_price}")
                return False
            if sl_price is not None and p <= float(sl_price):
                if raise_exc:
                    raise ValueError(f"Long TP {p} must be > SL {sl_price}")
                return False
        for i in range(len(active) - 1):
            if active[i] >= active[i + 1]:
                if raise_exc:
                    raise ValueError(
                        f"Long TP levels must be strictly increasing: {active[i]} >= {active[i + 1]}"
                    )
                return False
    elif direction == -1:
        if sl_price is not None and float(sl_price) <= float(entry_price):
            if raise_exc:
                raise ValueError(f"Short SL {sl_price} must be > entry {entry_price}")
            return False
        for p in active:
            if p > float(entry_price):
                if raise_exc:
                    raise ValueError(f"Short TP {p} must be <= entry {entry_price}")
                return False
            if sl_price is not None and p >= float(sl_price):
                if raise_exc:
                    raise ValueError(f"Short TP {p} must be < SL {sl_price}")
                return False
        # In slot order (TP1, TP2, TP3), TP1 is closest to entry (highest),
        # TP3 is furthest (lowest). So active[0] > active[1] > active[2].
        for i in range(len(active) - 1):
            if active[i] <= active[i + 1]:
                if raise_exc:
                    raise ValueError(
                        f"Short TP levels must be strictly decreasing: {active[i]} <= {active[i + 1]}"
                    )
                return False
    return True
