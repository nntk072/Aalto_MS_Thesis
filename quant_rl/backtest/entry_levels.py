"""Strategy stop filter and dynamic take-profit.

The env still opens the trade, sizes it, and writes the log. This module
places the stop on structure and the target on this bar's reachable reward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_rl.backtest.risk import compute_sl_tp_from_structure


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


def dynamic_tp(
    *,
    direction: int,
    entry_price: float,
    sl_price: float,
    reward_fraction: float,
    max_tp_distance: float | None = None,
) -> tuple[float | None, bool]:
    """Place the target between the stop distance and this bar's reachable distance.

    ``reward_fraction`` is in ``[0, 1]``. At 0 the target sits one stop-distance
    away (1R). At 1 it sits at ``max_tp_distance``, which may be past 5R.
    The realized ratio is that distance divided by the stop distance.

    Returns ``(tp_price, rejected)``. Rejected when a finite reachable
    distance is shorter than the stop, so the bar cannot pay 1R. A non-finite
    distance has no measured room, so the target stays at 1R.
    """
    if direction not in (1, -1):
        return None, False
    risk = abs(float(entry_price) - float(sl_price))
    if risk <= 1e-12:
        return None, False
    frac = float(np.clip(reward_fraction, 0.0, 1.0))
    room = float(max_tp_distance) if max_tp_distance is not None else float("nan")
    if np.isfinite(room) and room + 1e-9 < risk:
        return None, True
    extra = (room - risk) if np.isfinite(room) and room > risk else 0.0
    dist = risk + frac * extra
    return float(entry_price) + direction * dist, False


def resolve_strategy_entry(
    strategy: Any,
    *,
    direction: int,
    entry_price: float,
    reward_fraction: float,
    feat_row: pd.Series,
    min_dist: float,
    sl_anchor: float,
    buffer_pts: float,
    max_tp_distance: float,
) -> tuple[float | None, float | None, bool]:
    """Return ``(sl, tp, rr_rejected)`` for a strategy entry.

    The stop is structural. The target is a continuous fraction of this bar's
    reachable reward. ``rr_rejected`` is true when that reward is shorter
    than the stop. Dollar size is left to the caller.
    """
    cands = strategy.sl_candidates(direction=direction, row=feat_row)
    if not cands:
        sl_ref = strategy.sl_reference(direction=direction, row=feat_row)
        if sl_ref is not None and np.isfinite(float(sl_ref)):
            cands = [("sl_reference", float(sl_ref))]
    valid = filter_sl_candidates(
        direction=direction,
        entry_price=entry_price,
        candidates=cands,
        min_dist=min_dist,
    )
    if not valid:
        return None, None, False
    anchor = float(np.clip(sl_anchor, 0.0, 1.0))
    idx = min(int(round(anchor * (len(valid) - 1))), len(valid) - 1)
    _name, sl_level = valid[idx]
    try:
        sl_price, _tp_unused = compute_sl_tp_from_structure(
            direction=direction,
            entry_price=entry_price,
            structure_level=float(sl_level),
            rr_ratio=1.0,
            buffer_pts=buffer_pts,
        )
        if abs(entry_price - sl_price) + 1e-12 < min_dist:
            raise ValueError("SL below min distance")
    except ValueError:
        return None, None, False
    tp_price, rejected = dynamic_tp(
        direction=direction,
        entry_price=entry_price,
        sl_price=float(sl_price),
        reward_fraction=reward_fraction,
        max_tp_distance=max_tp_distance,
    )
    if rejected or tp_price is None:
        return None, None, rejected
    return float(sl_price), float(tp_price), False
