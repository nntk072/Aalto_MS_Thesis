"""Strategy stop filter and take-profit choice.

The env still opens the trade, sizes it, and writes the log. This module only
decides whether a structural stop and reward-ratio target are allowed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_rl.backtest.risk import compute_sl_tp_from_structure
from quant_rl.backtest.tp_reach import rr_reaches


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


def resolve_trader_tp(
    *,
    direction: int,
    entry_price: float,
    sl_price: float,
    rr_ratio: float,
    targets: dict[str, float],
    max_tp_distance: float | None = None,
) -> float:
    """RR TP, or the nearest structural target that still meets the ratio.

    Levels farther than ``max_tp_distance`` are ignored. Among the rest, the
    nearest level at or beyond the chosen ratio wins.
    """
    risk = abs(entry_price - sl_price)
    rr_tp = entry_price + direction * rr_ratio * risk
    best_struct: float | None = None
    best_implied = float("inf")
    cap = float(max_tp_distance) if max_tp_distance is not None else float("nan")
    for level in targets.values():
        if not np.isfinite(level):
            continue
        dist = abs(float(level) - entry_price)
        if np.isfinite(cap) and dist > cap + 1e-9:
            continue
        if direction == 1 and level > entry_price:
            implied = (level - entry_price) / risk if risk > 0 else 0.0
        elif direction == -1 and level < entry_price:
            implied = (entry_price - level) / risk if risk > 0 else 0.0
        else:
            continue
        if implied >= rr_ratio and implied < best_implied:
            best_implied = implied
            best_struct = float(level)
    return best_struct if best_struct is not None else float(rr_tp)


def resolve_strategy_entry(
    strategy: Any,
    *,
    direction: int,
    entry_price: float,
    rr_ratio: float,
    feat_row: pd.Series,
    min_dist: float,
    sl_anchor: float,
    buffer_pts: float,
    max_tp_distance: float,
) -> tuple[float | None, float | None, bool]:
    """Return ``(sl, tp, rr_rejected)`` for a strategy entry.

    ``rr_rejected`` is true when the chosen ratio does not fit the recent
    session range. A non-finite cap allows the trade. Dollar size is left to
    the caller.
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
        sl_price, _tp_rr = compute_sl_tp_from_structure(
            direction=direction,
            entry_price=entry_price,
            structure_level=float(sl_level),
            rr_ratio=rr_ratio,
            buffer_pts=buffer_pts,
        )
        if abs(entry_price - sl_price) + 1e-12 < min_dist:
            raise ValueError("SL below min distance")
    except ValueError:
        return None, None, False
    targets = strategy.target_candidates(direction=direction, row=feat_row)
    tp_price = resolve_trader_tp(
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        rr_ratio=rr_ratio,
        targets=targets,
        max_tp_distance=max_tp_distance,
    )
    if not rr_reaches(entry_price, sl_price, rr_ratio, max_tp_distance):
        return None, None, True
    return float(sl_price), float(tp_price), False
