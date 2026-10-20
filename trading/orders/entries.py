"""Structural stop and target placement for strategy entries.

The env still opens the trade and sizes it. This module picks one buffered
invalidation stop, then either one structural target or an EMA-21 exit.
It reads only columns already on the feature row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from trading.orders.selection import decode_tp_fractions, decode_tp_selections
from trading.orders.stops import (
    filter_sl_candidates,
    manipulation_state,
    stop_buffer,
    stop_menu,
)
from trading.orders.targets import (
    finite_level,
    target_menu,
)


@dataclass(frozen=True)
class TargetReference:
    """Provenance for one executable candidate; None means unverified metadata."""

    name: str
    timeframe: str
    original_price: float | None
    adjusted_price: float
    confirmed_at: str | None
    reason: str
    sources: tuple[str, ...] = ()
    fallback_reason: str | None = None


@dataclass(frozen=True)
class OrderGeometry:
    """One working stop and one exit, or a refusal."""

    sl_price: float | None
    tp_price: float | None
    sl_ref: str
    tp_ref: str
    sl_buffer: float
    planned_rr: float | None
    exit_mode: str
    manipulation: str
    n_sl: int
    n_tp: int
    rejected: bool
    reject_kind: str
    sl_index: int = -1
    tp_index: int = -1
    sl_menu: tuple[tuple[str, float], ...] = ()
    tp_menu: tuple[tuple[str, float], ...] = ()
    tp1_price: float | None = None
    tp2_price: float | None = None
    tp3_price: float | None = None
    tp1_ref: str = ""
    tp2_ref: str = ""
    tp3_ref: str = ""
    tp_lot_fractions: tuple[float, ...] = ()
    tp_indices: tuple[int | None, int | None, int | None] = (None, None, None)
    tp_references: tuple[TargetReference, ...] = ()
    sl_candidate_audit: tuple[dict[str, Any], ...] = ()


def _menu_pairs(
    menu: list[tuple[str, float, float]],
) -> tuple[tuple[str, float], ...]:
    """``(name, price)`` pairs from a ``(name, price, distance)`` menu."""
    return tuple((str(name), float(price)) for name, price, _ in menu)


def _with_menus(
    geometry: OrderGeometry,
    stops: list[tuple[str, float, float]],
    targets: list[tuple[str, float, float]],
    *,
    row: Any = None,
    source_prices: dict[str, float] | None = None,
    fallback_reason: str | None = None,
    stop_audit: list[dict[str, Any]] | None = None,
) -> OrderGeometry:
    """Attach the full candidate ladders so a chart can list every option."""
    selected = {i for i in geometry.tp_indices if i is not None}
    if not selected and geometry.tp_index >= 0 and geometry.exit_mode != "ema_21":
        selected = {geometry.tp_index}
    original_prices = source_prices or {}
    references: list[TargetReference] = []
    for idx, (name, px, _) in enumerate(targets):
        names = tuple(name.split("|"))
        first = names[0]
        stamp = row.get(f"{first}_confirmed_at") if row is not None else None
        confirmed = None if stamp is None or pd.isna(stamp) else pd.Timestamp(stamp).isoformat()
        references.append(
            TargetReference(
                name=name,
                timeframe=reference_timeframe(name),
                original_price=original_prices.get(first),
                adjusted_price=float(px),
                confirmed_at=confirmed,
                reason="selected" if idx in selected else "eligible",
                sources=names,
                fallback_reason=fallback_reason,
            )
        )
    return replace(
        geometry,
        sl_menu=_menu_pairs(stops),
        tp_menu=_menu_pairs(targets),
        tp_references=tuple(references),
        sl_candidate_audit=tuple(stop_audit or ()),
    )


def select_index(fraction: float, n: int) -> int:
    """Map a fraction in ``[0, 1]`` onto ``[0, n)``.

    Index 0 is the nearest candidate. ``round`` is not used: half of two
    candidates must land on the farther one.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    frac = float(np.clip(fraction, 0.0, 1.0))
    return min(int(frac * n), n - 1)


def dynamic_tp(
    *,
    direction: int,
    entry_price: float,
    sl_price: float,
    reward_fraction: float,
    max_tp_distance: float | None = None,
) -> tuple[float | None, bool]:
    """Legacy reachable-reward target used by the chart helper.

    Strategy entries do not call this. They pick a structural price or an
    EMA-21 exit.
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


def place_strategy_order(
    strategy: Any,
    *,
    direction: int,
    entry_price: float,
    feat_row: pd.Series | Any,
    min_dist: float,
    sl_fraction: float,
    tp_fraction: float,
    exit_mode: str,
    buffer_pts: float,
    max_tp_distance: float,
    rr_bounds: tuple[float, float] | None = None,
    htf: tuple[str, ...] = ("M5", "M15", "H1"),
    tp_selections: tuple[Any, Any, Any] | None = None,
    tp_z: tuple[float, float] | None = None,
    allow_multi_tp: bool = False,
    allow_simplex: bool = True,
    target_front_run_pts: float = 0.0,
    price_increment: float | None = None,
) -> OrderGeometry:
    """Pick the stop, then a structural target or an EMA-21 exit.

    ``exit_mode`` is ``ema_21`` or ``structural``. A missing ``ema_21`` does
    not fall back to a structural target. No survivor means no order.
    """
    mode = "ema_21" if exit_mode == "ema_21" else "structural"
    state = manipulation_state(feat_row)
    buf = stop_buffer(buffer_pts)
    stops, stop_audit = stop_menu(
        strategy,
        direction=direction,
        entry_price=float(entry_price),
        row=feat_row,
        min_dist=float(min_dist),
        buffer=buf,
        manipulation=state,
        htf=htf,
        price_increment=price_increment,
    )
    empty = OrderGeometry(
        None,
        None,
        "",
        "",
        buf,
        None,
        mode,
        state,
        len(stops),
        0,
        True,
        "sl",
    )
    if not stops:
        return _with_menus(empty, stops, [], stop_audit=stop_audit)
    anchor = None
    anchor_dist = None
    for name, price, dist in stops:
        if name == "sl_reference":
            anchor = price
            anchor_dist = dist
            break
    if anchor is None and stops:
        anchor = stops[0][1]
        anchor_dist = stops[0][2]
    if anchor_dist is not None and anchor_dist > 1e-12:
        desired_dist = float(np.clip(sl_fraction, 0.0, float("inf"))) * anchor_dist
        sl_at = min(range(len(stops)), key=lambda i: abs(stops[i][2] - desired_dist))
    else:
        sl_at = 0
    sl_name, sl_price, sl_dist = stops[sl_at]
    for candidate in stop_audit:
        candidate["selected"] = candidate["status"] == "eligible" and candidate["name"] == sl_name
    source_prices: dict[str, float] = {}
    targets = target_menu(
        direction=direction,
        entry_price=float(entry_price),
        row=feat_row,
        stop_dist=sl_dist,
        max_tp_distance=max_tp_distance,
        rr_bounds=rr_bounds,
        htf=htf,
        front_run_pts=target_front_run_pts,
        price_increment=price_increment,
        source_prices=source_prices,
    )
    tp_fallback_reason: str | None = None
    if rr_bounds is not None and targets and sl_dist > 1e-12:
        rr_lo, rr_hi = map(float, rr_bounds)
        if not any(rr_lo <= dist / sl_dist <= rr_hi for _, _, dist in targets):
            tp_fallback_reason = "no_structural_reference_within_rr_window"
    if mode == "ema_21":
        ema = finite_level(feat_row, "ema_21")
        if ema is None:
            return _with_menus(
                OrderGeometry(
                    None,
                    None,
                    "",
                    "",
                    buf,
                    None,
                    mode,
                    state,
                    len(stops),
                    len(targets),
                    True,
                    "ema",
                ),
                stops,
                targets,
                row=feat_row,
                source_prices=source_prices,
                fallback_reason=tp_fallback_reason,
                stop_audit=stop_audit,
            )
        return _with_menus(
            OrderGeometry(
                sl_price,
                None,
                sl_name,
                "ema_21",
                buf,
                None,
                mode,
                state,
                len(stops),
                len(targets),
                False,
                "",
                sl_at,
                -1,
            ),
            stops,
            targets,
            row=feat_row,
            source_prices=source_prices,
            fallback_reason=tp_fallback_reason,
            stop_audit=stop_audit,
        )
    if not targets:
        return _with_menus(
            OrderGeometry(
                None,
                None,
                "",
                "",
                buf,
                None,
                mode,
                state,
                len(stops),
                0,
                True,
                "tp",
            ),
            stops,
            [],
            stop_audit=stop_audit,
        )
    if allow_multi_tp or tp_selections is not None:
        raw1, raw2, raw3 = (
            tp_selections if tp_selections is not None else (tp_fraction, tp_fraction, tp_fraction)
        )
        tp1_idx, tp2_idx, tp3_idx = decode_tp_selections(raw1, raw2, raw3, len(targets))
        active_indices = [idx for idx in (tp1_idx, tp2_idx, tp3_idx) if idx is not None]
        if not active_indices:
            return _with_menus(
                OrderGeometry(
                    None,
                    None,
                    "",
                    "",
                    buf,
                    None,
                    mode,
                    state,
                    len(stops),
                    0,
                    True,
                    "tp",
                ),
                stops,
                [],
                stop_audit=stop_audit,
            )
        tp1_name, tp1_px = (
            (targets[tp1_idx][0], float(targets[tp1_idx][1])) if tp1_idx is not None else ("", None)
        )
        tp2_name, tp2_px = (
            (targets[tp2_idx][0], float(targets[tp2_idx][1])) if tp2_idx is not None else ("", None)
        )
        tp3_name, tp3_px = (
            (targets[tp3_idx][0], float(targets[tp3_idx][1])) if tp3_idx is not None else ("", None)
        )

        if tp_z is not None and allow_simplex:
            fractions = decode_tp_fractions(tp_z[0], tp_z[1], (tp1_idx, tp2_idx, tp3_idx))
        else:
            fractions = tuple(1.0 / len(active_indices) for _ in active_indices)

        nearest_px = tp1_px if tp1_px is not None else (tp2_px if tp2_px is not None else tp3_px)
        nearest_name = tp1_name if tp1_name else (tp2_name if tp2_name else tp3_name)
        nearest_idx = (
            tp1_idx if tp1_idx is not None else (tp2_idx if tp2_idx is not None else tp3_idx)
        )
        nearest_dist = (
            abs(float(nearest_px) - float(entry_price)) if nearest_px is not None else 0.0
        )

        return _with_menus(
            OrderGeometry(
                sl_price,
                nearest_px,
                sl_name,
                nearest_name,
                buf,
                nearest_dist / sl_dist if sl_dist > 1e-12 else None,
                mode,
                state,
                len(stops),
                len(targets),
                False,
                "",
                sl_at,
                nearest_idx if nearest_idx is not None else -1,
                (),
                (),
                tp1_px,
                tp2_px,
                tp3_px,
                tp1_name,
                tp2_name,
                tp3_name,
                fractions,
                (tp1_idx, tp2_idx, tp3_idx),
            ),
            stops,
            targets,
            row=feat_row,
            source_prices=source_prices,
            fallback_reason=tp_fallback_reason,
            stop_audit=stop_audit,
        )

    tp_at = select_index(tp_fraction, len(targets))
    tp_name, tp_price, tp_dist = targets[tp_at]
    return _with_menus(
        OrderGeometry(
            sl_price,
            tp_price,
            sl_name,
            tp_name,
            buf,
            tp_dist / sl_dist,
            mode,
            state,
            len(stops),
            len(targets),
            False,
            "",
            sl_at,
            tp_at,
            (),
            (),
            None,
            None,
            tp_price,
            "",
            "",
            tp_name,
            (1.0,),
            (None, None, tp_at),
        ),
        stops,
        targets,
        row=feat_row,
        source_prices=source_prices,
        fallback_reason=tp_fallback_reason,
        stop_audit=stop_audit,
    )


def reference_timeframe(name: str) -> str:
    """Timeframe prefix of a level name. Unprefixed names are M1."""
    head = str(name).split("|", 1)[0]
    for prefix in ("H1_", "M15_", "M5_"):
        if head.startswith(prefix):
            return prefix[:-1]
    return "M1" if head else ""


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
    htf: tuple[str, ...] = ("M5", "M15", "H1"),
) -> tuple[float | None, float | None, bool]:
    """Return ``(sl, tp, rr_rejected)`` for a structural target.

    ``rr_rejected`` is true when every profit-side price was inside the stop
    or beyond a finite session reach. Dollar size is left to the caller.
    """
    placed = place_strategy_order(
        strategy,
        direction=direction,
        entry_price=entry_price,
        feat_row=feat_row,
        min_dist=min_dist,
        sl_fraction=sl_anchor,
        tp_fraction=reward_fraction,
        exit_mode="structural",
        buffer_pts=buffer_pts,
        max_tp_distance=max_tp_distance,
        htf=htf,
    )
    if placed.rejected or placed.sl_price is None or placed.tp_price is None:
        return None, None, placed.reject_kind == "tp"
    return float(placed.sl_price), float(placed.tp_price), False


__all__ = [
    "OrderGeometry",
    "TargetReference",
    "dynamic_tp",
    "filter_sl_candidates",
    "manipulation_state",
    "place_strategy_order",
    "reference_timeframe",
    "resolve_strategy_entry",
    "select_index",
    "stop_buffer",
]
