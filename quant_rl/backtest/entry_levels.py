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

from quant_rl.eval.chart_levels import deviation_levels

_PRICE_TOL = 1e-4


def _tf_col_prefix(tf: str) -> str:
    """Normalise a time-frame key to the uppercase column prefix used by align_timeframes.

    ``DEFAULT_HTF_WINDOWS`` and ``HTF_BRANCHES`` use lowercase keys (``m5``), but
    ``align_timeframes`` prefixes columns with the TF string from ``_htf_list``
    which is uppercase (``M5``).  Without this normalisation, an ``htf`` tuple
    sourced from ``env._mtf_windows.keys()`` would miss every HTF level column.
    """
    return str(tf).upper()


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


def collapse_levels(
    items: list[tuple[str, float]], *, price_increment: float | None = None
) -> list[tuple[str, float]]:
    """Merge prices within ``1e-4`` and join their names with ``|``."""
    if price_increment is not None and (not np.isfinite(price_increment) or price_increment <= 0.0):
        raise ValueError("price_increment must be finite and positive")
    groups: list[tuple[str, float]] = []
    keys: list[int | float] = []
    for name, price in items:
        px = float(price)
        if not np.isfinite(px):
            continue
        key: int | float = int(np.rint(px / price_increment)) if price_increment is not None else px
        merged = False
        for i, (names, kept) in enumerate(groups):
            same_level = (
                key == keys[i] if price_increment is not None else abs(px - kept) <= _PRICE_TOL
            )
            if same_level:
                parts = names.split("|")
                if name not in parts:
                    groups[i] = (names + "|" + name, kept)
                merged = True
                break
        if not merged:
            groups.append((name, px))
            keys.append(key)
    return groups


def manipulation_state(row: pd.Series | Any) -> str:
    """``against`` while manipulation is active, ``done`` after it ends."""
    ended = _flag(row, "po3_manipulation_end")
    if ended:
        return "done"
    if _flag(row, "po3_manipulation_active"):
        return "against"
    return "none"


def stop_buffer(buffer_pts: float) -> float:
    """Stop buffer in price points; ``0.0`` means the stop sits exactly on the level.

    Every caller-facing document already says this — ``default.yaml`` labels
    ``env.sl_buffer_pts`` ``# structural stop buffer (0 = exact)``,
    ``strategy.risk.sl_mode`` reads ``"exact"``, and the TradingEnv docstring
    maps ``0.0`` to ``sl_mode: exact``. The implementation disagreed: it
    substituted ``0.25 * atr_5`` whenever ATR was usable, so an exact stop was
    unreachable and ``env.sl_buffer_pts`` was inert. The config now wins.

    The row argument is gone on purpose: a stop chosen for its structure should
    not be quietly widened by volatility without a config key saying so.
    """
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
    stops, stop_audit = _stop_menu(
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
    targets = _target_menu(
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
        ema = _finite(feat_row, "ema_21")
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
        from quant_rl.envs.tp_decoders import decode_tp_fractions, decode_tp_selections

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


def geometry_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(sl_ref, tp_ref)`` from the open records."""
    columns = [
        "sl_ref",
        "tp_ref",
        "n",
        "planned_rr_median",
        "planned_rr_min",
        "planned_rr_max",
        "realized_r_median",
    ]
    if trades.empty or "sl_ref" not in trades.columns:
        return pd.DataFrame(columns=columns)
    opens = trades
    if "type" in trades.columns:
        opens = trades.loc[trades["type"].astype(str).eq("open")]
    if opens.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = opens.groupby(["sl_ref", "tp_ref"], dropna=False, sort=False)
    for (sl_ref, tp_ref), group in grouped:
        planned = (
            pd.to_numeric(group["planned_rr"], errors="coerce") if "planned_rr" in group else None
        )
        realized = (
            pd.to_numeric(group["realized_r"], errors="coerce") if "realized_r" in group else None
        )
        ema = str(tp_ref) == "ema_21"
        if "exit_mode" in group.columns and group["exit_mode"].astype(str).eq("ema_21").all():
            ema = True
        rows.append(
            {
                "sl_ref": sl_ref,
                "tp_ref": tp_ref,
                "n": int(len(group)),
                "planned_rr_median": np.nan if ema or planned is None else float(planned.median()),
                "planned_rr_min": np.nan if ema or planned is None else float(planned.min()),
                "planned_rr_max": np.nan if ema or planned is None else float(planned.max()),
                "realized_r_median": np.nan if realized is None else float(realized.median()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _stop_menu(
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
        tf_up = _tf_col_prefix(str(tf))
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
                level = _finite(row, name)
                if level is None:
                    continue
                pivot_ms = _finite(row, f"{name}_pivot_time_ms")
                confirmed_ms = _finite(row, f"{name}_confirmed_at_ms")
                age_bars = _finite(row, f"{name}_age_bars")
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
            level = _finite(row, f"{tf_up}_{swing}")
            if level is not None:
                raw.append((f"{tf_up}_{swing}", level))
    if direction == 1:
        side_names = [
            "prev_day_low",
            "ctx_prev_day_low",
            "ctx_prev_week_low",
            "yesterday_low",
            "lastweek_low",
        ]
    else:
        side_names = [
            "prev_day_high",
            "ctx_prev_day_high",
            "ctx_prev_week_high",
            "yesterday_high",
            "lastweek_high",
        ]
    for name in side_names:
        level = _finite(row, name)
        if level is not None:
            raw.append((name, level))
    raw.extend(_deviations(row))
    smt_tfs = [_tf_col_prefix(str(tf)) for tf in htf]
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
    for name, level in collapse_levels(
        [(n, float(p)) for n, p in raw],
        price_increment=price_increment if candidate_schema_active else None,
    ):
        stop = level - buffer if direction == 1 else level + buffer
        if direction == 1 and stop >= entry_price:
            audit.append(
                {
                    "name": name,
                    "reference_price": level,
                    "stop_price": stop,
                    "distance": abs(entry_price - stop),
                    "status": "rejected",
                    "reason": "wrong_side",
                }
            )
            continue
        if direction == -1 and stop <= entry_price:
            audit.append(
                {
                    "name": name,
                    "reference_price": level,
                    "stop_price": stop,
                    "distance": abs(entry_price - stop),
                    "status": "rejected",
                    "reason": "wrong_side",
                }
            )
            continue
        dist = abs(entry_price - stop)
        if dist + 1e-12 < min_dist:
            audit.append(
                {
                    "name": name,
                    "reference_price": level,
                    "stop_price": stop,
                    "distance": dist,
                    "status": "rejected",
                    "reason": "below_min_distance",
                }
            )
            continue
        menu.append((name, float(stop), dist))
        audit.append(
            {
                "name": name,
                "reference_price": level,
                "stop_price": stop,
                "distance": dist,
                "status": "eligible",
                "reason": "",
            }
        )
    for item in audit:
        names = str(item.get("name", "")).split("|")
        provenance = [candidate_meta[name] for name in names if name in candidate_meta]
        if provenance:
            item["provenance"] = provenance
    menu.sort(key=lambda item: item[2])
    return menu, audit


def _target_menu(
    *,
    direction: int,
    entry_price: float,
    row: Any,
    stop_dist: float,
    max_tp_distance: float,
    rr_bounds: tuple[float, float] | None = None,
    htf: tuple[str, ...] = ("M5", "M15", "H1"),
    front_run_pts: float = 0.0,
    price_increment: float | None = None,
    source_prices: dict[str, float] | None = None,
) -> list[tuple[str, float, float]]:
    if not np.isfinite(front_run_pts) or front_run_pts < 0.0:
        raise ValueError("front_run_pts must be finite and nonnegative")
    if price_increment is not None and (not np.isfinite(price_increment) or price_increment <= 0.0):
        raise ValueError("price_increment must be finite and positive")
    if direction == 1:
        names = [
            "last_swing_high",
            "asian_high",
            "london_high",
            "prev_day_high",
            "ctx_prev_week_high",
            "sweep_high_level",
        ]
        gap_suffixes = ("ifvg_bear_low", "fvg_bear_low")
        swing_name = "last_swing_high"
    else:
        names = [
            "last_swing_low",
            "asian_low",
            "london_low",
            "prev_day_low",
            "ctx_prev_week_low",
            "sweep_low_level",
        ]
        gap_suffixes = ("ifvg_bull_high", "fvg_bull_high")
        swing_name = "last_swing_low"
    raw: list[tuple[str, float]] = []
    for name in names:
        level = _finite(row, name)
        if level is not None:
            raw.append((name, level))
    for tf in htf:
        tf_up = _tf_col_prefix(str(tf))
        level = _finite(row, f"{tf_up}_{swing_name}")
        if level is not None:
            raw.append((f"{tf_up}_{swing_name}", level))
    raw.extend(_suffixed(row, gap_suffixes))
    raw.extend(_deviations(row))
    if source_prices is not None:
        source_prices.update((name, float(price)) for name, price in raw)
    # A known confirmation after the decision is never eligible. The pivot
    # origin (`*_time`) is not a confirmation time and is not used here.
    decision_at = getattr(row, "name", None)
    if not isinstance(decision_at, pd.Timestamp):
        decision_at = None
    available: list[tuple[str, float]] = []
    for name, level in raw:
        confirmed_at = row.get(f"{name}_confirmed_at") if hasattr(row, "get") else None
        if confirmed_at is not None and not pd.isna(confirmed_at):
            if decision_at is None:
                continue  # explicit availability requires an actual decision clock
            try:
                if pd.Timestamp(confirmed_at) > decision_at:
                    continue
            except (TypeError, ValueError):
                continue  # fail closed on invalid or timezone-incompatible metadata
        available.append((name, level))
    room = float(max_tp_distance) if max_tp_distance is not None else float("nan")
    ranked: list[tuple[str, float, float, bool]] = []
    rr_lo, rr_hi = (
        (float(rr_bounds[0]), float(rr_bounds[1]))
        if rr_bounds is not None
        else (float("-inf"), float("inf"))
    )
    for name, level in collapse_levels(available):
        if front_run_pts:
            level = float(level) - direction * float(front_run_pts)
        if price_increment is not None:
            units = level / price_increment
            if direction == 1:
                level = float(np.floor(units + 1e-10) * price_increment)
            else:
                level = float(np.ceil(units - 1e-10) * price_increment)
        if direction == 1 and level <= entry_price:
            continue
        if direction == -1 and level >= entry_price:
            continue
        dist = abs(level - entry_price)
        if dist + 1e-12 < stop_dist:
            continue
        if np.isfinite(room) and dist > room + 1e-9:
            continue
        # strategy.risk.rr_range is the promised reward-to-risk window. It was
        # read into TradingEnv.rr_ratio_range and then never applied to the
        # menu, so ~44% of trades settled outside [1.5, 5.0].
        implied = dist / stop_dist if stop_dist > 1e-12 else float("inf")
        ranked.append((name, float(level), dist, rr_lo - 1e-9 <= implied <= rr_hi + 1e-9))
    # Enforce the window whenever the bar offers at least one level inside it.
    # Falling back when nothing qualifies keeps the menu usable instead of
    # rejecting a setup purely because no structural level happened to land in
    # the window — a stop that exists is still better than no trade at all.
    legal = [item for item in ranked if item[3]]
    chosen = legal or ranked
    menu = [(name, level, dist) for name, level, dist, _in_range in chosen]
    if price_increment is not None or front_run_pts:
        # Different original references can map to the same executable tick.
        deduped = collapse_levels([(name, level) for name, level, _ in menu])
        menu = [(name, px, abs(px - entry_price)) for name, px in deduped]
    menu.sort(key=lambda item: item[2])
    return menu


def _deviations(row: Any) -> list[tuple[str, float]]:
    lo = _finite(row, "last_swing_low")
    hi = _finite(row, "last_swing_high")
    if lo is None or hi is None:
        lo = _finite(row, "asian_low")
        hi = _finite(row, "asian_high")
    if lo is None or hi is None or hi <= lo:
        return []
    out: list[tuple[str, float]] = []
    for k, price in deviation_levels(lo, hi, lo, hi):
        out.append((f"dev_{k:g}", float(price)))
    return out


def _suffixed(row: Any, suffixes: tuple[str, ...]) -> list[tuple[str, float]]:
    index = getattr(row, "index", ())
    found: list[tuple[str, float]] = []
    for col in index:
        text = str(col)
        if any(text == suffix or text.endswith("_" + suffix) for suffix in suffixes):
            level = _finite(row, text)
            if level is not None:
                found.append((text, level))
    return found


def _finite(row: Any, name: str) -> float | None:
    index = getattr(row, "index", None)
    if index is not None and name not in index:
        return None
    try:
        value = float(row.get(name, np.nan))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _flag(row: Any, name: str) -> bool:
    value = _finite(row, name)
    return value is not None and value > 0.0
