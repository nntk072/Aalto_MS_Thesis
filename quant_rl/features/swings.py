"""Confirmed fractal pivots and ATR-filtered zigzag swings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import atr as compute_atr

TIMEFRAME_CONFIG: dict[str, dict[str, float | int]] = {
    "M1": {"left": 3, "right": 3, "atr_mult": 0.8, "atr_period": 14},
    "M5": {"left": 3, "right": 3, "atr_mult": 0.7, "atr_period": 14},
    "M15": {"left": 4, "right": 4, "atr_mult": 0.6, "atr_period": 14},
    "M30": {"left": 4, "right": 4, "atr_mult": 0.55, "atr_period": 14},
    "H1": {"left": 4, "right": 4, "atr_mult": 0.5, "atr_period": 14},
    "H4": {"left": 5, "right": 5, "atr_mult": 0.5, "atr_period": 14},
    "D1": {"left": 5, "right": 5, "atr_mult": 0.5, "atr_period": 14},
}


def detect_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Detect fractal pivots; features populate only from the confirmation bar.

    A pivot at bar ``i`` is known at bar ``i + right``. Structural price is
    close; liquidity extreme is high/low.

    Args:
        df: OHLC frame with a DatetimeIndex.
        left: Bars before the pivot that must be strictly lower/higher.
        right: Confirmation bars after the pivot.

    Returns:
        Frame with pivot high/low event, price, extreme, and location columns.
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    ev_h = np.zeros(n, dtype=bool)
    ev_l = np.zeros(n, dtype=bool)
    price_h = np.full(n, np.nan)
    price_l = np.full(n, np.nan)
    ext_h = np.full(n, np.nan)
    ext_l = np.full(n, np.nan)
    loc_h = np.full(n, np.nan)
    loc_l = np.full(n, np.nan)

    for i in range(left, n - right):
        conf = i + right
        left_c = close[i - left : i]
        right_c = close[i + 1 : i + right + 1]
        if left_c.size and right_c.size and close[i] > left_c.max() and close[i] > right_c.max():
            ev_h[conf] = True
            price_h[conf] = close[i]
            ext_h[conf] = high[i]
            loc_h[conf] = i
        if left_c.size and right_c.size and close[i] < left_c.min() and close[i] < right_c.min():
            ev_l[conf] = True
            price_l[conf] = close[i]
            ext_l[conf] = low[i]
            loc_l[conf] = i

    return pd.DataFrame(
        {
            "pivot_high_event": ev_h,
            "pivot_high_price": price_h,
            "pivot_high_extreme": ext_h,
            "pivot_high_location": loc_h,
            "pivot_low_event": ev_l,
            "pivot_low_price": price_l,
            "pivot_low_extreme": ext_l,
            "pivot_low_location": loc_l,
        },
        index=df.index,
    )


def detect_swings(
    df: pd.DataFrame,
    pivots: pd.DataFrame,
    atr: pd.Series | None = None,
    atr_mult: float = 0.5,
) -> pd.DataFrame:
    """ATR-filtered zigzag on confirmed pivots.

    A candidate is created at fractal confirmation, replaced if a more extreme
    candidate appears before reversal, and accepted when close reverses by
    ``atr_mult * ATR`` frozen at candidate confirmation. When ATR is missing
    the candidate is accepted on its confirmation bar.

    Args:
        df: OHLC frame aligned with ``pivots``.
        pivots: Output of :func:`detect_pivots`.
        atr: ATR series; computed from ``df`` when None.
        atr_mult: Reversal threshold in ATR units.
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    if atr is None:
        atr_arr = compute_atr(df, period=14).to_numpy(dtype=float)
    else:
        atr_arr = atr.reindex(df.index).to_numpy(dtype=float)

    ev_h = pivots["pivot_high_event"].to_numpy(dtype=bool)
    ev_l = pivots["pivot_low_event"].to_numpy(dtype=bool)
    ph = pivots["pivot_high_price"].to_numpy(dtype=float)
    pl = pivots["pivot_low_price"].to_numpy(dtype=float)
    eh = pivots["pivot_high_extreme"].to_numpy(dtype=float)
    el = pivots["pivot_low_extreme"].to_numpy(dtype=float)
    lh = pivots["pivot_high_location"].to_numpy(dtype=float)
    ll = pivots["pivot_low_location"].to_numpy(dtype=float)

    acc_h = np.zeros(n, dtype=bool)
    acc_l = np.zeros(n, dtype=bool)
    out_price_h = np.full(n, np.nan)
    out_price_l = np.full(n, np.nan)
    out_ext_h = np.full(n, np.nan)
    out_ext_l = np.full(n, np.nan)
    out_loc_h = np.full(n, np.nan)
    out_loc_l = np.full(n, np.nan)

    cand_h: dict[str, float] | None = None
    cand_l: dict[str, float] | None = None
    last_side = 0

    def _atr_at(i: int) -> float:
        a = atr_arr[i]
        return float(a) if np.isfinite(a) else 0.0

    for t in range(n):
        if ev_h[t] and last_side != 1:
            cand = {
                "price": float(ph[t]),
                "extreme": float(eh[t]),
                "loc": float(lh[t]),
                "atr": _atr_at(t),
            }
            if cand_h is None or cand["extreme"] >= cand_h["extreme"]:
                cand_h = cand
        if ev_l[t] and last_side != -1:
            cand = {
                "price": float(pl[t]),
                "extreme": float(el[t]),
                "loc": float(ll[t]),
                "atr": _atr_at(t),
            }
            if cand_l is None or cand["extreme"] <= cand_l["extreme"]:
                cand_l = cand

        if cand_h is not None and last_side != 1:
            thresh = atr_mult * cand_h["atr"]
            reversed_h = thresh <= 0.0 or close[t] <= cand_h["price"] - thresh
            if reversed_h:
                acc_h[t] = True
                out_price_h[t] = cand_h["price"]
                out_ext_h[t] = cand_h["extreme"]
                out_loc_h[t] = cand_h["loc"]
                last_side = 1
                cand_h = None
                continue
        if cand_l is not None and last_side != -1:
            thresh = atr_mult * cand_l["atr"]
            reversed_l = thresh <= 0.0 or close[t] >= cand_l["price"] + thresh
            if reversed_l:
                acc_l[t] = True
                out_price_l[t] = cand_l["price"]
                out_ext_l[t] = cand_l["extreme"]
                out_loc_l[t] = cand_l["loc"]
                last_side = -1
                cand_l = None

    cur_ph = cur_eh = cur_lh = np.nan
    cur_pl = cur_el = cur_ll = np.nan
    ffill_h = np.full(n, np.nan)
    ffill_l = np.full(n, np.nan)
    ffill_eh = np.full(n, np.nan)
    ffill_el = np.full(n, np.nan)
    ffill_lh = np.full(n, np.nan)
    ffill_ll = np.full(n, np.nan)
    for t in range(n):
        if acc_h[t]:
            cur_ph, cur_eh, cur_lh = out_price_h[t], out_ext_h[t], out_loc_h[t]
        if acc_l[t]:
            cur_pl, cur_el, cur_ll = out_price_l[t], out_ext_l[t], out_loc_l[t]
        ffill_h[t], ffill_eh[t], ffill_lh[t] = cur_ph, cur_eh, cur_lh
        ffill_l[t], ffill_el[t], ffill_ll[t] = cur_pl, cur_el, cur_ll

    return pd.DataFrame(
        {
            "swing_high_event": acc_h,
            "swing_high_price": ffill_h,
            "swing_high_extreme": ffill_eh,
            "swing_high_location": ffill_lh,
            "swing_low_event": acc_l,
            "swing_low_price": ffill_l,
            "swing_low_extreme": ffill_el,
            "swing_low_location": ffill_ll,
        },
        index=df.index,
    )


def classify_structure(swings: pd.DataFrame) -> pd.DataFrame:
    """Classify accepted swings as HH/HL/LH/LL.

    Args:
        swings: Output of :func:`detect_swings`.

    Returns:
        Frame with ``structure`` (string) plus one-hot HH/HL/LH/LL flags.
    """
    n = len(swings)
    ev_h = swings["swing_high_event"].to_numpy(dtype=bool)
    ev_l = swings["swing_low_event"].to_numpy(dtype=bool)
    px_h = swings["swing_high_price"].to_numpy(dtype=float)
    px_l = swings["swing_low_price"].to_numpy(dtype=float)
    labels = np.empty(n, dtype=object)
    labels[:] = ""
    hh = np.zeros(n)
    hl = np.zeros(n)
    lh = np.zeros(n)
    ll = np.zeros(n)
    prev_h = np.nan
    prev_l = np.nan
    for t in range(n):
        if ev_h[t]:
            if np.isfinite(prev_h):
                if px_h[t] > prev_h:
                    labels[t] = "HH"
                    hh[t] = 1.0
                elif px_h[t] < prev_h:
                    labels[t] = "LH"
                    lh[t] = 1.0
            prev_h = px_h[t]
        if ev_l[t]:
            if np.isfinite(prev_l):
                if px_l[t] > prev_l:
                    labels[t] = "HL"
                    hl[t] = 1.0
                elif px_l[t] < prev_l:
                    labels[t] = "LL"
                    ll[t] = 1.0
            prev_l = px_l[t]
    return pd.DataFrame(
        {
            "structure": labels,
            "hh": hh,
            "hl": hl,
            "lh": lh,
            "ll": ll,
        },
        index=swings.index,
    )


def swing_features(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    structure: pd.DataFrame,
    atr: pd.Series | None = None,
) -> pd.DataFrame:
    """Causal continuous swing features for the agent.

    Last swing means last swing confirmed and observable at the current bar.

    Args:
        df: OHLC frame.
        swings: Output of :func:`detect_swings`.
        structure: Output of :func:`classify_structure`.
        atr: ATR series; computed when None.
    """
    if atr is None:
        atr_s = compute_atr(df, period=14)
    else:
        atr_s = atr.reindex(df.index)
    atr_safe = atr_s.where(atr_s > 0)
    close = df["close"]
    last_h = swings["swing_high_extreme"]
    last_l = swings["swing_low_extreme"]
    dist_h = (last_h - close) / atr_safe
    dist_l = (close - last_l) / atr_safe
    nearest = pd.concat([dist_h.abs(), dist_l.abs()], axis=1).min(axis=1)
    side = pd.Series(np.nan, index=df.index)
    side = side.where(~swings["swing_high_event"], 1.0)
    side = side.where(~swings["swing_low_event"], -1.0)
    side = side.ffill()
    size = (last_h - last_l).abs() / atr_safe
    since_h = _bars_since(swings["swing_high_event"].to_numpy(dtype=bool))
    since_l = _bars_since(swings["swing_low_event"].to_numpy(dtype=bool))
    since = np.minimum(since_h, since_l)
    return pd.DataFrame(
        {
            "dist_to_last_high_atr": dist_h,
            "dist_to_last_low_atr": dist_l,
            "dist_to_nearest_swing_atr": nearest,
            "last_swing_side": side,
            "last_swing_size_atr": size,
            "bars_since_last_swing": since,
            "hh": structure["hh"],
            "hl": structure["hl"],
            "lh": structure["lh"],
            "ll": structure["ll"],
        },
        index=df.index,
    )


def _bars_since(events: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    n = len(events)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if events[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return out
