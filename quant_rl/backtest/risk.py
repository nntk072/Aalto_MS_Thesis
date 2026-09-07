"""Risk-based SL/TP/lot sizing calculations.

Computes stop loss, take profit, and lot sizes based on trade structure
and equity risk parameters, following MT5-style risk management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sl_tp_long(
    entry_price: float,
    last_swing_low: float,
    buffer_pts: float = 1.0,
    rr_ratio: float = 2.0,
) -> tuple[float, float]:
    """Compute SL and TP for a long trade.

    Parameters
    ----------
    entry_price : float
        Entry fill price.
    last_swing_low : float
        Most recent confirmed swing low price.
    buffer_pts : float
        Buffer below swing low for SL placement (in price points).
    rr_ratio : float
        Risk:reward ratio (e.g. 2.0 means TP distance = 2 × SL distance).

    Returns
    -------
    tuple[float, float]
        (sl_price, tp_price)

    Raises
    ------
    ValueError
        If ``last_swing_low`` is not strictly below ``entry_price``.
    """
    if last_swing_low >= entry_price:
        raise ValueError(
            f"Invalid long structure: swing low {last_swing_low} must be below entry {entry_price}"
        )
    sl_price = last_swing_low - buffer_pts
    r = entry_price - sl_price
    tp_price = entry_price + rr_ratio * r
    return sl_price, tp_price


def compute_sl_tp_short(
    entry_price: float,
    last_swing_high: float,
    buffer_pts: float = 1.0,
    rr_ratio: float = 2.0,
) -> tuple[float, float]:
    """Compute SL and TP for a short trade.

    Parameters
    ----------
    entry_price : float
        Entry fill price.
    last_swing_high : float
        Most recent confirmed swing high price.
    buffer_pts : float
        Buffer above swing high for SL placement (in price points).
    rr_ratio : float
        Risk:reward ratio.

    Returns
    -------
    tuple[float, float]
        (sl_price, tp_price)

    Raises
    ------
    ValueError
        If ``last_swing_high`` is not strictly above ``entry_price``.
    """
    if last_swing_high <= entry_price:
        raise ValueError(
            f"Invalid short structure: swing high {last_swing_high} must be above entry {entry_price}"
        )
    sl_price = last_swing_high + buffer_pts
    r = sl_price - entry_price
    tp_price = entry_price - rr_ratio * r
    return sl_price, tp_price


def compute_lots(
    equity: float,
    risk_frac: float,
    entry_price: float,
    sl_price: float,
    contract_size: float = 1.0,
    point_value: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    max_loss_cap: float | None = None,
) -> float:
    """Compute lot size from risk budget and SL distance.

    Parameters
    ----------
    equity : float
        Current account equity.
    risk_frac : float
        Fraction of equity at risk (e.g. 0.01 for 1%).
    entry_price : float
        Entry price.
    sl_price : float
        Stop loss price.
    contract_size : float
        Contract multiplier (default 1.0).
    point_value : float
        Dollar value per point per contract (default 1.0).
    min_lot : float
        Minimum lot size to trade.
    max_lot : float
        Maximum lot size to trade.
    max_loss_cap : float | None
        If set, cap the USD loss at this amount (e.g. 100 for $100).

    Returns
    -------
    float
        Computed lot size, clipped to [min_lot, max_lot].
    """
    risk_usd = equity * risk_frac
    sl_distance = abs(entry_price - sl_price)

    if sl_distance < 1e-8:
        # Avoid division by near-zero
        return min_lot

    lots = risk_usd / (sl_distance * contract_size * point_value)

    # Apply safety cap if configured
    if max_loss_cap is not None:
        max_lots_from_cap = max_loss_cap / (sl_distance * contract_size * point_value)
        lots = min(lots, max_lots_from_cap)

    # Clip to [min_lot, max_lot]
    lots = np.clip(lots, min_lot, max_lot)

    return float(lots)


def compute_sl_tp_from_structure(
    *,
    direction: int,
    entry_price: float,
    structure_level: float,
    rr_ratio: float,
    buffer_pts: float = 0.0,
) -> tuple[float, float]:
    """Compute structural SL/TP for Idea 1/Idea 2 trades (Agent.md §14).

    Long: SL = structure_level - buffer. Short: SL = structure_level + buffer.
    ``buffer_pts=0.0`` implements the ``sl_mode: exact`` configuration;
    a positive buffer implements ``sl_mode: buffered``.

    Raises
    ------
    ValueError
        If ``direction`` is not +/-1 or the geometry is invalid (the level is
        on the wrong side of the entry), so a backwards stop is never created.
    """
    if direction == 1:
        return compute_sl_tp_long(entry_price, structure_level, buffer_pts, rr_ratio)
    if direction == -1:
        return compute_sl_tp_short(entry_price, structure_level, buffer_pts, rr_ratio)
    raise ValueError(f"direction must be +1 or -1, got {direction}")


# Structural level columns probed for each liquidity target mode, in priority
# order. The first finite level wins; missing columns are skipped.
_TARGET_LEVELS: dict[str, dict[int, tuple[str, ...]]] = {
    "buyside_liquidity": {1: ("sweep_high_level", "london_high", "asian_high"), -1: ()},
    "sellside_liquidity": {1: (), -1: ("sweep_low_level", "london_low", "asian_low")},
    "previous_day_high_low": {1: ("prev_day_high",), -1: ("prev_day_low",)},
}


def _structural_target(
    direction: int, target_mode: str, row: pd.Series, entry_price: float
) -> float:
    """Return a valid structural target for the direction, or NaN."""
    per_dir = _TARGET_LEVELS.get(target_mode)
    if not per_dir:
        raise ValueError(f"unknown target_mode: {target_mode!r}")
    for col in per_dir.get(direction, ()):
        if col in row.index:
            level = float(row[col])
            if np.isfinite(level) and (
                level > entry_price if direction == 1 else level < entry_price
            ):
                return level
    return float("nan")


def resolve_tp_target(
    *,
    direction: int,
    entry_price: float,
    sl_price: float,
    rr_ratio: float,
    target_mode: str,
    row: pd.Series,
) -> float:
    """Resolve the take-profit price for a trade (Agent.md §15).

    Supported modes: ``rr``, ``buyside_liquidity``, ``sellside_liquidity``,
    ``previous_day_high_low``. Structural targets that are NaN or on the
    wrong side of the entry fall back to the RR target (``rr_fallback``
    policy) — an invalid TP is never silently created.

    Raises
    ------
    ValueError
        If ``target_mode`` is unknown or the RR fallback itself is invalid.
    """
    risk = (entry_price - sl_price) if direction == 1 else (sl_price - entry_price)
    rr_tp = entry_price + direction * rr_ratio * risk
    if risk > 0 and direction in (1, -1):
        fallback_ok = True
    else:
        fallback_ok = False

    if target_mode == "rr":
        if not fallback_ok:
            raise ValueError(f"invalid RR geometry: risk {risk} must be positive")
        return rr_tp

    level = _structural_target(direction, target_mode, row, entry_price)
    if np.isfinite(level):
        return level
    if not fallback_ok:
        raise ValueError(
            f"invalid target {target_mode!r} and RR fallback unavailable (risk {risk})"
        )
    return rr_tp
