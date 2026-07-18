"""Risk-based SL/TP/lot sizing calculations.

Computes stop loss, take profit, and lot sizes based on trade structure
and equity risk parameters, following MT5-style risk management.
"""
from __future__ import annotations

import numpy as np


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
    """
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
    """
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

    lots = risk_usd / (sl_distance * contract_size)

    # Apply safety cap if configured
    if max_loss_cap is not None:
        max_lots_from_cap = max_loss_cap / (sl_distance * contract_size)
        lots = min(lots, max_lots_from_cap)

    # Clip to [min_lot, max_lot]
    lots = np.clip(lots, min_lot, max_lot)

    return float(lots)
