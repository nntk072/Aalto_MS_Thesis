"""Risk-based position sizing and SL/TP calculation.

Core formulas:
- risk_usd = equity * risk_frac
- sl_distance = |entry - sl_price|
- lots = risk_usd / (sl_distance * contract_size)
- tp_price = entry ± rr_ratio * sl_distance
"""
from __future__ import annotations

import numpy as np


def compute_sl_price(
    entry_price: float,
    direction: int,
    last_swing_low: float | None,
    last_swing_high: float | None,
    buffer_pts: float = 1.0,
) -> float:
    """Compute SL price from swing structure.
    
    Parameters
    ----------
    entry_price : float
        Fill price at entry.
    direction : int
        +1 for long, -1 for short.
    last_swing_low : float | None
        Most recent swing low price (for long SL).
    last_swing_high : float | None
        Most recent swing high price (for short SL).
    buffer_pts : float
        Safety offset beyond swing (default 1.0 pt).
    
    Returns
    -------
    float
        SL price level. If no swing available, uses entry ± buffer as fallback.
    """
    if direction == 1:  # Long
        if last_swing_low is not None and not np.isnan(last_swing_low):
            return last_swing_low - buffer_pts
        else:
            return entry_price - 2.0  # Fallback
    else:  # Short
        if last_swing_high is not None and not np.isnan(last_swing_high):
            return last_swing_high + buffer_pts
        else:
            return entry_price + 2.0  # Fallback


def compute_tp_price(
    entry_price: float,
    sl_price: float,
    direction: int,
    rr_ratio: float = 2.0,
) -> float:
    """Compute TP price from entry, SL, and R:R ratio.
    
    Parameters
    ----------
    entry_price : float
        Fill price at entry.
    sl_price : float
        Stop loss price level.
    direction : int
        +1 for long, -1 for short.
    rr_ratio : float
        Risk-to-Reward ratio (TP distance / SL distance).
    
    Returns
    -------
    float
        TP price level.
    """
    r = abs(entry_price - sl_price)
    if direction == 1:  # Long
        return entry_price + rr_ratio * r
    else:  # Short
        return entry_price - rr_ratio * r


def compute_lots(
    equity: float,
    risk_frac: float,
    sl_distance: float,
    contract_size: float = 1.0,
    max_loss_usd: float | None = None,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
) -> float:
    """Compute position size (lots) from equity risk and SL distance.
    
    Parameters
    ----------
    equity : float
        Current account equity in USD.
    risk_frac : float
        Risk as fraction of equity (e.g. 0.01 = 1%).
    sl_distance : float
        Distance from entry to SL in price units.
    contract_size : float
        Contract size multiplier (default 1.0).
    max_loss_usd : float | None
        Hard cap on per-trade loss (e.g. 100.0). If given, lots are capped.
    min_lot : float
        Minimum lot size (default 0.01).
    max_lot : float
        Maximum lot size (default 100.0).
    
    Returns
    -------
    float
        Position size in lots, clipped to [min_lot, max_lot].
    """
    risk_usd = equity * risk_frac
    
    if sl_distance <= 0 or np.isnan(sl_distance):
        return min_lot
    
    # Base calculation
    lots = risk_usd / (sl_distance * contract_size)
    
    # Apply hard USD cap if given
    if max_loss_usd is not None and max_loss_usd > 0:
        max_lots_from_cap = max_loss_usd / (sl_distance * contract_size)
        lots = min(lots, max_lots_from_cap)
    
    # Clip to configured bounds
    lots = np.clip(lots, min_lot, max_lot)
    
    return float(lots)
