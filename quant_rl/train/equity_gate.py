"""Train-year equity gate: net rise across the trading calendar.

Losing trades and losing days are allowed. The replay fails when the
account does not finish above its start, the equity-versus-time slope is
not positive, or a max-loss breach stops trading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def equity_slope(equity: pd.Series | np.ndarray[Any, Any]) -> float:
    """Least-squares slope of equity against bar order. Flat is 0."""
    y = np.asarray(equity, dtype=float)
    if y.size < 2 or not np.isfinite(y).all():
        return 0.0
    x = np.arange(y.size, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def assess_train_equity(
    equity: pd.Series,
    *,
    initial: float,
    breached: bool,
) -> dict[str, Any]:
    """Return the train-year gate result.

    ``ok`` is true only when end equity is above ``initial``, the path
    slopes up, and trading was not stopped by a loss-cap breach.
    """
    if equity is None or len(equity) == 0:
        return {
            "ok": False,
            "end_equity": float(initial),
            "equity_delta": 0.0,
            "slope": 0.0,
            "breached": bool(breached),
            "reason": "empty_equity",
        }
    end = float(np.asarray(equity, dtype=float)[-1])
    slope = equity_slope(equity)
    delta = end - float(initial)
    reasons: list[str] = []
    if not (end > float(initial)):
        reasons.append("end_not_above_start")
    if not (slope > 0.0):
        reasons.append("slope_not_positive")
    trail = _max_peak_trailing_dd(equity)
    if trail > 0.10:
        reasons.append("peak_trailing_dd")
    if breached:
        reasons.append("max_loss_stopped_trading")
    return {
        "ok": not reasons,
        "end_equity": end,
        "equity_delta": delta,
        "slope": slope,
        "max_peak_trailing_dd": trail,
        "breached": bool(breached),
        "reason": "ok" if not reasons else ",".join(reasons),
    }


def _max_peak_trailing_dd(equity: pd.Series | np.ndarray[Any, Any]) -> float:
    """Largest peak-to-trough drawdown on the path, as a fraction of the peak."""
    y = np.asarray(equity, dtype=float)
    if y.size == 0:
        return 0.0
    peak = np.maximum.accumulate(y)
    return float(np.max((peak - y) / np.maximum(peak, 1e-9)))
