"""Episode equity diagnostics shared by simulation and training reports."""

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


def assess_equity_snapshot(
    *,
    end_equity: float,
    initial: float,
    slope: float,
    max_peak_trailing_dd: float,
    breached: bool,
    peak_trailing_dd_limit: float = 0.0,
) -> dict[str, Any]:
    """Evaluate configured survival limits and retain all equity diagnostics."""
    end = float(end_equity)
    start = float(initial)
    drawdown = float(max_peak_trailing_dd)
    drawdown_limit = float(peak_trailing_dd_limit)
    reasons: list[str] = []
    if drawdown_limit > 0.0 and drawdown > drawdown_limit:
        reasons.append("peak_trailing_dd")
    if breached:
        reasons.append("max_loss_stopped_trading")
    return {
        "ok": not reasons,
        "end_equity": end,
        "equity_delta": end - start,
        "slope": float(slope),
        "max_peak_trailing_dd": drawdown,
        "peak_trailing_dd_limit": drawdown_limit,
        "breached": bool(breached),
        "reason": "ok" if not reasons else ",".join(reasons),
    }


def assess_train_equity(
    equity: pd.Series,
    *,
    initial: float,
    breached: bool,
    peak_trailing_dd_limit: float = 0.0,
) -> dict[str, Any]:
    """Return the hard loss-cap survival gate plus PnL/drawdown metrics.

    ``peak_trailing_dd_limit`` activates the same drawdown policy used by the
    environment. With a zero limit, measured drawdown is report-only. An actual
    hard loss-cap breach always fails this survival check.
    """
    if equity is None or len(equity) == 0:
        return {
            "ok": False,
            "end_equity": float(initial),
            "equity_delta": 0.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.0,
            "peak_trailing_dd_limit": float(peak_trailing_dd_limit),
            "breached": bool(breached),
            "reason": "empty_equity",
        }
    end = float(np.asarray(equity, dtype=float)[-1])
    return assess_equity_snapshot(
        end_equity=end,
        initial=float(initial),
        slope=equity_slope(equity),
        max_peak_trailing_dd=_max_peak_trailing_dd(equity),
        peak_trailing_dd_limit=float(peak_trailing_dd_limit),
        breached=bool(breached),
    )


def _max_peak_trailing_dd(equity: pd.Series | np.ndarray[Any, Any]) -> float:
    """Largest peak-to-trough drawdown on the path, as a fraction of the peak."""
    y = np.asarray(equity, dtype=float)
    if y.size == 0:
        return 0.0
    peak = np.maximum.accumulate(y)
    return float(np.max((peak - y) / np.maximum(peak, 1e-9)))
