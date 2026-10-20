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
) -> dict[str, Any]:
    """Evaluate risk survival from scalars; profit and slope remain metrics."""
    end = float(end_equity)
    start = float(initial)
    reasons: list[str] = []
    if float(max_peak_trailing_dd) > 0.10:
        reasons.append("peak_trailing_dd")
    if breached:
        reasons.append("max_loss_stopped_trading")
    return {
        "ok": not reasons,
        "end_equity": end,
        "equity_delta": end - start,
        "slope": float(slope),
        "max_peak_trailing_dd": float(max_peak_trailing_dd),
        "breached": bool(breached),
        "reason": "ok" if not reasons else ",".join(reasons),
    }


def assess_train_equity(
    equity: pd.Series,
    *,
    initial: float,
    breached: bool,
) -> dict[str, Any]:
    """Return the train-year risk-survival gate plus PnL/slope metrics.

    ``ok`` depends only on the 10% peak trailing drawdown cap and loss-cap
    breach. End equity and slope are included in the result for reporting.
    """
    if equity is None or len(equity) == 0:
        return {
            "ok": False,
            "end_equity": float(initial),
            "equity_delta": 0.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.0,
            "breached": bool(breached),
            "reason": "empty_equity",
        }
    end = float(np.asarray(equity, dtype=float)[-1])
    return assess_equity_snapshot(
        end_equity=end,
        initial=float(initial),
        slope=equity_slope(equity),
        max_peak_trailing_dd=_max_peak_trailing_dd(equity),
        breached=bool(breached),
    )


def _max_peak_trailing_dd(equity: pd.Series | np.ndarray[Any, Any]) -> float:
    """Largest peak-to-trough drawdown on the path, as a fraction of the peak."""
    y = np.asarray(equity, dtype=float)
    if y.size == 0:
        return 0.0
    peak = np.maximum.accumulate(y)
    return float(np.max((peak - y) / np.maximum(peak, 1e-9)))
