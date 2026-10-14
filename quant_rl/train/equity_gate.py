"""Train-year survival gate, with profitability retained as reported metrics.

The replay fails on a risk-limit breach or excessive peak trailing drawdown.
Net PnL and equity slope are recorded but are not acceptance requirements.
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


def decide_early_abort(
    episodes: list[dict[str, Any]],
    *,
    num_timesteps: int,
    min_timesteps: int,
    window: int,
    nonfinite: bool,
    stop_on_nonfinite: bool = True,
    stop_on_no_trades: bool = False,
    stop_on_equity_gate: bool = True,
) -> str | None:
    """Return a stop reason, or ``None`` to keep training.

    Non-finite metrics stop immediately. Optional no-trade stops and survival
    gate stops wait until ``min_timesteps`` and require every episode in the
    last ``window`` to fail. One episode that passes survival keeps training.
    """
    if stop_on_nonfinite and nonfinite:
        return "nonfinite"
    if int(num_timesteps) < int(min_timesteps) or int(window) < 1:
        return None
    if len(episodes) < int(window):
        return None
    recent = episodes[-int(window) :]
    if stop_on_no_trades and all(int(ep.get("n_trades", 0)) == 0 for ep in recent):
        return "no_trades"
    if stop_on_equity_gate and all(not _episode_passes_gate(ep) for ep in recent):
        reasons = ",".join(_episode_gate(ep)["reason"] for ep in recent)
        return f"equity_gate:{reasons}"
    return None


def _episode_gate(ep: dict[str, Any]) -> dict[str, Any]:
    return assess_equity_snapshot(
        end_equity=float(ep.get("end_equity", ep.get("start_equity", 0.0))),
        initial=float(ep.get("start_equity", 0.0)),
        slope=float(ep.get("slope", 0.0)),
        max_peak_trailing_dd=float(ep.get("max_peak_trailing_dd", 0.0)),
        breached=bool(ep.get("breach_reason") or ep.get("breached")),
    )


def _episode_passes_gate(ep: dict[str, Any]) -> bool:
    return bool(_episode_gate(ep)["ok"])


def _max_peak_trailing_dd(equity: pd.Series | np.ndarray[Any, Any]) -> float:
    """Largest peak-to-trough drawdown on the path, as a fraction of the peak."""
    y = np.asarray(equity, dtype=float)
    if y.size == 0:
        return 0.0
    peak = np.maximum.accumulate(y)
    return float(np.max((peak - y) / np.maximum(peak, 1e-9)))
