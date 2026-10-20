"""Train-year survival gate, with profitability retained as reported metrics.

The replay fails on a risk-limit breach or excessive peak trailing drawdown.
Net PnL and equity slope are recorded but are not acceptance requirements.
"""

from __future__ import annotations

from typing import Any

from simulation.environment.episode_equity import (
    assess_equity_snapshot,
    assess_train_equity,
    equity_slope,
)

__all__ = ["assess_train_equity", "decide_early_abort", "equity_slope"]


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
