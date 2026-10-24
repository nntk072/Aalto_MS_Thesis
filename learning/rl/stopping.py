"""RL training stop criteria derived from completed environment episodes."""

from __future__ import annotations

from typing import Any


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

    Non-finite metrics stop immediately. Optional no-trade stops and hard-risk
    stops wait until ``min_timesteps`` and require every episode in the last
    ``window`` to fail. A peak-drawdown policy miss never stops optimization.
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
    if stop_on_equity_gate and all(_episode_has_hard_breach(ep) for ep in recent):
        return "equity_gate:max_loss_stopped_trading"
    return None


def _episode_has_hard_breach(ep: dict[str, Any]) -> bool:
    """Detect hard account breaches without using peak-DD fit as a stop."""
    return bool(ep.get("breach_reason") or ep.get("breached"))


__all__ = ["decide_early_abort"]
