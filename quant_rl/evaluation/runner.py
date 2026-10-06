"""Episode runner: steps a policy through an environment and scores it.

The runner is policy-agnostic: ``action_fn`` maps observations to actions,
so classical baselines, supervised models and RL agents all share the same
evaluation loop. Trade PnLs are read from the environment's ``trade_log``
and breaches from ``breach_events``, both maintained by :class:`TradingEnv`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np

from .distributions import compute_distribution_metrics
from .metrics import DEFAULT_PERIODS_PER_YEAR, PerformanceMetrics, compute_metrics


def run_episode(
    env: Any,
    action_fn: Callable[
        [Any], int | float | np.floating[Any] | np.integer[Any] | np.ndarray[Any, Any]
    ],
    max_steps: int = 100_000,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    """Run one full episode and return performance metrics.

    Args:
        env: A gym-style environment (e.g. ``TradingEnv``) exposing
            ``reset()``, ``step()`` and, ideally, ``trade_log`` /
            ``breach_events`` lists.
        action_fn: Maps the observation to an action accepted by ``env``.
        max_steps: Safety cap on the number of environment steps.
        periods_per_year: Bars per year used to annualise ratios.

    Returns:
        Metrics computed over the collected equity curve and trades.
    """
    observation, _ = env.reset()
    equities: list[float] = []
    done = truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        observation, _, done, truncated, info = env.step(action_fn(observation))
        steps += 1
        # Some environments emit an empty info dict on the terminal step
        if "equity" in info:
            equities.append(float(info["equity"]))

    if not equities:
        raise ValueError("episode produced no steps; equity curve is empty")

    trade_pnls = [float(trade["pnl"]) for trade in getattr(env, "trade_log", []) if "pnl" in trade]
    breach_count = len(getattr(env, "breach_events", []))
    initial_balance = float(getattr(env, "initial_balance", equities[0]))

    dist = compute_distribution_metrics(trade_pnls)
    extras = {
        f"pnl_dist_{k}": float(v) for k, v in asdict(dist).items() if isinstance(v, (int, float))
    }
    # Entry-state lifecycle + refusal funnel, so a collapsing FSM is
    # distinguishable from a working one in the report (Tier 1 capture).
    entry_diag = getattr(env, "_entry_diag", None)
    if entry_diag:
        opened = float(entry_diag.get("opened", 0))
        attempts = opened + sum(v for k, v in entry_diag.items() if k.startswith("rejected_"))
        extras["entry_opened"] = float(opened)
        extras["entry_attempts"] = float(attempts)
        extras["entry_rejection_rate"] = round(1.0 - opened / attempts, 6) if attempts else 0.0
        for key, value in entry_diag.items():
            if key.startswith("rejected_"):
                extras[f"entry_{key}"] = float(value)
                extras[f"entry_{key}_rate"] = round(value / attempts, 6) if attempts else 0.0
    state_diag = getattr(env, "entry_state_diagnostics", None)
    if callable(state_diag):
        extras.update(state_diag())

    return compute_metrics(
        equities,
        initial_balance=initial_balance,
        trade_pnls=trade_pnls,
        periods_per_year=periods_per_year,
        breach_count=breach_count,
        extras=extras,
    )
