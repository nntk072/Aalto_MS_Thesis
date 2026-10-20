"""Adapt backend-neutral decisions to actions accepted by an environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from decisioning.contracts import DecisionInput, DecisionProvider, to_policy_action


def make_provider_action_fn(
    provider: DecisionProvider,
    *,
    input_schema_version: str,
    deterministic: bool = True,
    continuous_actions: bool = False,
    strategy_actions: bool = False,
) -> Callable[[dict[str, Any]], int | float | np.ndarray[Any, Any]]:
    """Return the action callable used by generic evaluation rollouts.

    The provider returns named semantic actions. This boundary restores the
    environment's established scalar or coordinate-vector representation.
    """

    def action_fn(observation: dict[str, Any]) -> int | float | np.ndarray[Any, Any]:
        decision = provider.predict(
            DecisionInput(observation=observation, schema_version=input_schema_version),
            deterministic=deterministic,
        )
        action = to_policy_action(decision)
        if strategy_actions:
            return np.asarray(action, dtype=np.float32).reshape(-1)
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if continuous_actions:
            return float(values[0])
        return int(values[0])

    return action_fn


__all__ = ["make_provider_action_fn"]
