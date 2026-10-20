"""Action-coordinate conversion used for environment telemetry."""

from __future__ import annotations

from typing import Any

import numpy as np
from decisioning.actions import ActionSchema

_DECISION_KEYS: tuple[str, ...] = (
    "action_direction",
    "action_intensity",
    "action_stop",
    "action_risk",
    "action_target",
    "action_sl_mode",
    "action_exit",
    "intensity_u",
    "stop_u",
    "risk_u",
    "target_u",
    "sl_mode_u",
    "exit_u",
)


def _empty_decision() -> dict[str, float]:
    return {key: float("nan") for key in _DECISION_KEYS}


def _at(values: np.ndarray[Any, Any], index: int) -> float:
    if values.size <= index:
        return float("nan")
    return float(values[index])


def _decision_from_action(
    raw: np.ndarray[Any, Any],
    unit: np.ndarray[Any, Any],
    schema: ActionSchema,
) -> dict[str, float]:
    """Record raw/unit action coordinates using the versioned layout."""
    result = _empty_decision()
    for name, idx in schema.indices.items():
        result[f"action_{name}"] = _at(raw, idx)
        result[f"{name}_u"] = _at(unit, idx)
    return result


__all__ = ["_DECISION_KEYS", "_at", "_decision_from_action", "_empty_decision"]
