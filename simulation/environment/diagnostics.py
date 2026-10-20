"""Runtime diagnostics owned by the simulation environment.

These helpers collect per-step environment state without depending on training
or callback orchestration. Training callbacks may consume these diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from simulation.environment.rewards import PnLReward


class DebugStepError(Exception):
    """A step failure carrying the action, reward parts, and position."""

    def __init__(
        self,
        reason: str,
        snapshot: Mapping[str, Any],
        original: BaseException | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot = dict(snapshot)
        self.original = original


def new_debug_window() -> dict[str, Any]:
    """Empty per-env counters for one rollout."""
    return {
        "n": 0,
        "parts": {},
        "opens": 0,
        "shape_on_open": 0,
        "shape_on_hold": 0,
        "breach_events": 0,
        "entries": 0,
        "risk_sum": 0.0,
        "soft_daily_nz": 0,
        "soft_daily_repeat": 0,
        "soft_year_nz": 0,
        "soft_year_repeat": 0,
        "prev_soft_daily": False,
        "prev_soft_year": False,
        "rewards": [],
        "closes": 0,
        "structure_sl": 0,
        "structure_tp": 0,
    }


def note_debug_step(
    window: dict[str, Any],
    *,
    reward: float,
    reward_parts: Mapping[str, float],
    opened: bool,
    entered: bool,
    risk_frac: float,
    close_reason: str,
    action: Any,
    position: bool,
    position_direction: int,
) -> None:
    """Update counters. A non-finite reward raises :class:`DebugStepError`."""
    if not math.isfinite(float(reward)):
        raise DebugStepError(
            "non-finite reward",
            {
                "action": _jsonable(action),
                "reward": None if not math.isfinite(float(reward)) else float(reward),
                "reward_parts": _jsonable(dict(reward_parts)),
                "position": bool(position),
                "position_direction": int(position_direction),
            },
        )
    window["n"] += 1
    for key, value in reward_parts.items():
        window["parts"][str(key)] = float(window["parts"].get(str(key), 0.0)) + float(value)
    shaped = float(reward_parts.get("strategy", 0.0))
    if opened:
        window["opens"] += 1
        if shaped != 0.0:
            window["shape_on_open"] += 1
    elif shaped != 0.0:
        window["shape_on_hold"] += 1
    if float(reward_parts.get("breach", 0.0)) != 0.0:
        window["breach_events"] += 1
    if entered:
        window["entries"] += 1
        window["risk_sum"] += float(risk_frac)
    daily = float(reward_parts.get("soft_daily", 0.0)) != 0.0
    year = float(reward_parts.get("soft_year", 0.0)) != 0.0
    if daily:
        window["soft_daily_nz"] += 1
        if window["prev_soft_daily"]:
            window["soft_daily_repeat"] += 1
    if year:
        window["soft_year_nz"] += 1
        if window["prev_soft_year"]:
            window["soft_year_repeat"] += 1
    window["prev_soft_daily"] = daily
    window["prev_soft_year"] = year
    window["rewards"].append(float(reward))
    if close_reason:
        window["closes"] += 1
        if close_reason == "structure_sl":
            window["structure_sl"] += 1
        elif close_reason == "structure_tp":
            window["structure_tp"] += 1


def summarize_window(window: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one env window into a JSON-ready summary."""
    n = int(window["n"])
    rewards = [float(value) for value in window["rewards"]]
    nonzero = float(np.mean(np.abs(rewards) > 1e-8)) if rewards else 0.0
    return {
        "n": n,
        "parts": {key: float(value) for key, value in window["parts"].items()},
        "opens": int(window["opens"]),
        "shape_on_open": int(window["shape_on_open"]),
        "shape_on_hold": int(window["shape_on_hold"]),
        "breach_events": int(window["breach_events"]),
        "entries": int(window["entries"]),
        "risk_sum": float(window["risk_sum"]),
        "soft_daily_nz": int(window["soft_daily_nz"]),
        "soft_daily_repeat": int(window["soft_daily_repeat"]),
        "soft_year_nz": int(window["soft_year_nz"]),
        "soft_year_repeat": int(window["soft_year_repeat"]),
        "spike_share": _spike_share(rewards),
        "nonzero": nonzero,
        "closes": int(window["closes"]),
        "structure_sl": int(window["structure_sl"]),
        "structure_tp": int(window["structure_tp"]),
    }


def wiring_from_env(env: Any) -> dict[str, Any]:
    """Reward and risk settings the train env was built with."""
    reward = getattr(env, "reward_fn", None)
    base = getattr(reward, "_dsr_fn", None)
    if isinstance(reward, PnLReward):
        base = reward
    return {
        "reward_mode": str(getattr(env, "reward_mode", "")),
        "episodic": bool(getattr(env, "episodic", False)),
        "breach_penalty": float(
            getattr(base, "breach_penalty", getattr(reward, "breach_penalty", 0.0)) or 0.0
        ),
        "eval_breach_penalty": -10.0,
        "composite_dsr_weight": float(getattr(reward, "dsr_weight", 0.0) or 0.0),
        "pnl_dsr_weight": float(getattr(base, "dsr_weight", 0.0) or 0.0),
        "strategy_weight": float(getattr(reward, "strategy_weight", 0.0) or 0.0),
        "risk_floor": float(getattr(env, "risk_floor", 0.0) or 0.0),
        "risk_frac_range": [float(value) for value in getattr(env, "risk_frac_range", ())],
    }


def _spike_share(rewards: Sequence[float]) -> float:
    if not rewards:
        return 0.0
    values = np.asarray(rewards, dtype=float)
    centered = values - float(np.mean(values))
    total = float(np.dot(centered, centered))
    if total <= 0.0:
        return 0.0
    k = max(1, int(0.01 * values.size))
    top = np.argpartition(np.abs(values), -k)[-k:]
    return float(np.dot(centered[top], centered[top]) / total)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
