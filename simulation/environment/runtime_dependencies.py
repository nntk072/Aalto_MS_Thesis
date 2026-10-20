# ruff: noqa: F401
"""Gymnasium environment for RL-based trading with structure-aware SL/TP.

Wraps the backtest engine with a standard Gym interface. Actions support:
- Discrete: {hold=0, enter_long=1-9, enter_short=10-18, exit=19}
- Continuous: Box(-1, 1) for proportional position sizing

Observation: Dict space with time-series features (60-bar window) + account state.
Reward: Differential Sharpe Ratio (DSR) or Sweep Confirmation Reward.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from features import (
    attach_reachable_r,
    select_legacy_obs_columns,
    select_obs_columns,
)
from market_data.loading.ticks import (
    TickBook,
    TickExecutionUnavailable,
    resolve_tick_execution,
    resolve_tick_execution_by_deadline,
    resolve_tick_exit,
)
from market_data.transforms.sessions import ny_session_mask
from simulation.contracts.actions import (
    ActionSchema,
    DiscreteActionSchema,
    build_action_schema,
    build_discrete_trading_action_schema,
)
from trading.account.state import AccountState
from trading.execution.broker import Broker, Position
from trading.execution.costs import COST_US100, CostModel
from trading.risk.guardrails import FTMOGuardrails
from trading.risk.sizing import compute_lots, compute_sl_tp_long, compute_sl_tp_short
from trading.strategies import BaselineStrategy, TradingStrategy

from .entry_state import (
    ENTRY_STATE_DIM,
    EntryCandidate,
    EntryEvidence,
    EntryStateMachine,
    entry_state_one_hot,
    immutable_setup_levels,
)
from .market_view import BarView, FeatureRow
from .rewards import DSRReward, EquityDeltaReward, PnLReward, RMultipleReward
from .sweep_reward import CompositeReward, SweepConfirmationReward

if TYPE_CHECKING:
    from models.auxiliary.vae import VAE
from simulation.contracts.observations import (
    OBSERVATION_SCHEMA_V1,
    OBSERVATION_SCHEMA_V2,
    OBSERVATION_SCHEMA_V3,
    account_state_v2,
    build_observation_schema,
)
from simulation.environment.diagnostics import (
    DebugStepError,
    new_debug_window,
    note_debug_step,
    summarize_window,
    wiring_from_env,
)
from simulation.environment.episode_equity import assess_train_equity
from trading.policy.config import TradingPolicyConfig

from .observations import (
    DEFAULT_HTF_WINDOWS,
    HTF_BRANCHES,
    closed_window,
    normalized_account_vector,
    pad_observation_window,
    run_starts,
    split_obs_columns,
)

#: Fallback bar spacing when the index has no usable delta (single bar, or a
#: non-monotonic index). The env steps the M1 spine, so one minute is the safe
#: default rather than the old hard-coded five.
_DEFAULT_MINUTES_PER_STEP = 1.0

# Broker session clock used for wall-clock timestamps (trade log, equity curve).
# Frames mode indexes are tz-aware in this zone; the bundle mode stores the same
# UTC epoch as a naive datetime64[ns] array, so both paths are normalised here.
BAR_TIME_TZ = "Etc/GMT-3"


def _minutes_per_step(index: pd.DatetimeIndex) -> float:
    """Median bar spacing in minutes, read from the index itself.

    Assumes nothing about the timeframe: the environment is fed M1 bars today,
    but a resampled feed must not silently make every time-based term wrong.
    """
    if index is None or len(index) < 2:
        return _DEFAULT_MINUTES_PER_STEP
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    deltas = np.diff(nanos)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return _DEFAULT_MINUTES_PER_STEP
    minutes = float(np.median(positive)) / 60_000_000_000.0
    return minutes if minutes > 0.0 else _DEFAULT_MINUTES_PER_STEP


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


def _empty_entry_diag(entry_state_machine: bool = False) -> dict[str, int]:
    """Per-episode entry diagnostics (strategy_actions).

    Defined once so ``__init__`` and ``reset`` cannot drift apart. Every
    ``rejected_*`` bucket corresponds to exactly one ``entry_rejection_reason``
    string; ``entry_rejected_total`` in the step info is derived from the
    ``rejected_`` prefix rather than stored, so it cannot go stale.
    """
    diagnostics = {
        # Requested entry suppressed before the order was built.
        "hold_low_intensity": 0,
        "hold_no_context": 0,
        "hold_risk_floor": 0,
        "rejected_overnight": 0,
        "rejected_blackout": 0,
        "rejected_session_cap": 0,
        "rejected_cooldown": 0,
        "rejected_manipulation": 0,
        "rejected_gate": 0,
        "rejected_soft_brick": 0,
        "rejected_terminal": 0,
        "rejected_entry_pending": 0,
        # Order geometry rejected (place_strategy_order returned no survivor).
        "rejected_sl": 0,
        "rejected_rr": 0,
        "rejected_ema": 0,
        # Geometry existed but the stop was already through the fill quote.
        "rejected_sl_already_hit": 0,
        "opened": 0,
    }
    if entry_state_machine:
        diagnostics["rejected_entry_state"] = 0
    return diagnostics


#: Rejection reason -> ``_entry_diag`` bucket. Every site that refuses an entry
#: must go through :meth:`TradingEnv._reject`, which increments the mapped
#: bucket and sets the reason atomically, so counter and reason cannot drift
#: apart the way the old "ema increments rejected_sl but reports no_valid_tp"
#: mismatch did. A reason missing from this map raises ``KeyError`` at the
#: rejection site instead of silently dropping into the funnel.
_REJECTION_COUNTERS: dict[str, str] = {
    "overnight_boundary": "rejected_overnight",
    "open_blackout": "rejected_blackout",
    "session_cap": "rejected_session_cap",
    "entry_cooldown": "rejected_cooldown",
    "manipulation_unconfirmed": "rejected_manipulation",
    "entry_gate": "rejected_gate",
    "premium_block": "rejected_gate",
    "discount_block": "rejected_gate",
    "missing_previous_session": "rejected_gate",
    "soft_brick": "rejected_soft_brick",
    "entry_state_not_armed": "rejected_entry_state",
    "no_valid_sl": "rejected_sl",
    "no_valid_tp": "rejected_rr",
    "no_ema_exit": "rejected_ema",
    "sl_already_hit": "rejected_sl_already_hit",
    "terminal_no_fill_bar": "rejected_terminal",
    "entry_pending": "rejected_entry_pending",
}


def _previous_session_audit(
    strategy: object, row: FeatureRow | pd.Series, decision_close: float
) -> dict[str, Any]:
    """Audit the Idea 2 reference on each accepted or rejected entry attempt."""
    if not bool(getattr(strategy, "previous_session_pd_gate", False)):
        return {}
    code = float(row.get("previous_session_code", np.nan))
    code_to_label = {1: "asian", 2: "london", 3: "ny"}
    completed_ns = float(row.get("previous_session_completed_at_ns", np.nan))
    return {
        "decision_close": float(decision_close),
        "previous_session_high": float(row.get("previous_session_high", np.nan)),
        "previous_session_low": float(row.get("previous_session_low", np.nan)),
        "previous_session_midpoint": float(row.get("previous_session_midpoint", np.nan)),
        "previous_session_code": code,
        "previous_session_label": code_to_label.get(int(code), "") if np.isfinite(code) else "",
        "previous_session_completed_at_ns": int(completed_ns)
        if np.isfinite(completed_ns)
        else None,
    }


_PRICE_TOL = 1e-4
# Monotonic SL mode ordering: fixed → breakeven → trailing.
# Used to advance the state machine only forward.
_SL_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}
_TP_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}


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


class RuntimeMixinBase:
    """Typing bridge for mixins composed only through TradingEnv.

    Runtime mixins intentionally depend on sibling mixins and TradingEnv-owned
    state. During static checking, unknown sibling attributes are treated as
    dynamically supplied by the final composition. At runtime this class adds
    no fallback behavior, so missing attributes still fail normally.
    """

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...


__all__ = [
    "RuntimeMixinBase",
    "warnings",
    "asdict",
    "Path",
    "Any",
    "cast",
    "gym",
    "np",
    "pd",
    "spaces",
    "attach_reachable_r",
    "select_legacy_obs_columns",
    "select_obs_columns",
    "TickBook",
    "TickExecutionUnavailable",
    "resolve_tick_execution",
    "resolve_tick_execution_by_deadline",
    "resolve_tick_exit",
    "ny_session_mask",
    "ActionSchema",
    "DiscreteActionSchema",
    "build_action_schema",
    "build_discrete_trading_action_schema",
    "AccountState",
    "Broker",
    "Position",
    "COST_US100",
    "CostModel",
    "FTMOGuardrails",
    "compute_lots",
    "compute_sl_tp_long",
    "compute_sl_tp_short",
    "BaselineStrategy",
    "TradingStrategy",
    "ENTRY_STATE_DIM",
    "EntryCandidate",
    "EntryEvidence",
    "EntryStateMachine",
    "entry_state_one_hot",
    "immutable_setup_levels",
    "BarView",
    "FeatureRow",
    "DSRReward",
    "EquityDeltaReward",
    "PnLReward",
    "RMultipleReward",
    "CompositeReward",
    "SweepConfirmationReward",
    "OBSERVATION_SCHEMA_V1",
    "OBSERVATION_SCHEMA_V2",
    "OBSERVATION_SCHEMA_V3",
    "account_state_v2",
    "build_observation_schema",
    "DebugStepError",
    "new_debug_window",
    "note_debug_step",
    "summarize_window",
    "wiring_from_env",
    "assess_train_equity",
    "TradingPolicyConfig",
    "DEFAULT_HTF_WINDOWS",
    "HTF_BRANCHES",
    "closed_window",
    "normalized_account_vector",
    "pad_observation_window",
    "run_starts",
    "split_obs_columns",
    "_DEFAULT_MINUTES_PER_STEP",
    "BAR_TIME_TZ",
    "_minutes_per_step",
    "_DECISION_KEYS",
    "_empty_decision",
    "_empty_entry_diag",
    "_REJECTION_COUNTERS",
    "_previous_session_audit",
    "_PRICE_TOL",
    "_SL_MODE_RANK",
    "_TP_MODE_RANK",
    "_at",
    "_decision_from_action",
]
