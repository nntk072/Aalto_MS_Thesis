"""Versioned account-state observation contracts.

Observation meaning is part of a policy's input schema. In particular, equal
vector widths do not establish that two checkpoints use the same coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_rl.envs.entry_state import ENTRY_STATES, entry_state_one_hot

OBSERVATION_SCHEMA_V1 = "account-state-v1"
OBSERVATION_SCHEMA_V2 = "account-state-v2"

V1_COORDINATES = (
    "log_equity_over_initial",
    "position_direction",
    "open_pnl_over_initial",
    "unrealized_r_legacy",
    "initial_stop_width_over_price",
    "trailing_drawdown_pct",
)

V2_COORDINATES = (
    "log_equity_over_initial",
    "position_direction",
    "open_pnl_over_initial",
    "unrealized_r",
    "trailing_drawdown_over_limit",
    "daily_loss_over_limit",
    "remaining_daily_loss_budget_over_limit",
    "loss_from_initial_over_limit",
    "remaining_max_loss_budget_over_limit",
    "planned_risk_over_limit",
    "initial_stop_distance_over_price",
    "active_sl_headroom_over_initial_risk",
    "remaining_size_fraction",
    "position_age_over_scale",
    "tp1_hit",
    "tp2_hit",
    "tp3_hit",
    "distance_to_tp1_over_initial_risk",
    "distance_to_tp2_over_initial_risk",
    "distance_to_tp3_over_initial_risk",
    "sl_mode_fixed",
    "sl_mode_breakeven",
    "sl_mode_trailing",
    "tp_mode_fixed",
    "tp_mode_breakeven",
    "tp_mode_trailing",
    "exit_mode_structural",
    "exit_mode_ema",
)


@dataclass(frozen=True)
class ObservationSchema:
    """Explicit coordinate names and version for the account branch."""

    version: str
    coordinates: tuple[str, ...]
    entry_state_visibility: bool = False

    @property
    def width(self) -> int:
        return len(self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "observation_width": self.width,
            "observation_coordinates": list(self.coordinates),
            "entry_state_visibility": self.entry_state_visibility,
        }


def build_observation_schema(
    version: str = OBSERVATION_SCHEMA_V1,
    *,
    entry_state_visibility: bool = False,
) -> ObservationSchema:
    """Return a named schema; never infer semantics from a vector width."""
    coordinates: tuple[str, ...]
    if version == OBSERVATION_SCHEMA_V1:
        coordinates = V1_COORDINATES
    elif version == OBSERVATION_SCHEMA_V2:
        coordinates = V2_COORDINATES
    else:
        raise ValueError(f"unknown observation schema version: {version!r}")
    if entry_state_visibility:
        coordinates = (*coordinates, *(f"entry_state_{name.lower()}" for name in ENTRY_STATES))
    return ObservationSchema(version, coordinates, entry_state_visibility)


def validate_observation_schema(
    actual: dict[str, Any] | None,
    expected: ObservationSchema,
    *,
    allow_legacy_v1: bool = False,
) -> None:
    """Validate saved observation semantics before checkpoint evaluation/resume."""
    if actual is None:
        if expected.version == OBSERVATION_SCHEMA_V1 and allow_legacy_v1:
            return
        raise ValueError(
            "checkpoint has no observation schema metadata; explicit schema metadata is "
            f"required for {expected.version}"
        )
    saved = {
        "version": actual.get("version"),
        "observation_width": actual.get("observation_width"),
        "observation_coordinates": actual.get("observation_coordinates"),
        "entry_state_visibility": actual.get("entry_state_visibility"),
    }
    if saved != expected.to_dict():
        raise ValueError(
            "checkpoint observation schema differs from requested environment schema: "
            f"saved={saved!r}, requested={expected.to_dict()!r}"
        )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def account_state_v2(
    *,
    equity: float,
    initial_balance: float,
    position_direction: float,
    open_pnl: float,
    planned_risk_usd: float = 0.0,
    trailing_drawdown_pct: float = 0.0,
    trailing_drawdown_limit: float = 0.0,
    daily_loss: float = 0.0,
    daily_loss_limit: float = 0.0,
    loss_from_initial: float = 0.0,
    max_loss_limit: float = 0.0,
    risk_per_trade_limit: float = 0.0,
    current_price: float = 0.0,
    entry_price: float = 0.0,
    initial_sl_price: float | None = None,
    active_sl_price: float | None = None,
    remaining_size: float = 0.0,
    initial_size: float = 0.0,
    position_age_bars: float = 0.0,
    position_age_scale_bars: float = 60.0,
    tp_hit_mask: int = 0,
    tp_prices: tuple[float | None, float | None, float | None] = (None, None, None),
    sl_mode: str = "fixed",
    tp_mode: str = "fixed",
    exit_mode: str = "structural",
    entry_state: str | None = None,
) -> np.ndarray[Any, Any]:
    """Build normalized v2 state without changing account or risk semantics.

    ``daily_loss`` intentionally retains the existing realized-loss-only
    definition. ``active_sl_headroom_over_initial_risk`` is positive when the
    current price is on the safe side of the active stop.
    """
    has_position = float(position_direction) != 0.0
    initial_risk = (
        abs(float(entry_price) - float(initial_sl_price))
        if has_position and initial_sl_price is not None
        else 0.0
    )
    stop_width = (
        abs(float(entry_price) - float(initial_sl_price))
        if has_position and initial_sl_price is not None
        else 0.0
    )
    active_headroom = (
        float(position_direction) * (float(current_price) - float(active_sl_price))
        if has_position and active_sl_price is not None
        else 0.0
    )
    target_distances = [
        float(position_direction) * (float(target) - float(current_price))
        if has_position and target is not None
        else 0.0
        for target in tp_prices
    ]
    daily_ratio = _ratio(float(daily_loss), float(daily_loss_limit))
    max_loss_ratio = _ratio(float(loss_from_initial), float(max_loss_limit))
    sl_modes = (
        tuple(float(sl_mode == mode) for mode in ("fixed", "breakeven", "trailing"))
        if has_position
        else (0.0, 0.0, 0.0)
    )
    tp_modes = (
        tuple(float(tp_mode == mode) for mode in ("fixed", "breakeven", "trailing"))
        if has_position
        else (0.0, 0.0, 0.0)
    )
    exit_modes = (
        (float(exit_mode == "structural"), float(exit_mode == "ema_21"))
        if has_position
        else (0.0, 0.0)
    )
    values = [
        float(np.log(equity / initial_balance)) if initial_balance > 0.0 and equity > 0.0 else 0.0,
        float(position_direction),
        _ratio(float(open_pnl), float(initial_balance)),
        _ratio(float(open_pnl), float(planned_risk_usd)) if has_position else 0.0,
        _ratio(float(trailing_drawdown_pct), float(trailing_drawdown_limit)),
        daily_ratio,
        1.0 - daily_ratio if daily_loss_limit > 0.0 else 0.0,
        max_loss_ratio,
        1.0 - max_loss_ratio if max_loss_limit > 0.0 else 0.0,
        _ratio(float(planned_risk_usd), float(risk_per_trade_limit)) if has_position else 0.0,
        _ratio(stop_width, float(current_price)),
        _ratio(active_headroom, initial_risk),
        _ratio(float(remaining_size), float(initial_size)) if has_position else 0.0,
        _ratio(float(position_age_bars), float(position_age_scale_bars)) if has_position else 0.0,
        *(
            float(bool(int(tp_hit_mask) & (1 << index))) if has_position else 0.0
            for index in range(3)
        ),
        *(_ratio(distance, initial_risk) for distance in target_distances),
        *sl_modes,
        *tp_modes,
        *exit_modes,
    ]
    if entry_state is not None:
        values.extend(entry_state_one_hot(entry_state))
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("account-state-v2 produced a non-finite observation")
    return result
