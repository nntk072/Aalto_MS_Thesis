"""Contracts for explicit account observation versions and v2 state geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_rl.envs.entry_state import ENTRY_STATE_DIM, entry_state_one_hot
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.observation_schema import (
    OBSERVATION_SCHEMA_V1,
    OBSERVATION_SCHEMA_V2,
    V2_COORDINATES,
    account_state_v2,
    build_observation_schema,
    validate_observation_schema,
)
from tests.test_envs.test_account_normalization import _make_bars, _make_features


def _state(**overrides: object) -> np.ndarray[Any, Any]:
    values: dict[str, object] = {
        "equity": 100_000.0,
        "initial_balance": 100_000.0,
        "position_direction": 1.0,
        "open_pnl": 100.0,
        "planned_risk_usd": 500.0,
        "trailing_drawdown_pct": 0.01,
        "trailing_drawdown_limit": 0.07,
        "daily_loss": 100.0,
        "daily_loss_limit": 5_000.0,
        "loss_from_initial": 200.0,
        "max_loss_limit": 10_000.0,
        "risk_per_trade_limit": 1_000.0,
        "current_price": 101.0,
        "entry_price": 100.0,
        "initial_sl_price": 98.0,
        "active_sl_price": 99.0,
        "remaining_size": 0.5,
        "initial_size": 1.0,
        "position_age_bars": 30.0,
        "position_age_scale_bars": 60.0,
        "tp_hit_mask": 1,
        "tp_prices": (104.0, 106.0, 108.0),
        "sl_mode": "breakeven",
        "tp_mode": "trailing",
        "exit_mode": "ema_21",
    }
    values.update(overrides)
    return account_state_v2(**values)  # type: ignore[arg-type]


def test_v2_schema_has_exact_named_width() -> None:
    schema = build_observation_schema(OBSERVATION_SCHEMA_V2)
    assert schema.width == len(V2_COORDINATES) == 28
    assert schema.coordinates == V2_COORDINATES


@pytest.mark.parametrize(
    ("coordinate", "field", "new_value"),
    [
        ("daily_loss_over_limit", "daily_loss", 900.0),
        ("loss_from_initial_over_limit", "loss_from_initial", 2_000.0),
        ("active_sl_headroom_over_initial_risk", "active_sl_price", 100.5),
        ("distance_to_tp1_over_initial_risk", "tp_prices", (102.0, 106.0, 108.0)),
    ],
)
def test_risk_and_position_state_changes_are_observable(
    coordinate: str, field: str, new_value: object
) -> None:
    index = V2_COORDINATES.index(coordinate)
    assert _state()[index] != _state(**{field: new_value})[index]


def test_partial_tp_and_modes_are_observable() -> None:
    state = _state()
    assert state[V2_COORDINATES.index("remaining_size_fraction")] == pytest.approx(0.5)
    assert state[V2_COORDINATES.index("tp1_hit")] == 1.0
    assert state[V2_COORDINATES.index("tp2_hit")] == 0.0
    assert state[V2_COORDINATES.index("sl_mode_breakeven")] == 1.0
    assert state[V2_COORDINATES.index("tp_mode_trailing")] == 1.0
    assert state[V2_COORDINATES.index("exit_mode_ema")] == 1.0


def test_short_target_distances_are_positive_toward_target() -> None:
    state = _state(
        position_direction=-1.0,
        entry_price=100.0,
        current_price=101.0,
        initial_sl_price=103.0,
        active_sl_price=102.0,
        tp_prices=(98.0, 96.0, 94.0),
    )
    assert state[V2_COORDINATES.index("distance_to_tp1_over_initial_risk")] == pytest.approx(1.0)


def test_no_position_clears_position_geometry() -> None:
    state = _state(position_direction=0.0)
    position_coordinates = V2_COORDINATES[10:]
    assert np.all(state[10:] == 0.0)
    assert len(position_coordinates) == 18


def test_daily_loss_definition_is_not_reinterpreted() -> None:
    # Positive realized loss is represented as positive normalized budget use.
    assert _state()[V2_COORDINATES.index("daily_loss_over_limit")] == pytest.approx(0.02)


def test_observation_schema_validation_fails_closed_for_v2() -> None:
    expected = build_observation_schema(OBSERVATION_SCHEMA_V2)
    with pytest.raises(ValueError, match="no observation schema metadata"):
        validate_observation_schema(None, expected)
    legacy_v1 = build_observation_schema(OBSERVATION_SCHEMA_V1)
    with pytest.raises(ValueError, match="no observation schema metadata"):
        validate_observation_schema(None, legacy_v1)
    validate_observation_schema(None, legacy_v1, allow_legacy_v1=True)
    with pytest.raises(ValueError, match="differs"):
        validate_observation_schema(
            build_observation_schema(OBSERVATION_SCHEMA_V1).to_dict(), expected
        )


def test_v2_entry_fsm_coordinates_follow_the_state_machine() -> None:
    bars = _make_bars(120)
    features = _make_features(bars)
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        entry_state_machine=True,
        entry_state_observation=True,
        observation_schema_version=OBSERVATION_SCHEMA_V2,
    )
    obs, _ = env.reset(seed=17)
    assert obs["account"].shape == (len(V2_COORDINATES) + ENTRY_STATE_DIM,)
    for _ in range(5):
        assert env.entry_state_machine is not None
        expected_state = np.asarray(entry_state_one_hot(env.entry_state_machine.state))
        np.testing.assert_array_equal(obs["account"][-ENTRY_STATE_DIM:], expected_state)
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break


def test_legacy_sequence_column_mode_reconstructs_pre_filter_coordinates() -> None:
    bars = _make_bars(120)
    features = _make_features(bars)
    features["po3_ifvg_time_ns"] = np.asarray(bars.index, dtype="datetime64[ns]").astype(np.int64)

    current = TradingEnv(bars=bars, features=features, obs_window=10)
    legacy = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        episodic=False,
        legacy_sequence_columns=True,
    )

    assert "po3_ifvg_time_ns" not in current._obs_columns
    assert "po3_ifvg_time_ns" in legacy._obs_columns
    current_obs, _ = current.reset(seed=7)
    legacy_obs, _ = legacy.reset(seed=7)
    assert legacy_obs["seq"].shape[1] == current_obs["seq"].shape[1] + 1
