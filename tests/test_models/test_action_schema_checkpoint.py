"""Checkpoint schema metadata must survive save/load and reject mismatches."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from quant_rl.action_schema import build_action_schema, build_discrete_trading_action_schema
from quant_rl.models.agent import (
    LegacyFourToFiveActionAdapter,
    bind_discrete_action_schema,
    validate_model_action_schema,
    validate_model_discrete_action_schema,
)


def _model(width: int, key: str | None, version: str | None) -> SimpleNamespace:
    policy = SimpleNamespace(_action_layout_key=key, _action_schema_version=version)
    return SimpleNamespace(policy=policy, action_space=SimpleNamespace(shape=(width,)))


def test_checkpoint_rejects_same_width_different_semantics() -> None:
    schema = build_action_schema(has_direction=True)
    other = build_action_schema(has_sl_mode=True)
    assert schema.width == other.width
    with pytest.raises(ValueError, match="layout"):
        validate_model_action_schema(_model(schema.width, other.key, schema.version), schema)


def test_checkpoint_rejects_missing_version_by_default() -> None:
    schema = build_action_schema(has_direction=True)
    with pytest.raises(ValueError, match="unversioned"):
        validate_model_action_schema(_model(schema.width, None, None), schema)


def test_checkpoint_legacy_path_is_explicit() -> None:
    schema = build_action_schema(has_direction=True)
    model = _model(schema.width, None, None)
    with pytest.raises(ValueError, match="explicit legacy layout key"):
        validate_model_action_schema(model, schema, allow_legacy=True)
    validate_model_action_schema(
        model,
        schema,
        allow_legacy=True,
        legacy_layout_key=schema.key,
    )
    assert model._action_schema["coordinates"] == list(schema.names)
    assert model._action_schema["checkpoint_metadata"] == "legacy_unversioned"


def test_unversioned_same_width_checkpoint_requires_the_selected_layout() -> None:
    schema = build_action_schema(has_direction=True)
    other = build_action_schema(has_sl_mode=True)
    model = _model(schema.width, None, None)
    with pytest.raises(ValueError, match="explicit legacy layout"):
        validate_model_action_schema(
            model,
            schema,
            allow_legacy=True,
            legacy_layout_key=other.key,
        )


def test_legacy_4d_no_exit_migration_appends_structural_exit() -> None:
    schema = build_action_schema()
    model = _model(4, None, None)
    migration = validate_model_action_schema(
        model,
        schema,
        allow_legacy=True,
        legacy_layout_key="legacy4_no_exit_v1",
    )
    assert migration == "legacy4_no_exit_v1_to_overlay-action-v2-5"

    class _Model:
        def predict(self, observation, *, deterministic):
            return np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32), None

    adapter = LegacyFourToFiveActionAdapter(_Model())
    action, state = adapter.predict(np.zeros(1), deterministic=True)
    assert state is None
    assert action.tolist() == pytest.approx([0.1, -0.2, 0.3, -0.4, -1.0])


def test_checkpoint_validation_exposes_versioned_schema_manifest() -> None:
    schema = build_action_schema(has_direction=True)
    model = _model(schema.width, schema.key, schema.version)

    validate_model_action_schema(model, schema)

    assert model._action_schema["version"] == schema.version
    assert model._action_schema["key"] == schema.key
    assert model._action_schema["checkpoint_metadata"] == "versioned"


def test_discrete_checkpoint_requires_and_validates_named_categories() -> None:
    schema = build_discrete_trading_action_schema()
    model = SimpleNamespace(action_space=spaces.Discrete(schema.width), policy=SimpleNamespace())
    with pytest.raises(ValueError, match="unversioned discrete"):
        validate_model_discrete_action_schema(model, schema, schema.to_dict())

    bind_discrete_action_schema(model, schema)

    assert validate_model_discrete_action_schema(model, schema, schema.to_dict()) == "verified"
    assert model._action_schema["coordinates"][0] == "hold"
    assert model._action_schema["coordinates"][1] == "enter_long_risk_low_rr_low"
    assert model._action_schema["coordinates"][10] == "enter_short_risk_low_rr_low"
    assert model._action_schema["coordinates"][-1] == "exit"


def test_discrete_checkpoint_legacy_schema_requires_exact_opt_in() -> None:
    schema = build_discrete_trading_action_schema()
    model = SimpleNamespace(action_space=spaces.Discrete(schema.width), policy=SimpleNamespace())

    with pytest.raises(ValueError, match="exact discrete layout key"):
        validate_model_discrete_action_schema(
            model,
            schema,
            source_schema=None,
            allow_legacy=True,
        )

    disposition = validate_model_discrete_action_schema(
        model,
        schema,
        source_schema=None,
        allow_legacy=True,
        legacy_layout_key=schema.key,
    )

    assert disposition == "legacy_unverified"
    assert model._action_schema["checkpoint_metadata"] == "legacy_unverified"


def test_checkpoint_rejects_mismatched_width_even_with_legacy_opt_in() -> None:
    schema = build_action_schema(has_sl_mode=True, has_multi_tp=True)
    with pytest.raises(ValueError, match="width"):
        validate_model_action_schema(
            _model(5, None, None),
            schema,
            allow_legacy=True,
            legacy_layout_key=schema.key,
        )
