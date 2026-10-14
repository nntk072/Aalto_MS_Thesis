"""Tests for the separate, staged learning-quality registry."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from quant_rl.train.quality_registry import (
    apply_quality_variant,
    get_quality_variant,
    load_quality_registry,
)


def test_quality_registry_is_separate_and_staged() -> None:
    registry = load_quality_registry()
    assert registry.development_control == "ladder_a3_multi_tp"
    assert len(registry.stages) >= 5
    stage, overrides = get_quality_variant("q1_state_v2")
    assert stage == "q1_state"
    assert overrides["env.observation_schema_version"] == "account-state-v2"


def test_apply_quality_variant_clones_config_and_applies_only_declared_keys() -> None:
    base = OmegaConf.create({"env": {"n_envs": 8, "observation_schema_version": "v1"}})
    resolved = apply_quality_variant(base, "q1_state_v2")
    assert base.env.observation_schema_version == "v1"
    assert resolved.env.observation_schema_version == "account-state-v2"
    assert resolved.env.n_envs == 8
    assert resolved.env.strategy_actions is True
    assert resolved.env.allow_multi_tp is True
    assert resolved.agent.arch == "tcn"
    assert resolved.quality_diagnostics.enabled is True
    assert resolved.quality_variant_id == "q1_state_v2"


def test_later_stage_refuses_to_run_before_winner_selection() -> None:
    with pytest.raises(ValueError, match="select the prior-stage winner"):
        apply_quality_variant(OmegaConf.create({}), "q3a_rr_fixed500")
