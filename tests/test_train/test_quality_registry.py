"""Tests for the separate, staged learning-quality registry."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from configuration import load_config
from training.configuration.quality import (
    apply_quality_variant,
    get_quality_variant,
    load_quality_registry,
    quality_variant_resource_contract,
    validate_quality_resource_shape,
)
from training.setup.models import _log_effective_ppo


def test_quality_registry_is_separate_and_staged() -> None:
    registry = load_quality_registry()
    assert registry.development_control == "ladder_a3_multi_tp"
    assert [str(stage.id) for stage in registry.stages] == [
        "q1_state",
        "q2_rollout",
        "q3_reward",
        "q4_action",
        "q5_constraints",
        "q6_entry_fsm",
        "q7_encoder",
    ]
    q0 = registry.stages[0].variants[0]
    assert q0.id == "q0_control_a3"
    assert q0.status == "control"
    stage, overrides = get_quality_variant("q1_state_v2")
    assert stage == "q1_state"
    assert overrides["env.observation_schema_version"] == "account-state-v2"


def test_quality_registry_rejects_unknown_override_path(tmp_path) -> None:
    registry_path = tmp_path / "quality.yaml"
    registry_path.write_text(
        """registry_version: 1
development_control: control
stages:
  - id: q0
    control: control
    variants:
      - id: control
        parent: null
        status: control
        overrides: {env.typo_that_would_be_ignored: true}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not exist in config/base/default.yaml"):
        load_quality_registry(registry_path)


def test_apply_quality_variant_clones_config_and_applies_only_declared_keys() -> None:
    base = load_config()
    base.env.n_envs = 8
    base.env.observation_schema_version = "v1"
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


def test_q2_rollout_geometry_arms_are_registered_but_fail_closed() -> None:
    for n_envs in (16, 32, 64):
        variant_id = f"q2_state_v2_nenv{n_envs}"
        stage, overrides = get_quality_variant(variant_id)
        assert stage == "q2_rollout"
        assert overrides["env.n_envs"] == n_envs
        assert overrides["ppo.n_steps"] == 8192 // n_envs
        assert overrides["env.n_envs"] * overrides["ppo.n_steps"] == 8192
        assert quality_variant_resource_contract(variant_id) == {
            "exact_n_envs": True,
            "expected_n_envs": n_envs,
            "aggregate_rollout_steps": 8192,
        }
        with pytest.raises(ValueError, match="not implemented and selectable"):
            apply_quality_variant(OmegaConf.create({}), variant_id)


@pytest.mark.parametrize("n_envs", (16, 32, 64))
def test_q2_exact_environment_resource_contract(n_envs: int) -> None:
    variant_id = f"q2_state_v2_nenv{n_envs}"
    validate_quality_resource_shape(
        variant_id,
        requested_n_envs=n_envs,
        effective_n_envs=n_envs,
        aggregate_rollout_steps=8192,
    )
    with pytest.raises(ValueError, match=f"requires exactly {n_envs} environments"):
        validate_quality_resource_shape(
            variant_id,
            requested_n_envs=n_envs,
            effective_n_envs=n_envs // 2,
            aggregate_rollout_steps=8192,
        )
    with pytest.raises(ValueError, match="requires requested_n_envs"):
        validate_quality_resource_shape(
            variant_id,
            requested_n_envs=n_envs // 2,
            effective_n_envs=n_envs // 2,
            aggregate_rollout_steps=8192,
        )
    with pytest.raises(ValueError, match="requires aggregate_rollout_steps=8192"):
        validate_quality_resource_shape(
            variant_id,
            requested_n_envs=n_envs,
            effective_n_envs=n_envs,
            aggregate_rollout_steps=4096,
        )


def test_quality_registry_rejects_unknown_resource_contract_field(tmp_path) -> None:
    registry_path = tmp_path / "quality.yaml"
    registry_path.write_text(
        """registry_version: 1
development_control: control
stages:
  - id: q0
    control: control
    variants:
      - id: control
        parent: null
        status: control
        overrides: {}
        resource_contract: {invented_limit: 1}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown resource contract keys"):
        load_quality_registry(registry_path)


def test_q2_training_log_records_effective_rollout_and_update_geometry(caplog) -> None:
    cfg = load_config(["env.n_envs=16", "ppo.batch_size=64", "ppo.n_epochs=3", "ppo.n_steps=128"])
    model = SimpleNamespace(n_steps=8, n_envs=16, ent_coef=0.01)
    caplog.set_level(logging.INFO, logger="training.setup.models")

    _log_effective_ppo(model, cfg)

    assert (
        "logical_n_envs=16 sb3_n_steps_per_env=8 aggregate_rollout_steps=128 "
        "batch_size=64 n_epochs=3 minibatches_per_epoch=2 updates_per_rollout=6"
    ) in caplog.text


def test_q2_training_log_rejects_effective_environment_count_mismatch() -> None:
    cfg = load_config(["env.n_envs=16"])
    model = SimpleNamespace(n_steps=8, n_envs=8, ent_coef=0.01)
    with pytest.raises(RuntimeError, match="effective PPO n_envs=8"):
        _log_effective_ppo(model, cfg)


def test_q6_has_three_explicit_arms_with_only_visibility_changed() -> None:
    registry = load_quality_registry()
    stage = next(stage for stage in registry.stages if stage.id == "q6_entry_fsm")
    arms = {str(variant.id): variant for variant in stage.variants}
    assert set(arms) == {"q6_fsm_visible", "q6_fsm_hidden", "q6_no_fsm_control"}
    assert arms["q6_fsm_visible"].overrides["env.entry_state_machine"] is True
    assert arms["q6_fsm_visible"].overrides["env.entry_state_observation"] is True
    assert arms["q6_fsm_hidden"].overrides["env.entry_state_machine"] is True
    assert arms["q6_fsm_hidden"].overrides["env.entry_state_observation"] is False
    assert arms["q6_no_fsm_control"].overrides["env.entry_state_machine"] is False
    assert all(str(variant.status) == "implementation_unverified" for variant in arms.values())
    for variant_id in arms:
        with pytest.raises(
            ValueError, match="not implemented and selectable|select the prior-stage winner"
        ):
            apply_quality_variant(OmegaConf.create({}), variant_id)


def test_unverified_hybrid_policy_remains_fail_closed() -> None:
    with pytest.raises(
        ValueError, match="not implemented and selectable|select the prior-stage winner"
    ):
        apply_quality_variant(OmegaConf.create({}), "q4b_hybrid_distribution")


def test_unverified_registry_arms_are_all_fail_closed() -> None:
    registry = load_quality_registry()
    unverified = [
        str(variant.id)
        for stage in registry.stages
        for variant in stage.variants
        if str(variant.get("status", "")) == "implementation_unverified"
    ]
    assert unverified
    for variant_id in unverified:
        with pytest.raises(
            ValueError, match="not implemented and selectable|select the prior-stage winner"
        ):
            apply_quality_variant(OmegaConf.create({}), variant_id)


def test_lagrangian_is_implemented_but_waits_for_q4_parent_freeze() -> None:
    base = OmegaConf.create({"ppo": {"n_steps": 8}})
    stage, overrides = get_quality_variant("q5_lagrangian")
    assert stage == "q5_constraints"
    assert overrides["constrained_rl.enabled"] is True
    registry = load_quality_registry()
    q5 = next(
        variant
        for stage_config in registry.stages
        for variant in stage_config.variants
        if variant.id == "q5_lagrangian"
    )
    assert q5.status == "implemented_unit_tested"
    with pytest.raises(ValueError, match="select the prior-stage winner"):
        apply_quality_variant(base, "q5_lagrangian")
    assert "constrained_rl" not in base


def test_q6_registers_hidden_fsm_comparison_arm() -> None:
    registry = load_quality_registry()
    q6 = next(stage for stage in registry.stages if stage.id == "q6_entry_fsm")
    hidden = next(variant for variant in q6.variants if variant.id == "q6_fsm_hidden")
    assert hidden.overrides["env.entry_state_machine"] is True
    assert hidden.overrides["env.entry_state_observation"] is False
