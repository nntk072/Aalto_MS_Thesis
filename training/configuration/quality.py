"""Validated access to the separate RL learning-quality experiment registry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from configuration.paths import repository_root

DEFAULT_QUALITY_REGISTRY = repository_root(__file__) / "config" / "quality_experiments.yaml"

_SELECTABLE_STATUSES = frozenset(
    {"control", "implemented_unit_tested", "existing_encoder_implementation"}
)
_MISSING = object()


def _override_type_matches(expected: object, value: object) -> bool:
    if expected is None or value is None:
        return True
    if type(expected) is bool:
        return type(value) is bool
    if type(expected) is int:
        return type(value) is int
    if type(expected) is float:
        return type(value) in (int, float)
    return isinstance(value, type(expected))


def _validate_override_schema(registry: DictConfig) -> None:
    """Reject registry typos and type drift against the canonical default config."""
    from configuration import load_config

    base = load_config()
    for stage in registry.stages:
        for variant in stage.variants:
            overrides = variant.get("overrides", {})
            if not isinstance(overrides, DictConfig):
                raise ValueError(f"quality variant {variant.id!r} overrides must be a map")
            for key, value in overrides.items():
                path = str(key)
                expected = OmegaConf.select(base, path, default=_MISSING)
                if expected is _MISSING:
                    raise ValueError(
                        f"quality variant {variant.id!r} override {path!r} "
                        "does not exist in config/base/default.yaml"
                    )
                if not _override_type_matches(expected, value):
                    raise ValueError(
                        f"quality variant {variant.id!r} override {path!r} has "
                        f"type {type(value).__name__}; default config expects "
                        f"{type(expected).__name__}"
                    )
            resource_contract = variant.get("resource_contract", {})
            if not isinstance(resource_contract, Mapping):
                raise ValueError(f"quality variant {variant.id!r} resource_contract must be a map")
            unknown_contract_keys = set(resource_contract) - {
                "exact_n_envs",
                "expected_n_envs",
                "aggregate_rollout_steps",
            }
            if unknown_contract_keys:
                raise ValueError(
                    f"quality variant {variant.id!r} has unknown resource contract "
                    f"keys: {sorted(unknown_contract_keys)}"
                )
            if (
                "exact_n_envs" in resource_contract
                and type(resource_contract["exact_n_envs"]) is not bool
            ):
                raise ValueError(
                    f"quality variant {variant.id!r} resource_contract.exact_n_envs must be boolean"
                )
            for field in ("expected_n_envs", "aggregate_rollout_steps"):
                if field in resource_contract and (
                    type(resource_contract[field]) is not int or int(resource_contract[field]) <= 0
                ):
                    raise ValueError(
                        f"quality variant {variant.id!r} resource_contract.{field} "
                        "must be a positive integer"
                    )
            if bool(resource_contract.get("exact_n_envs", False)) and (
                "expected_n_envs" not in resource_contract
            ):
                raise ValueError(
                    f"quality variant {variant.id!r} exact_n_envs requires "
                    "resource_contract.expected_n_envs"
                )


def load_quality_registry(path: str | Path = DEFAULT_QUALITY_REGISTRY) -> DictConfig:
    """Load and validate staged quality variants without touching historical experiments."""
    loaded = OmegaConf.load(Path(path))
    if not isinstance(loaded, DictConfig):
        raise TypeError("quality registry root must be a mapping")
    registry = loaded
    stages = registry.get("stages")
    if not stages:
        raise ValueError("quality registry requires at least one staged experiment")
    variant_ids: set[str] = set()
    for stage in stages:
        stage_id = str(stage.get("id", ""))
        variants = stage.get("variants")
        if not stage_id or not variants:
            raise ValueError("every quality stage requires an id and variants")
        for variant in variants:
            variant_id = str(variant.get("id", ""))
            if not variant_id or variant_id in variant_ids:
                raise ValueError(f"missing or duplicate quality variant id: {variant_id!r}")
            parent = variant.get("parent")
            if (
                parent is not None
                and parent not in variant_ids
                and not str(parent).endswith("_winner")
            ):
                raise ValueError(
                    f"quality variant {variant_id!r} references unknown or future parent {parent!r}"
                )
            variant_ids.add(variant_id)
    if not registry.get("development_control"):
        raise ValueError("quality registry requires a development_control")
    for stage in stages:
        control = str(stage.get("control", ""))
        if control not in variant_ids and not control.endswith("_winner"):
            raise ValueError(f"quality stage {stage.id!r} has unknown control {control!r}")
    _validate_override_schema(registry)
    return registry


def get_quality_variant(
    variant_id: str,
    path: str | Path = DEFAULT_QUALITY_REGISTRY,
) -> tuple[str, DictConfig]:
    """Return the owning stage and dot-key overrides for one registered variant."""
    registry = load_quality_registry(path)
    for stage in registry.stages:
        for variant in stage.variants:
            if str(variant.id) == variant_id:
                overrides = variant.get("overrides", {})
                if not isinstance(overrides, DictConfig):
                    raise ValueError(f"quality variant {variant_id!r} overrides must be a map")
                return str(stage.id), overrides
    raise KeyError(f"unknown quality variant: {variant_id}")


def quality_variant_metadata(variant_id: str) -> dict[str, str | None]:
    """Return stage, direct parent, and status for provenance records."""
    registry = load_quality_registry()
    for stage in registry.stages:
        for variant in stage.variants:
            if str(variant.id) == variant_id:
                parent = variant.get("parent")
                return {
                    "id": variant_id,
                    "stage": str(stage.id),
                    "parent": None if parent is None else str(parent),
                    "status": str(variant.get("status", "unspecified")),
                }
    raise KeyError(f"unknown quality variant: {variant_id}")


def quality_variant_resource_contract(variant_id: str) -> dict[str, bool | int]:
    """Return validated runtime resource constraints for a quality variant."""
    registry = load_quality_registry()
    for stage in registry.stages:
        for variant in stage.variants:
            if str(variant.id) == variant_id:
                contract = variant.get("resource_contract", {})
                return {
                    str(key): value if type(value) is int else bool(value)
                    for key, value in contract.items()
                }
    raise KeyError(f"unknown quality variant: {variant_id}")


def validate_quality_resource_shape(
    variant_id: str,
    *,
    requested_n_envs: int,
    effective_n_envs: int,
    aggregate_rollout_steps: int,
) -> None:
    """Fail closed when resource scaling or CLI overrides change a Q2 arm."""
    contract = quality_variant_resource_contract(variant_id)
    expected_n_envs = contract.get("expected_n_envs")
    if contract.get("exact_n_envs", False):
        if requested_n_envs != expected_n_envs:
            raise ValueError(
                f"quality variant {variant_id!r} requires requested_n_envs="
                f"{expected_n_envs}, but config/CLI resolved {requested_n_envs}"
            )
        if effective_n_envs != expected_n_envs:
            raise ValueError(
                f"quality variant {variant_id!r} requires exactly {expected_n_envs} "
                f"environments, but resource scaling selected {effective_n_envs}; "
                "use an allocation that satisfies the registered resource contract"
            )
    expected_rollout = contract.get("aggregate_rollout_steps")
    if expected_rollout is not None and aggregate_rollout_steps != expected_rollout:
        raise ValueError(
            f"quality variant {variant_id!r} requires aggregate_rollout_steps="
            f"{expected_rollout}, but config/CLI resolved {aggregate_rollout_steps}"
        )


def apply_quality_variant(cfg: DictConfig, variant_id: str) -> DictConfig:
    """Apply a complete resolved parent lineage, refusing an unselected winner."""
    registry = load_quality_registry()
    variants = {str(variant.id): variant for stage in registry.stages for variant in stage.variants}
    if variant_id not in variants:
        raise KeyError(f"unknown quality variant: {variant_id}")
    lineage: list[DictConfig] = []
    current = variant_id
    while current:
        variant = variants.get(current)
        if variant is None:
            raise ValueError(
                f"quality variant {variant_id!r} depends on unresolved parent {current!r}; "
                "select the prior-stage winner before continuing"
            )
        parent = variant.get("parent")
        if parent is not None and str(parent).endswith("_winner"):
            raise ValueError(
                f"quality variant {variant_id!r} depends on unresolved parent {parent!r}; "
                "select the prior-stage winner before continuing"
            )
        status = str(variant.get("status", "unspecified"))
        if status not in _SELECTABLE_STATUSES:
            raise ValueError(
                f"quality variant {current!r} is not implemented and selectable (status={status!r})"
            )
        lineage.append(variant)
        current = str(parent) if parent is not None else ""
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(cloned, DictConfig):
        raise TypeError("quality base config must be a mapping")
    result = cloned
    for variant in reversed(lineage):
        for key, value in variant.get("overrides", {}).items():
            path = str(key)
            expected = OmegaConf.select(result, path, default=_MISSING)
            if expected is _MISSING:
                raise ValueError(
                    f"quality variant {variant_id!r} override {path!r} "
                    "does not exist in the supplied base config"
                )
            if not _override_type_matches(expected, value):
                raise ValueError(
                    f"quality variant {variant_id!r} override {path!r} has "
                    f"type {type(value).__name__}; supplied config expects "
                    f"{type(expected).__name__}"
                )
            OmegaConf.update(result, path, value, merge=True, force_add=False)
    result.quality_variant_id = variant_id
    return result


__all__ = [
    "DEFAULT_QUALITY_REGISTRY",
    "apply_quality_variant",
    "get_quality_variant",
    "load_quality_registry",
    "quality_variant_metadata",
    "quality_variant_resource_contract",
    "validate_quality_resource_shape",
]
