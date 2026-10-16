"""Validated access to the separate RL learning-quality experiment registry."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_QUALITY_REGISTRY = (
    Path(__file__).resolve().parents[2] / "config" / "quality_experiments.yaml"
)

_SELECTABLE_STATUSES = frozenset(
    {"control", "implemented_unit_tested", "existing_encoder_implementation"}
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
            OmegaConf.update(result, str(key), value, merge=True, force_add=True)
    result.quality_variant_id = variant_id
    return result


__all__ = [
    "DEFAULT_QUALITY_REGISTRY",
    "apply_quality_variant",
    "get_quality_variant",
    "load_quality_registry",
    "quality_variant_metadata",
]
