from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Validator = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class ArtifactKind:
    name: str
    schema_version: int
    validator: Validator | None = None


_REGISTRY: dict[str, ArtifactKind] = {}


def register_artifact_kind(kind: ArtifactKind) -> None:
    existing = _REGISTRY.get(kind.name)
    if existing is not None and existing != kind:
        raise ValueError(f"artifact kind already registered: {kind.name}")
    _REGISTRY[kind.name] = kind


def artifact_kind(name: str) -> ArtifactKind:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown artifact kind: {name}") from exc
