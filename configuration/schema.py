from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CURRENT_CONFIG_SCHEMA_VERSION = 1


class ConfigSchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResolvedConfigMetadata:
    schema_version: int = CURRENT_CONFIG_SCHEMA_VERSION
    source: str | None = None


def reject_unknown_keys(
    values: Mapping[str, Any], allowed: set[str], *, path: str = "config"
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigSchemaError(f"unknown keys at {path}: {unknown}")
