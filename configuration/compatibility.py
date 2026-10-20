from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import CURRENT_CONFIG_SCHEMA_VERSION, ConfigSchemaError


def migrate_config_mapping(
    config: Mapping[str, Any], *, from_version: int, to_version: int = CURRENT_CONFIG_SCHEMA_VERSION
) -> dict[str, Any]:
    if from_version == to_version:
        return dict(config)
    raise ConfigSchemaError(f"no config migration registered for v{from_version} -> v{to_version}")
