from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class ConfigMergeError(ValueError):
    pass


def deep_merge(
    base: Mapping[str, Any], overlay: Mapping[str, Any], *, reject_type_changes: bool = False
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value, reject_type_changes=reject_type_changes)
        elif (
            reject_type_changes
            and key in result
            and result[key] is not None
            and value is not None
            and type(result[key]) is not type(value)
        ):
            raise ConfigMergeError(
                f"type change for {key!r}: {type(result[key]).__name__} -> {type(value).__name__}"
            )
        else:
            result[key] = deepcopy(value)
    return result
