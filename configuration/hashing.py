from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()
