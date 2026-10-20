from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactId:
    kind: str
    digest: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.digest}"


def artifact_id(kind: str, semantic_inputs: Any) -> ArtifactId:
    payload = json.dumps(
        semantic_inputs, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return ArtifactId(kind, hashlib.sha256(payload).hexdigest())
