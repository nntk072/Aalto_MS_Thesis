from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .retention import RetentionClass


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    kind: str
    artifact_id: str
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    retention: RetentionClass = RetentionClass.RUN_RETAINED
    parents: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["retention"] = str(self.retention)
        return result

    @classmethod
    def read(cls, path: str | Path) -> ArtifactManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["retention"] = RetentionClass(data["retention"])
        data["parents"] = tuple(data.get("parents", ()))
        return cls(**data)
