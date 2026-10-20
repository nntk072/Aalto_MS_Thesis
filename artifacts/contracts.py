"""Backend-neutral identity for stored artifact payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Logical artifact identity, independent of its storage location."""

    key: str
    media_type: str
    sha256: str


class ArtifactStore(Protocol):
    """Read and write opaque payloads through a logical artifact reference."""

    def put(self, reference: ArtifactRef, payload: bytes) -> None: ...

    def get(self, reference: ArtifactRef) -> bytes: ...


__all__ = ["ArtifactRef", "ArtifactStore"]
