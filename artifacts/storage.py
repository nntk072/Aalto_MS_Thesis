"""Filesystem adapter for the artifact storage contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

from artifacts.atomic import write_bytes_atomic
from artifacts.contracts import ArtifactRef


class FileSystemArtifactStore:
    """Store payloads by opaque logical key, validating content on every read."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, reference: ArtifactRef, payload: bytes) -> None:
        self._validate(reference, payload)
        destination = self._path(reference)
        write_bytes_atomic(destination, payload)

    def get(self, reference: ArtifactRef) -> bytes:
        payload = self._path(reference).read_bytes()
        self._validate(reference, payload)
        return payload

    def _path(self, reference: ArtifactRef) -> Path:
        key_digest = hashlib.sha256(reference.key.encode("utf-8")).hexdigest()
        return self.root / key_digest[:2] / key_digest[2:]

    @staticmethod
    def _validate(reference: ArtifactRef, payload: bytes) -> None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != reference.sha256:
            raise ValueError(f"artifact payload checksum mismatch for {reference.key!r}: {actual}")


__all__ = ["FileSystemArtifactStore"]
