from __future__ import annotations

import hashlib

from artifacts.contracts import ArtifactRef, ArtifactStore
from artifacts.storage import FileSystemArtifactStore


class _InMemoryArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def put(self, reference: ArtifactRef, payload: bytes) -> None:
        self.values[reference.key] = payload

    def get(self, reference: ArtifactRef) -> bytes:
        return self.values[reference.key]


def test_filesystem_and_memory_stores_share_the_same_contract(tmp_path) -> None:
    payload = b"frozen run manifest"
    reference = ArtifactRef(
        key="runs/seed42/manifest",
        media_type="application/json",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    stores: list[ArtifactStore] = [
        FileSystemArtifactStore(tmp_path / "artifacts"),
        _InMemoryArtifactStore(),
    ]

    for store in stores:
        store.put(reference, payload)
        assert store.get(reference) == payload
