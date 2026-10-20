from __future__ import annotations

from pathlib import Path

from artifacts.atomic import write_json_atomic


def cache_metadata_path(entry: str | Path) -> Path:
    return Path(entry) / "manifest.json"


def mark_cache_complete(entry: str | Path, metadata: dict[str, object]) -> Path:
    root = Path(entry)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache_metadata_path(root), metadata)
    return root


def cache_is_complete(entry: str | Path) -> bool:
    return cache_metadata_path(entry).is_file()
