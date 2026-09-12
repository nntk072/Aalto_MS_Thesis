"""Snapshot and restore the workspace before orchestra edits files."""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from typing import Any

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    "mlruns",
    ".cache",
}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".parquet"}
_MANIFEST = "manifest.json"
_ARCHIVE = "tree.tar.gz"


def _rel(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def iter_backup_paths(workspace: Path) -> list[Path]:
    """Source files to snapshot (skips venv, git, orchestra state, caches)."""
    found: list[Path] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = _rel(path, workspace)
        parts = rel.parts
        if any(part in _SKIP_DIRS for part in parts):
            continue
        if parts[:2] == ("orchestra", "state"):
            continue
        if parts and parts[0] == "data":
            continue
        if rel.suffix in _SKIP_SUFFIXES:
            continue
        found.append(rel)
    return found


def create_snapshot(workspace: Path, dest: Path) -> dict[str, Any]:
    """Write a tar.gz of the workspace and return backup metadata."""
    dest.mkdir(parents=True, exist_ok=True)
    files = iter_backup_paths(workspace)
    archive = dest / _ARCHIVE
    with tarfile.open(archive, "w:gz") as tar:
        for rel in files:
            tar.add(workspace / rel, arcname=rel.as_posix())
    slim = {
        "archive": str(archive),
        "files": len(files),
        "created_at": time.time(),
    }
    (dest / _MANIFEST).write_text(
        json.dumps({**slim, "paths": [p.as_posix() for p in files]}, indent=2),
        encoding="utf-8",
    )
    return slim


def restore_snapshot(workspace: Path, dest: Path) -> int:
    """Restore a snapshot. Returns the number of files extracted.

    Deletes workspace files (same include rules) that were not in the snapshot
    so orchestra-created files disappear.
    """
    manifest = json.loads((dest / _MANIFEST).read_text(encoding="utf-8"))
    keep = set(manifest["paths"])
    for rel in iter_backup_paths(workspace):
        if rel.as_posix() not in keep:
            (workspace / rel).unlink()
    archive = dest / _ARCHIVE
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe archive member: {name}")
        try:
            tar.extractall(workspace, filter="data")
        except TypeError:
            tar.extractall(workspace)
    return len(keep)


def list_slots(task_root: Path) -> list[str]:
    """Backup slot names for a task (legacy flat snapshot is ``start``)."""
    if (task_root / _MANIFEST).is_file():
        return ["start"]
    if not task_root.is_dir():
        return []
    return sorted(p.name for p in task_root.iterdir() if p.is_dir() and (p / _MANIFEST).is_file())


def resolve_slot(task_root: Path, step: str) -> Path:
    """Resolve ``5`` / ``implementation`` / ``00-start`` to a snapshot directory."""
    key = step.strip().lower().replace("_", "-").replace(" ", "-")
    if key in {"0", "start"}:
        key = "00-start"
    slots = list_slots(task_root)
    if key == "start" or key == "00-start":
        if (task_root / "00-start" / _MANIFEST).is_file():
            return task_root / "00-start"
        if (task_root / _MANIFEST).is_file():
            return task_root
    exact = task_root / key
    if (exact / _MANIFEST).is_file():
        return exact
    if key.isdigit():
        prefix = f"{int(key):02d}-"
        matches = [s for s in slots if s.startswith(prefix)]
        if matches:
            return task_root / matches[-1]
    named = [s for s in slots if key in s]
    if len(named) == 1:
        return task_root / named[0]
    if named:
        return task_root / named[-1]
    raise FileNotFoundError(f"No backup slot {step!r} under {task_root}")
