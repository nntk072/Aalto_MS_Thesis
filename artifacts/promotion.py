from __future__ import annotations

import shutil
from pathlib import Path

from .manifest import ArtifactManifest
from .retention import RetentionClass


def promote_to_baseline(
    source: str | Path, destination: str | Path, manifest: ArtifactManifest
) -> Path:
    if manifest.retention is not RetentionClass.BASELINE_PROMOTABLE:
        raise ValueError("artifact is not baseline-promotable")
    src, dst = Path(source), Path(destination)
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    return dst
