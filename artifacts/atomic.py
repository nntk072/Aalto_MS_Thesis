from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_replace(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return _atomic_replace(Path(path), text.encode(encoding))


def write_json_atomic(path: str | Path, value: Any) -> Path:
    return write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
