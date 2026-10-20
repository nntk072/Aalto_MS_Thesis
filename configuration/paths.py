from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    root: Path
    config: Path
    data: Path
    cache: Path
    outputs: Path
    logs: Path
    baseline: Path

    @classmethod
    def from_root(cls, root: str | Path) -> RepositoryPaths:
        resolved = Path(root).expanduser().resolve()
        return cls(
            resolved,
            resolved / "config",
            resolved / "data",
            resolved / "cache",
            resolved / "outputs",
            resolved / "logs",
            resolved / "baseline",
        )


def repository_root(anchor: str | Path | None = None) -> Path:
    """Find the repository root by walking upward to ``pyproject.toml``."""
    start = Path(anchor).resolve() if anchor is not None else Path.cwd().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"could not locate repository root from {start}")
