from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ACTIVE_SURFACES = (
    "Makefile",
    "docker-compose.yml",
    "tools",
    "scripts/compute",
    ".github/workflows",
)

FORBIDDEN_RUNTIME_PATHS = (
    "scripts/docker/",
    "scripts/triton/",
    "scripts/setup/",
    "scripts/matrix/",
    "training.train.",
    "evaluation.analysis",
    "quant_rl",
    "mt5_trading",
)


def _surface_files() -> list[Path]:
    files: list[Path] = []
    for rel in ACTIVE_SURFACES:
        path = ROOT / rel
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return files


def test_execution_surfaces_do_not_reference_legacy_paths() -> None:
    violations: list[str] = []
    for path in _surface_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_RUNTIME_PATHS:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not violations, "legacy execution-surface paths remain:\n" + "\n".join(violations)


def test_canonical_check_surface_exists() -> None:
    check = ROOT / "tools" / "check"
    assert check.is_file()
    text = check.read_text(encoding="utf-8")
    for gate in (
        "deps",
        "quality",
        "architecture",
        "contracts",
        "unit",
        "build",
        "integration",
        "parity",
        "training-smoke",
        "fast",
        "full",
        "gpu",
        "triton",
    ):
        assert gate in text
