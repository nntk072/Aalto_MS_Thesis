from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = (
    "configuration",
    "artifacts",
    "market_data",
    "features",
    "trading",
    "simulation",
    "models",
    "training",
    "evaluation",
    "experiments",
    "live",
    "integrations",
    "compute",
    "orchestra",
)
LEGACY_PREFIXES = (
    "quant_rl",
    "mt5_trading",
    "evaluation.analysis",
    "simulation.envs",
    "training.train",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _production_files() -> list[Path]:
    return sorted(
        p
        for package in PRODUCTION
        for p in (ROOT / package).rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_production_never_imports_scripts() -> None:
    violations = []
    for path in _production_files():
        for name in _imports(path):
            if name == "scripts" or name.startswith("scripts."):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "production imports scripts:\n" + "\n".join(violations)


def test_production_has_no_legacy_imports() -> None:
    violations = []
    for path in _production_files():
        for name in _imports(path):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in LEGACY_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "legacy imports remain:\n" + "\n".join(violations)


DEPENDENCY_RULES = {
    "configuration": {"training", "evaluation", "experiments", "live"},
    "artifacts": {"training", "evaluation", "experiments", "live"},
    "market_data": {"features", "trading", "training", "evaluation"},
    "features": {"training", "evaluation", "live", "integrations"},
    "trading": {"training", "evaluation", "experiments", "scripts"},
    "simulation": {"training", "evaluation", "scripts"},
    "models": {"training"},
    "live": {"integrations"},
}


def test_package_dependency_direction() -> None:
    violations: list[str] = []
    for package, forbidden in DEPENDENCY_RULES.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                if name.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not violations, "forbidden package dependencies:\n" + "\n".join(violations)
