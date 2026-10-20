from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

PUBLIC_PACKAGES = (
    "configuration",
    "artifacts",
    "market_data",
    "features",
    "trading",
    "simulation",
    "models",
    "decisioning",
    "learning",
    "training",
    "evaluation",
    "evaluation.plotting",
    "experiments",
    "live",
    "integrations",
    "compute",
    "orchestra",
)
PUBLIC_CLIS = (
    "market_data.cli",
    "training.cli",
    "evaluation.cli",
    "experiments.cli",
    "compute.cli",
)


def test_public_packages_import() -> None:
    for name in PUBLIC_PACKAGES:
        assert importlib.import_module(name) is not None


def test_stable_cli_modules_import() -> None:
    for name in PUBLIC_CLIS:
        module = importlib.import_module(name)
        assert callable(module.main)


def test_top_level_packages_are_included_in_wheel() -> None:
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    wheel_packages = set(config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    public_roots = {name.split(".", maxsplit=1)[0] for name in PUBLIC_PACKAGES}

    assert public_roots <= wheel_packages
