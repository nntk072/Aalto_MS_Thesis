from __future__ import annotations

import importlib

PUBLIC_PACKAGES = (
    "configuration",
    "artifacts",
    "market_data",
    "features",
    "trading",
    "simulation",
    "models",
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
