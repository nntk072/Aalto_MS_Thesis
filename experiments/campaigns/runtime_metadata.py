"""Focused campaign execution runtime metadata services."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from typing import Any


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "omegaconf",
        "pyyaml",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "gpu": gpu,
    }
