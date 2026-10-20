from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_package_directories_are_removed() -> None:
    for rel in ("quant_rl", "mt5_trading", "evaluation/analysis", "training/train"):
        path = ROOT / rel
        python_files = list(path.rglob("*.py")) if path.exists() else []
        assert not python_files, f"legacy Python modules remain under {rel}: {python_files[:5]}"
