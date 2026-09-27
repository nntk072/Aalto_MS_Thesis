"""Checkpoint save must skip dashboard train hook (Py 3.12 mp pickle guard)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_rl.train.callbacks import save_ppo_checkpoint


class _FakeModel:
    def __init__(self) -> None:
        self.saved: list[tuple[str, tuple[str, ...] | None]] = []

    def save(self, path: Any, exclude: tuple[str, ...] | list[str] | None = None) -> None:
        ex = tuple(exclude) if exclude is not None else None
        self.saved.append((Path(path).name, ex))


def test_save_ppo_checkpoint_excludes_train_hook() -> None:
    model = _FakeModel()
    save_ppo_checkpoint(model, "/tmp/ppo_ckpt")
    assert model.saved == [("ppo_ckpt", ("train",))]
