"""Small runtime factory for paradigm-specific learner implementations."""

from __future__ import annotations

from typing import Any

from learning.contracts import Learner


def build_learner(backend: str, **options: Any) -> Learner:
    """Construct one supported learner backend behind the stable learner port."""
    if backend == "stable_baselines3":
        from learning.rl.learner import build_sb3_learner

        return build_sb3_learner(**options)
    if backend == "supervised_torch_reference":
        from learning.supervised.reference import build_reference_torch_learner

        return build_reference_torch_learner(**options)
    raise ValueError(f"unsupported learning backend: {backend!r}")


__all__ = ["build_learner"]
