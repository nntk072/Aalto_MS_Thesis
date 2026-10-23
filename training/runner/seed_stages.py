"""Typed handoff results for the per-seed training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learning.contracts import FitResult


@dataclass(slots=True)
class PreparedSeedRun:
    """Seed-local runtime components assembled before fitting."""

    seed: int
    seed_dir: Path
    model_dir: Path
    environment: Any
    model: Any
    requested_timesteps: int


@dataclass(slots=True)
class TrainedSeedRun:
    """Fitted model and provenance returned by the selected learner."""

    model_path: Path
    fit_result: FitResult
    abort_reason: str | None


@dataclass(slots=True)
class EvaluatedSeedRun:
    """Train and report-only test evaluation results for one seed."""

    train_result: dict[str, Any]
    test_result: dict[str, Any]
    train_metrics: Any
    test_metrics: Any
    train_gate: dict[str, Any]
    train_activity_gate: dict[str, Any]


@dataclass(slots=True)
class PersistedSeedRun:
    """Paths and schema identities written for a completed seed."""

    manifest_path: Path
    sequence_feature_schema: dict[str, Any]


__all__ = [
    "EvaluatedSeedRun",
    "PersistedSeedRun",
    "PreparedSeedRun",
    "TrainedSeedRun",
]
