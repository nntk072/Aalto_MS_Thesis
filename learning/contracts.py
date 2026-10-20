"""Workflow-facing contracts shared by learning paradigms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from artifacts.contracts import ArtifactRef
from market_data.contracts import DatasetRef


@dataclass(frozen=True, slots=True)
class FitRequest:
    """Reproducible fitting request without backend-specific model objects."""

    dataset: DatasetRef
    config_identity: str
    input_schema_version: str
    seed: int
    requested_steps: int

    def __post_init__(self) -> None:
        if not self.config_identity or not self.input_schema_version:
            raise ValueError("fit config identity and input schema version are required")
        if self.seed < 0:
            raise ValueError("fit seed must be nonnegative")
        if self.requested_steps < 0:
            raise ValueError("requested steps must be nonnegative")


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    step: int
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class FitResult:
    """Backend-neutral evidence produced by one fitting run.

    ``requested_steps`` is the additional work requested for this fit;
    ``training_steps`` is the backend's cumulative completed-step count after
    fitting, which can be lower than the request after an early stop.
    """

    artifact: ArtifactRef
    dataset: DatasetRef
    config_identity: str
    input_schema_version: str
    requested_steps: int
    training_steps: int
    history: tuple[TrainingPoint, ...] = ()
    validation_metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.requested_steps < 0 or self.training_steps < 0:
            raise ValueError("fit result step counts must be nonnegative")


class Learner(Protocol):
    """Fits model parameters from a stable request and returns artifact identity."""

    def fit(self, request: FitRequest) -> FitResult: ...


__all__ = ["FitRequest", "FitResult", "Learner", "TrainingPoint"]
