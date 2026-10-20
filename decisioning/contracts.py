"""Stable inputs and outputs shared by decision-provider backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np

ObservationValue: TypeAlias = np.ndarray[Any, Any] | float | int


@dataclass(frozen=True)
class ActionSchemaRef:
    """Names and version of the action layout used by a decision."""

    version: str
    key: str
    coordinates: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class DecisionInput:
    """An observation plus its versioned semantic schema."""

    observation: Mapping[str, ObservationValue]
    schema_version: str


@dataclass(frozen=True)
class DiscreteAction:
    """A categorical choice with its stable category name."""

    index: int
    name: str


@dataclass(frozen=True)
class ContinuousAction:
    """A scalar continuous action in the provider's declared action schema."""

    value: float


@dataclass(frozen=True)
class StructuredAction:
    """Named continuous coordinates from a versioned action schema."""

    coordinates: Mapping[str, float]


DecisionAction: TypeAlias = DiscreteAction | ContinuousAction | StructuredAction


@dataclass(frozen=True)
class Decision:
    """A semantic action and the schema that defines its meaning."""

    action: DecisionAction
    action_schema: ActionSchemaRef


@dataclass(frozen=True)
class PredictorMetadata:
    """Backend-neutral description of a provider's input and output."""

    provider_id: str
    model_family: str
    training_paradigm: str
    input_schema_version: str
    action_schema: ActionSchemaRef


@dataclass(frozen=True)
class DecisionArtifactRef:
    """Stable identifier for a persisted provider or model bundle."""

    uri: str
    format: str
    version: str | None = None


class DecisionProvider(Protocol):
    """Produces one schema-validated decision for each observation."""

    @property
    def metadata(self) -> PredictorMetadata: ...

    def predict(self, input: DecisionInput, *, deterministic: bool = True) -> Decision: ...


def to_policy_action(decision: Decision) -> int | np.ndarray[Any, Any]:
    """Convert a decision back to the action representation expected by Gym."""
    action = decision.action
    if isinstance(action, DiscreteAction):
        return action.index
    if isinstance(action, ContinuousAction):
        return np.asarray([action.value], dtype=np.float32)
    ordered = decision.action_schema.coordinates
    if set(action.coordinates) != set(ordered):
        raise ValueError("structured decision coordinates disagree with its action schema")
    return np.asarray([action.coordinates[name] for name in ordered], dtype=np.float32)


__all__ = [
    "ActionSchemaRef",
    "ContinuousAction",
    "Decision",
    "DecisionArtifactRef",
    "DecisionInput",
    "DecisionProvider",
    "DiscreteAction",
    "PredictorMetadata",
    "StructuredAction",
    "to_policy_action",
]
