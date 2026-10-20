"""Stable decision-provider adapter for existing Stable-Baselines3 policies."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from decisioning.contracts import (
    ActionSchemaRef,
    ContinuousAction,
    Decision,
    DecisionInput,
    DiscreteAction,
    PredictorMetadata,
    StructuredAction,
)


class SB3DecisionProvider:
    """Wrap an SB3-compatible ``predict`` model without owning its lifecycle."""

    def __init__(
        self,
        model: Any,
        *,
        input_schema_version: str,
        provider_id: str = "sb3-policy",
    ) -> None:
        self._model = model
        self._input_schema_version = input_schema_version
        self._action_schema = self._resolve_action_schema(model)
        self._metadata = PredictorMetadata(
            provider_id=provider_id,
            model_family=type(model).__name__,
            training_paradigm="reinforcement_learning",
            input_schema_version=input_schema_version,
            action_schema=self._action_schema,
        )

    @property
    def metadata(self) -> PredictorMetadata:
        return self._metadata

    def predict(self, input: DecisionInput, *, deterministic: bool = True) -> Decision:
        if input.schema_version != self._input_schema_version:
            raise ValueError(
                f"observation schema {input.schema_version!r} differs from provider schema "
                f"{self._input_schema_version!r}"
            )
        action = self._model.predict(input.observation, deterministic=deterministic)[0]
        return Decision(
            action=self._decode_action(action),
            action_schema=self._action_schema,
        )

    def _decode_action(self, action: Any) -> DiscreteAction | ContinuousAction | StructuredAction:
        if self._action_schema.kind == "discrete":
            values = np.asarray(action).reshape(-1)
            if values.size != 1:
                raise ValueError("SB3 discrete action must contain exactly one category")
            index = int(values[0])
            if index < 0 or index >= len(self._action_schema.coordinates):
                raise ValueError(f"SB3 discrete action {index} is outside the action schema")
            return DiscreteAction(index=index, name=self._action_schema.coordinates[index])

        values = np.asarray(action, dtype=np.float32).reshape(-1)
        names = self._action_schema.coordinates
        if values.size != len(names):
            raise ValueError(
                f"SB3 action width {values.size} differs from schema width {len(names)}"
            )
        if len(names) == 1:
            return ContinuousAction(value=float(values[0]))
        return StructuredAction(
            coordinates={name: float(value) for name, value in zip(names, values, strict=True)}
        )

    @staticmethod
    def _resolve_action_schema(model: Any) -> ActionSchemaRef:
        action_space = getattr(model, "action_space", None)
        saved = getattr(model, "_action_schema", None)
        if isinstance(action_space, spaces.Discrete):
            if isinstance(saved, dict) and saved.get("kind") == "discrete":
                names = tuple(str(name) for name in saved.get("coordinates", ()))
                version = str(saved.get("version", ""))
                key = str(saved.get("key", ""))
            else:
                raise ValueError("SB3 discrete model requires a persisted action schema")
            if len(names) != action_space.n or not version or not key:
                raise ValueError("SB3 discrete action schema is incomplete or has the wrong width")
            return ActionSchemaRef(version=version, key=key, coordinates=names, kind="discrete")

        shape = getattr(action_space, "shape", None)
        if not isinstance(action_space, spaces.Box) or shape is None or len(shape) != 1:
            raise TypeError("SB3 provider requires a one-dimensional Box or Discrete action space")
        if not isinstance(saved, dict):
            raise ValueError("SB3 continuous model requires a persisted action schema")
        names = tuple(str(name) for name in saved.get("coordinates", ()))
        version = str(saved.get("version", ""))
        key = str(saved.get("key", ""))
        if len(names) != int(shape[0]) or not version or not key:
            raise ValueError("SB3 continuous action schema is incomplete or has the wrong width")
        return ActionSchemaRef(version=version, key=key, coordinates=names, kind="continuous")


__all__ = ["SB3DecisionProvider"]
