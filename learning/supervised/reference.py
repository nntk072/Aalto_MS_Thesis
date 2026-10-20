"""Small Torch classifier proving the non-RL learner and decision-provider path.

This backend is an architectural reference for supervised predictors. It has
no Gymnasium, simulator, or Stable-Baselines3 dependency. Production supervised
experiments should provide their own data preparation and model configuration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from decisioning.actions import build_discrete_trading_action_schema
from decisioning.contracts import (
    ActionSchemaRef,
    Decision,
    DecisionInput,
    DiscreteAction,
    PredictorMetadata,
)
from torch import nn

from artifacts.contracts import ArtifactRef
from learning.contracts import FitRequest, FitResult, Learner


class _ReferenceClassifier(nn.Module):
    def __init__(self, sequence_shape: tuple[int, int], account_width: int, n_actions: int) -> None:
        super().__init__()
        input_width = sequence_shape[0] * sequence_shape[1] + account_width
        self.network = nn.Sequential(
            nn.Linear(input_width, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, sequence: torch.Tensor, account: torch.Tensor) -> torch.Tensor:
        batch = sequence.shape[0]
        flattened = sequence.reshape(batch, -1)
        return cast(torch.Tensor, self.network(torch.cat((flattened, account), dim=1)))


def _action_schema_ref() -> ActionSchemaRef:
    schema = build_discrete_trading_action_schema()
    return ActionSchemaRef(
        version=schema.version,
        key=schema.key,
        coordinates=schema.names,
        kind="discrete",
    )


class ReferenceTorchProvider:
    """Wrap a fitted Torch classifier as a backend-neutral decision provider."""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_schema_version: str,
        action_schema: ActionSchemaRef | None = None,
    ) -> None:
        self.model = model.eval()
        self._metadata = PredictorMetadata(
            provider_id="reference-torch-supervised",
            model_family="torch_mlp_classifier",
            training_paradigm="supervised",
            input_schema_version=input_schema_version,
            action_schema=action_schema or _action_schema_ref(),
        )

    @property
    def metadata(self) -> PredictorMetadata:
        return self._metadata

    def predict(self, input: DecisionInput, *, deterministic: bool = True) -> Decision:
        del deterministic  # Classification uses deterministic argmax inference.
        if input.schema_version != self.metadata.input_schema_version:
            raise ValueError("decision input schema does not match the fitted predictor")
        sequence = np.asarray(input.observation["seq"], dtype=np.float32)
        account = np.asarray(input.observation["account"], dtype=np.float32)
        if sequence.ndim != 2 or account.ndim != 1:
            raise ValueError("reference classifier expects one sequence and one account vector")
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(sequence[None, ...]),
                torch.from_numpy(account[None, ...]),
            )
            index = int(torch.argmax(logits, dim=1).item())
        schema = self.metadata.action_schema
        if index >= len(schema.coordinates):
            raise ValueError("classifier output is outside its declared action schema")
        return Decision(
            action=DiscreteAction(index=index, name=schema.coordinates[index]),
            action_schema=schema,
        )


class ReferenceTorchLearner(Learner):
    """Fit a small categorical classifier and persist a self-describing artifact."""

    def __init__(
        self,
        observations: Mapping[str, np.ndarray[Any, Any]],
        labels: np.ndarray[Any, Any],
        *,
        checkpoint_path: str | Path,
        learning_rate: float = 1e-2,
    ) -> None:
        self.observations = observations
        self.labels = labels
        self.checkpoint_path = Path(checkpoint_path)
        self.learning_rate = learning_rate
        self.provider: ReferenceTorchProvider | None = None

    def fit(self, request: FitRequest) -> FitResult:
        sequence = np.asarray(self.observations.get("seq"), dtype=np.float32)
        account = np.asarray(self.observations.get("account"), dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int64)
        if sequence.ndim != 3 or account.ndim != 2 or len(sequence) != len(account):
            raise ValueError("training observations must have aligned [N,T,F] and [N,A] arrays")
        if labels.shape != (len(sequence),) or len(labels) == 0:
            raise ValueError("training labels must have one class for each observation")
        action_schema = _action_schema_ref()
        if np.any(labels < 0) or np.any(labels >= len(action_schema.coordinates)):
            raise ValueError("training labels must belong to the declared action schema")
        if not np.isfinite(sequence).all() or not np.isfinite(account).all():
            raise ValueError("training observations must be finite")

        torch.manual_seed(request.seed)
        sequence_shape = (int(sequence.shape[1]), int(sequence.shape[2]))
        model = _ReferenceClassifier(
            sequence_shape, account.shape[1], len(action_schema.coordinates)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        x_sequence = torch.from_numpy(sequence)
        x_account = torch.from_numpy(account)
        y = torch.from_numpy(labels)
        model.train()
        for _ in range(request.requested_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x_sequence, x_account), y)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_family": "torch_mlp_classifier",
                "training_paradigm": "supervised",
                "input_schema_version": request.input_schema_version,
                "dataset_id": request.dataset.dataset_id,
                "config_identity": request.config_identity,
                "seed": request.seed,
                "action_schema": {
                    "version": action_schema.version,
                    "key": action_schema.key,
                    "coordinates": action_schema.coordinates,
                    "kind": action_schema.kind,
                },
                "sequence_shape": tuple(sequence.shape[1:]),
                "account_width": int(account.shape[1]),
                "state_dict": model.state_dict(),
            },
            self.checkpoint_path,
        )
        digest = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        self.provider = ReferenceTorchProvider(
            model,
            input_schema_version=request.input_schema_version,
            action_schema=action_schema,
        )
        return FitResult(
            artifact=ArtifactRef(
                key=f"model:{self.checkpoint_path.name}",
                media_type="application/x-pytorch-state-dict",
                sha256=digest,
            ),
            dataset=request.dataset,
            config_identity=request.config_identity,
            input_schema_version=request.input_schema_version,
            requested_steps=request.requested_steps,
            training_steps=request.requested_steps,
        )


def load_reference_torch_provider(path: str | Path) -> ReferenceTorchProvider:
    """Load a reference artifact and validate its declared schemas before use."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("reference classifier artifact must contain a mapping")
    if payload.get("model_family") != "torch_mlp_classifier":
        raise ValueError("unsupported reference classifier model family")
    if payload.get("training_paradigm") != "supervised":
        raise ValueError("reference classifier artifact is not supervised")
    input_schema_version = payload.get("input_schema_version")
    sequence_shape = payload.get("sequence_shape")
    account_width = payload.get("account_width")
    action_data = payload.get("action_schema")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(input_schema_version, str)
        or not isinstance(sequence_shape, (tuple, list))
        or len(sequence_shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in sequence_shape)
        or not isinstance(account_width, int)
        or account_width <= 0
        or not isinstance(action_data, dict)
        or not isinstance(action_data.get("coordinates"), (tuple, list))
        or not all(isinstance(value, str) for value in action_data["coordinates"])
        or not isinstance(state_dict, dict)
    ):
        raise ValueError("reference classifier artifact metadata is incomplete")
    coordinates = tuple(action_data["coordinates"])
    if not coordinates:
        raise ValueError("reference classifier action schema is empty")
    action_schema = ActionSchemaRef(
        version=str(action_data.get("version", "")),
        key=str(action_data.get("key", "")),
        coordinates=coordinates,
        kind=str(action_data.get("kind", "")),
    )
    if not action_schema.version or not action_schema.key or action_schema.kind != "discrete":
        raise ValueError("reference classifier action schema is invalid")
    model = _ReferenceClassifier(
        (int(sequence_shape[0]), int(sequence_shape[1])), account_width, len(coordinates)
    )
    model.load_state_dict(state_dict)
    return ReferenceTorchProvider(
        model,
        input_schema_version=input_schema_version,
        action_schema=action_schema,
    )


def build_reference_torch_learner(
    observations: Mapping[str, np.ndarray[Any, Any]],
    labels: np.ndarray[Any, Any],
    *,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-2,
) -> Learner:
    """Construct the optional supervised reference backend."""
    return ReferenceTorchLearner(
        observations,
        labels,
        checkpoint_path=checkpoint_path,
        learning_rate=learning_rate,
    )


__all__ = [
    "ReferenceTorchLearner",
    "ReferenceTorchProvider",
    "build_reference_torch_learner",
    "load_reference_torch_provider",
]
