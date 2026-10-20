from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from decisioning.contracts import DecisionInput, DiscreteAction
from evaluation.rollout.model import evaluate_provider
from learning.build import build_learner
from learning.contracts import FitRequest
from learning.supervised.reference import (
    ReferenceTorchLearner,
    load_reference_torch_provider,
)
from market_data.contracts import DatasetRef, MarketEventSchema


def _request() -> FitRequest:
    schema = MarketEventSchema("bar", "ohlc-v1", ("time", "close"), "time")
    dataset = DatasetRef(
        "reference-supervised-test",
        "US100.cash",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        "UTC",
        schema,
        "test-source:sha256:supervised",
    )
    return FitRequest(dataset, "config:reference", "account-state-v1", seed=7, requested_steps=8)


def _learner(checkpoint: Path) -> ReferenceTorchLearner:
    rng = np.random.default_rng(13)
    observations = {
        "seq": rng.normal(size=(16, 10, 4)).astype(np.float32),
        "account": np.zeros((16, 6), dtype=np.float32),
    }
    labels = np.zeros(16, dtype=np.int64)
    learner = cast(
        ReferenceTorchLearner,
        build_learner(
            "supervised_torch_reference",
            observations=observations,
            labels=labels,
            checkpoint_path=checkpoint,
        ),
    )
    return learner


def test_supervised_learner_fits_self_describing_artifact_and_provider(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classifier.pt"
    learner = _learner(checkpoint)

    result = learner.fit(_request())

    assert checkpoint.is_file()
    assert result.artifact.sha256
    assert result.training_steps == 8
    provider = load_reference_torch_provider(checkpoint)
    assert provider.metadata.training_paradigm == "supervised"
    assert provider.metadata.model_family == "torch_mlp_classifier"
    decision = provider.predict(
        DecisionInput(
            observation={
                "seq": np.zeros((10, 4), dtype=np.float32),
                "account": np.zeros(6, dtype=np.float32),
            },
            schema_version="account-state-v1",
        ),
    )
    assert isinstance(decision.action, DiscreteAction)
    assert (
        decision.action.name == provider.metadata.action_schema.coordinates[decision.action.index]
    )


def test_supervised_provider_runs_through_shared_evaluation_path(tmp_path: Path) -> None:
    learner = _learner(tmp_path / "classifier.pt")
    learner.fit(_request())
    provider = load_reference_torch_provider(tmp_path / "classifier.pt")

    index = pd.date_range("2025-01-06 16:30", periods=32, freq="1min", tz="Etc/GMT-3")
    close = 20_000.0 + np.arange(len(index), dtype=np.float64) * 0.1
    bars = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tickvol": 100.0,
            "vol": 0.0,
            "spread": 0.6,
        },
        index=index,
    )
    features = pd.DataFrame(0.0, index=index, columns=["f0"])
    features["last_swing_low"] = close - 5.0
    features["last_swing_high"] = close + 5.0

    result = evaluate_provider(
        provider,
        bars,
        features,
        obs_window=10,
        observation_schema_version="account-state-v1",
        block_overnight=False,
    )

    assert result["n_steps"] > 0
