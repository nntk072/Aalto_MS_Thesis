from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifacts.contracts import ArtifactRef
from learning.contracts import FitRequest, FitResult, Learner
from learning.rl.learner import SB3Learner
from market_data.contracts import DatasetRef, MarketEventSchema


def _request() -> FitRequest:
    schema = MarketEventSchema("bar", "ohlc-v1", ("time", "close"), "time")
    dataset = DatasetRef(
        "tiny-dataset",
        "US100.cash",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
        "UTC",
        schema,
        "test-source:sha256:fixture",
    )
    return FitRequest(dataset, "config:sha256:fixture", "input-v1", seed=5, requested_steps=10)


class _TinyLearner:
    def fit(self, request: FitRequest) -> FitResult:
        payload = f"{request.dataset.dataset_id}:{request.seed}".encode()
        artifact = ArtifactRef(
            key="tiny/model",
            media_type="application/octet-stream",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        return FitResult(
            artifact=artifact,
            dataset=request.dataset,
            config_identity=request.config_identity,
            input_schema_version=request.input_schema_version,
            requested_steps=request.requested_steps,
            training_steps=request.requested_steps,
        )


def test_learner_implementations_share_the_backend_neutral_fit_contract() -> None:
    learner: Learner = _TinyLearner()

    result = learner.fit(_request())

    assert result.config_identity == "config:sha256:fixture"
    assert result.input_schema_version == "input-v1"
    assert result.dataset == _request().dataset
    assert result.training_steps == 10
    assert result.artifact.key == "tiny/model"


def test_fit_request_rejects_invalid_seed_and_empty_fit() -> None:
    request = _request()
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        FitRequest(
            request.dataset,
            request.config_identity,
            request.input_schema_version,
            seed=-1,
            requested_steps=1,
        )
    with pytest.raises(ValueError, match="requested steps"):
        FitRequest(
            request.dataset,
            request.config_identity,
            request.input_schema_version,
            seed=1,
            requested_steps=-1,
        )


def test_sb3_learner_persists_a_zero_remaining_step_checkpoint(tmp_path: Path) -> None:
    class _Model:
        num_timesteps = 10

        def learn(self, **_kwargs: object) -> None:
            raise AssertionError("a zero-step fit must not start another learning loop")

        def save(self, path: str | Path, *, exclude: list[str]) -> None:
            assert exclude == ["train", "learn"]
            Path(f"{path}.zip").write_bytes(b"saved checkpoint")

    checkpoint_path = tmp_path / "ppo_final"
    result = SB3Learner(
        _Model(),
        callbacks=(),
        checkpoint_path=checkpoint_path,
        log_interval=1,
        reset_num_timesteps=False,
    ).fit(replace(_request(), requested_steps=0))

    assert result.training_steps == 10
    assert result.artifact.sha256 == hashlib.sha256(b"saved checkpoint").hexdigest()
