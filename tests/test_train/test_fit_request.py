from __future__ import annotations

from datetime import UTC, datetime

from omegaconf import OmegaConf

from training.setup.learner import build_fit_request


def test_training_fit_request_carries_data_config_input_and_seed_identity() -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    request = build_fit_request(
        dataset_record={
            "sha256": "a" * 64,
            "start": timestamp.isoformat(),
            "end": timestamp.isoformat(),
            "columns": ["open", "high", "low", "close"],
        },
        instrument_id="US100.cash",
        config=OmegaConf.create({"agent": {"arch": "tcn"}}),
        input_schema={"observation": {"version": "account-state-v1"}},
        seed=42,
        requested_steps=0,
    )

    assert request.dataset.instrument_id == "US100.cash"
    assert request.dataset.schema.fields == ("time", "open", "high", "low", "close")
    assert request.dataset.start == timestamp
    assert request.dataset.end > request.dataset.start
    assert request.dataset.provenance_id == f"sha256:{'a' * 64}"
    assert request.config_identity.startswith("sha256:")
    assert request.input_schema_version.startswith("input:")
    assert request.seed == 42
    assert request.requested_steps == 0
