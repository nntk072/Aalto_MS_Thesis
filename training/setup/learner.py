"""Training-workflow composition for stable dataset and learner requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from learning import Learner, build_learner
from learning.contracts import FitRequest
from market_data.contracts import DatasetRef, MarketEventSchema


def build_fit_request(
    *,
    dataset_record: dict[str, Any],
    instrument_id: str,
    config: DictConfig,
    input_schema: dict[str, Any],
    seed: int,
    requested_steps: int,
) -> FitRequest:
    """Resolve a stable fitting request from prepared run identities."""
    digest = dataset_record.get("sha256")
    start_value, end_value = dataset_record.get("start"), dataset_record.get("end")
    columns_value = dataset_record.get("columns")
    if (
        not isinstance(digest, str)
        or not digest
        or not isinstance(start_value, str)
        or not isinstance(end_value, str)
        or not isinstance(columns_value, list)
    ):
        raise ValueError("training dataset record lacks a hash, bounds, or column schema")
    start = pd.Timestamp(start_value)
    end = pd.Timestamp(end_value)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("training dataset timestamps must be timezone-aware")
    end += pd.Timedelta(microseconds=1)
    fields = ("time", *(str(column) for column in columns_value if str(column) != "time"))
    event_schema = MarketEventSchema(
        event_type="bar",
        version="ohlc-event-v1",
        fields=fields,
        time_field="time",
    )
    dataset = DatasetRef(
        dataset_id=f"bars:{instrument_id}:{digest}",
        instrument_id=instrument_id,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        timezone=str(start.tzinfo),
        schema=event_schema,
        provenance_id=f"sha256:{digest}",
    )
    config_payload = OmegaConf.to_yaml(config, resolve=True).encode("utf-8")
    config_identity = hashlib.sha256(config_payload).hexdigest()
    schema_identity = hashlib.sha256(
        json.dumps(input_schema, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return FitRequest(
        dataset=dataset,
        config_identity=f"sha256:{config_identity}",
        input_schema_version=f"input:{schema_identity}",
        seed=seed,
        requested_steps=requested_steps,
    )


def build_training_learner(
    *,
    model: Any,
    callbacks: Any,
    checkpoint_path: str | Path,
    log_interval: int,
    reset_num_timesteps: bool,
) -> Learner:
    """Select the configured learner implementation for this RL workflow."""
    return build_learner(
        "stable_baselines3",
        model=model,
        callbacks=callbacks,
        checkpoint_path=checkpoint_path,
        log_interval=log_interval,
        reset_num_timesteps=reset_num_timesteps,
    )


__all__ = ["build_fit_request", "build_training_learner"]
