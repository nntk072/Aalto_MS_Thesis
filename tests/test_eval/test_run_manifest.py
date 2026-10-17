"""Regression tests for complete and content-addressed trial manifests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from gymnasium import Env, spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from quant_rl.eval.run_manifest import (
    _source_state,
    array_sha256,
    build_evaluation_provenance,
    build_sequence_feature_schema,
    build_stop_candidate_schema,
    build_training_run_manifest,
    dataframe_sha256,
    file_sha256,
    validate_sequence_feature_schema,
    write_run_manifest,
)

_ACTION_SCHEMA = {
    "version": "action-v1",
    "key": "test",
    "width": 2,
    "coordinates": ["entry", "exit"],
    "has_direction": False,
    "has_sl_mode": False,
    "has_tp_mode": False,
    "has_multi_tp": False,
    "has_simplex": False,
}
_OBSERVATION_SCHEMA = {
    "version": "observation-v1",
    "observation_width": 2,
    "observation_coordinates": ["equity", "position"],
    "entry_state_visibility": False,
}


@pytest.mark.parametrize(
    ("action_schema", "observation_schema", "message"),
    [
        (None, _OBSERVATION_SCHEMA, "explicit action schema"),
        ({**_ACTION_SCHEMA, "width": 3}, _OBSERVATION_SCHEMA, "action schema requires unique"),
        (
            {key: value for key, value in _ACTION_SCHEMA.items() if key != "has_direction"},
            _OBSERVATION_SCHEMA,
            "requires explicit action flags",
        ),
        (_ACTION_SCHEMA, None, "explicit observation schema"),
        (
            _ACTION_SCHEMA,
            {**_OBSERVATION_SCHEMA, "observation_width": 3},
            "observation schema requires unique",
        ),
        (
            _ACTION_SCHEMA,
            {
                key: value
                for key, value in _OBSERVATION_SCHEMA.items()
                if key != "entry_state_visibility"
            },
            "requires entry_state_visibility",
        ),
    ],
)
def test_manifest_requires_complete_action_and_observation_schemas(
    tmp_path: Path,
    action_schema: dict[str, Any] | None,
    observation_schema: dict[str, Any] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_training_run_manifest(
            repository=tmp_path,
            config_path=tmp_path / "config.yaml",
            datasets={},
            split={},
            seed=1,
            completed_steps=1,
            action_schema=action_schema,
            observation_schema=observation_schema,
            sequence_feature_schema=None,
            stop_candidate_schema=None,
            feature_cache_path=None,
            feature_cache_hash=None,
            checkpoint_path=None,
            execution_protocol={},
            metrics_protocol={},
            artifact_paths={},
        )


def test_evaluation_provenance_records_source_and_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import quant_rl.eval.run_manifest as run_manifest

    monkeypatch.setattr(
        run_manifest,
        "_source_state",
        lambda repository: {"available": True, "head": "abc123", "dirty_files": []},
    )
    monkeypatch.setattr(
        run_manifest,
        "_runtime_versions",
        lambda: {"python": "3.12.0", "platform": "test"},
    )

    provenance = build_evaluation_provenance(tmp_path)

    assert provenance["source"]["available"] is True
    assert provenance["source"]["head"] == "abc123"
    assert provenance["runtime"]["python"] == "3.12.0"
    assert provenance["python_executable"]


def test_manifest_records_config_dataset_checkpoint_and_split_hashes(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    config = repository / "config.yaml"
    config.write_text("seed: 7\n")
    checkpoint = repository / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    cache = repository / "features.parquet"
    cache.write_bytes(b"feature-cache")
    index = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=index)
    datasets = {
        "train_bars": frame,
        "train_features": frame,
        "test_bars": frame,
        "test_features": frame,
    }
    training_log = repository / "training_log.csv"
    training_log.write_text("step\n100\n")
    trade_log = repository / "trade_log.json"
    trade_log.write_text("[]\n")

    manifest = build_training_run_manifest(
        repository=repository,
        config_path=config,
        datasets=datasets,
        split={"train_end": "2024-01-01", "test_start": "2024-01-02"},
        seed=7,
        completed_steps=100,
        action_schema={
            "version": "overlay-action-v2",
            "key": "6_dir",
            "width": 6,
            "coordinates": ["direction", "intensity", "stop", "risk", "target", "exit"],
            "has_direction": True,
            "has_sl_mode": False,
            "has_tp_mode": False,
            "has_multi_tp": False,
            "has_simplex": False,
        },
        observation_schema={
            "version": "account-state-v2",
            "observation_width": 28,
            "observation_coordinates": [f"coordinate_{i}" for i in range(28)],
            "entry_state_visibility": False,
        },
        sequence_feature_schema=build_sequence_feature_schema(
            ["ema_9", "po3_ifvg_active", "reachable_r"]
        ),
        stop_candidate_schema={
            "schema_version": "rq1-stop-menu-v1",
            "menu": "recent_confirmed_unswept_mtf_swings_v1",
        },
        training_protocol={"seed": 7, "requested_steps": 100, "completed_steps": 100},
        feature_cache_path=cache,
        feature_cache_hash="cache-identity",
        checkpoint_path=checkpoint,
        execution_protocol={"fill_delay_ms": 0},
        metrics_protocol={"frequency": "bar_close_equity"},
        training_architecture={
            "arch": "lite_tcn",
            "encoder_class": "LiteTCNEncoder",
            "encoder_channels": 96,
            "encoder_dilations": [1, 4, 10],
            "encoder_latent_dim": 128,
            "policy_head_architecture": [128, 64],
            "policy_parameter_count": 12345,
            "vector_backend": "serial",
            "logical_n_envs": 64,
            "physical_worker_count": 1,
        },
        artifact_paths={
            "checkpoint": "model.zip",
            "training_log": "training_log.csv",
            "trade_log": "trade_log.json",
        },
    )

    assert manifest["config"]["sha256"] == file_sha256(config)
    assert manifest["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert manifest["datasets"]["train_bars"]["sha256"] == dataframe_sha256(frame)
    assert manifest["split"] == {"train_end": "2024-01-01", "test_start": "2024-01-02"}
    assert manifest["action_schema"]["key"] == "6_dir"
    assert manifest["observation_schema"]["version"] == "account-state-v2"
    assert manifest["training_protocol"]["requested_steps"] == 100
    assert manifest["feature_cache"]["sha256"] == "cache-identity"
    assert manifest["feature_cache"]["file_sha256"] == file_sha256(cache)
    assert manifest["manifest_version"] == 5
    assert manifest["sequence_feature_schema"]["columns"] == [
        "ema_9",
        "po3_ifvg_active",
        "reachable_r",
    ]
    assert manifest["sequence_feature_schema"]["width"] == 3
    assert manifest["stop_candidate_schema"]["menu"] == "recent_confirmed_unswept_mtf_swings_v1"
    assert manifest["training_architecture"]["physical_worker_count"] == 1
    assert manifest["training_architecture"]["encoder_channels"] == 96

    output = repository / "run_manifest.json"
    write_run_manifest(output, manifest)
    assert json.loads(output.read_text()) == manifest
    assert not output.with_suffix(".json.tmp").exists()


def test_sequence_feature_schema_is_order_sensitive_and_fails_closed() -> None:
    expected = build_sequence_feature_schema(["ema_9", "reachable_r"])
    same = build_sequence_feature_schema(["ema_9", "reachable_r"])
    reordered = build_sequence_feature_schema(["reachable_r", "ema_9"])

    assert expected["sha256"] == same["sha256"]
    assert expected["sha256"] != reordered["sha256"]
    assert validate_sequence_feature_schema(same, expected) == "verified"
    with pytest.raises(ValueError, match="differs from the training manifest"):
        validate_sequence_feature_schema(reordered, expected)
    with pytest.raises(ValueError, match="lacks the sequence feature schema"):
        validate_sequence_feature_schema(None, expected)
    assert (
        validate_sequence_feature_schema(None, expected, allow_legacy=True) == "legacy_unverified"
    )


def _expanded_stop_schema_config():
    return OmegaConf.create(
        {
            "features": {
                "include_pd_context": True,
                "structure": {
                    "enabled": True,
                    "timeframes": ["M5", "M15", "H1"],
                    "swing_candidate_count": 3,
                    "candidate_price_increment": None,
                },
            },
            "encoder": {"windows": {"m30": 8}},
            "costs": {"point_size": 0.1},
            "env": {"sl_buffer_pts": 1.0},
            "risk": {"min_sl_points": 10.0, "min_sl_atr_mult": 0.5},
            "strategy": {"manipulation_filter": "against"},
        }
    )


def test_stop_candidate_schema_records_expanded_multitimeframe_contract() -> None:
    cfg = _expanded_stop_schema_config()
    schema = build_stop_candidate_schema(cfg)
    assert schema["menu"] == "recent_confirmed_unswept_mtf_swings_v1"
    assert schema["candidate_count_per_side_per_timeframe"] == 3
    assert schema["timeframes"] == ["M5", "M15", "H1"]
    assert schema["stop_menu_timeframes"] == ["M5", "M15", "H1", "M30"]
    assert schema["deduplication"]["price_increment"] == 0.1
    assert schema["feature_stop_sources"]["structure"]["swing_candidate_count"] == 3
    assert schema["strategy_config"]["manipulation_filter"] == "against"


class _SchemaRoundTripEnv(Env[npt.NDArray[np.float32], np.int64]):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space: spaces.Discrete[np.int64] = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(
        self, action: np.int64
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


def test_stop_candidate_schema_survives_sb3_checkpoint_roundtrip(tmp_path) -> None:
    schema = build_stop_candidate_schema(_expanded_stop_schema_config())
    model = PPO(
        "MlpPolicy", _SchemaRoundTripEnv(), n_steps=8, batch_size=4, n_epochs=1, device="cpu"
    )
    setattr(model, "stop_candidate_schema", schema)
    path = tmp_path / "checkpoint"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    assert getattr(restored, "stop_candidate_schema") == schema


def test_manifest_fails_closed_without_split_boundaries(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n")
    frame = pd.DataFrame({"close": [1.0]})

    with pytest.raises(ValueError, match="train_end and test_start"):
        build_training_run_manifest(
            repository=tmp_path,
            config_path=config,
            datasets={"train": frame},
            split={"train_end": "2024-01-01"},
            seed=1,
            completed_steps=1,
            action_schema=_ACTION_SCHEMA,
            observation_schema=_OBSERVATION_SCHEMA,
            feature_cache_path=None,
            feature_cache_hash=None,
            checkpoint_path=None,
            execution_protocol={},
            metrics_protocol={},
            artifact_paths={},
        )


def test_manifest_requires_exact_train_test_data_and_saved_artifacts(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n")
    with pytest.raises(ValueError, match="train/test bars and feature identities"):
        build_training_run_manifest(
            repository=tmp_path,
            config_path=config,
            datasets={"train": pd.DataFrame({"close": [1.0]})},
            split={"train_end": "2024-01-01", "test_start": "2024-01-02"},
            seed=1,
            completed_steps=1,
            action_schema=_ACTION_SCHEMA,
            observation_schema=_OBSERVATION_SCHEMA,
            feature_cache_path=None,
            feature_cache_hash=None,
            checkpoint_path=None,
            execution_protocol={},
            metrics_protocol={},
            artifact_paths={},
        )


def test_manifest_requires_an_existing_checkpoint(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("seed: 1\n")
    cache = tmp_path / "features.parquet"
    cache.write_bytes(b"cache")
    frame = pd.DataFrame({"close": [1.0]})
    datasets = {
        name: frame for name in ("train_bars", "train_features", "test_bars", "test_features")
    }
    with pytest.raises(FileNotFoundError, match="checkpoint required"):
        build_training_run_manifest(
            repository=tmp_path,
            config_path=config,
            datasets=datasets,
            split={"train_end": "2024-01-01", "test_start": "2024-01-02"},
            seed=1,
            completed_steps=1,
            action_schema=_ACTION_SCHEMA,
            observation_schema=_OBSERVATION_SCHEMA,
            feature_cache_path=cache,
            feature_cache_hash="hash",
            checkpoint_path=None,
            execution_protocol={},
            metrics_protocol={},
            artifact_paths={
                "checkpoint": "model.zip",
                "training_log": "training_log.csv",
                "trade_log": "trade_log.json",
            },
        )


def test_array_hash_includes_shape_and_dtype() -> None:
    base = np.asarray([1, 2], dtype=np.int32)
    assert array_sha256(base) != array_sha256(base.reshape(1, 2))
    assert array_sha256(base) != array_sha256(base.astype(np.int64))


def test_source_state_hashes_files_inside_untracked_directories(tmp_path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    nested = repository / "outputs" / "nested"
    nested.mkdir(parents=True)
    (repository / "outputs" / "manifest.json").write_text("{}\n")
    (nested / "run.csv").write_text("step\n1\n")
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, "a" * 40 + "\n", ""),
            subprocess.CompletedProcess([], 0, "?? outputs/\n", ""),
            subprocess.CompletedProcess(
                [], 0, "outputs/manifest.json\noutputs/nested/run.csv\n", ""
            ),
        )
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    state = _source_state(repository)

    files = {record["path"]: record.get("sha256") for record in state["dirty_files"]}
    assert files == {
        "outputs/manifest.json": file_sha256(repository / "outputs/manifest.json"),
        "outputs/nested/run.csv": file_sha256(nested / "run.csv"),
    }
