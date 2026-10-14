"""Regression tests for complete and content-addressed trial manifests."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from quant_rl.eval.run_manifest import (
    _source_state,
    array_sha256,
    build_training_run_manifest,
    dataframe_sha256,
    file_sha256,
    write_run_manifest,
)


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
        action_schema={"version": 1, "key": "6_dir"},
        feature_cache_path=cache,
        feature_cache_hash="cache-identity",
        checkpoint_path=checkpoint,
        execution_protocol={"fill_delay_ms": 0},
        metrics_protocol={"frequency": "bar_close_equity"},
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
    assert manifest["feature_cache"]["sha256"] == "cache-identity"

    output = repository / "run_manifest.json"
    write_run_manifest(output, manifest)
    assert json.loads(output.read_text()) == manifest
    assert not output.with_suffix(".json.tmp").exists()


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
            action_schema=None,
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
            action_schema=None,
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
            action_schema=None,
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
