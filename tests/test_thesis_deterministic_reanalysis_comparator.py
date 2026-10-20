"""Deterministic report comparison requires shared provenance and raw outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from scripts.research.thesis.evidence.compare_deterministic_reanalysis import (
    IDENTITY_FIELDS,
    NUMERICAL_ARTIFACTS,
    SPLITS,
    compare_reanalyses,
)


def _write_reanalysis(path: Path) -> None:
    path.mkdir()
    source_run = path.parent / "source_run"
    source_run.mkdir(exist_ok=True)
    source_config = source_run / "config.yaml"
    source_config.write_text("source: true\n", encoding="utf-8")
    checkpoint = source_run / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    feature_cache = source_run / "features.parquet"
    feature_cache.write_bytes(b"feature-cache")
    source_feature_cache = {
        "identity": "features.parquet",
        "sha256": "feature-content-hash",
        "file_sha256": hashlib.sha256(feature_cache.read_bytes()).hexdigest(),
    }
    source_datasets = {
        "train_bars": {"sha256": "train-bars-hash"},
        "test_bars": {"sha256": "test-bars-hash"},
    }
    source_manifest = {
        "config": {"sha256": hashlib.sha256(source_config.read_bytes()).hexdigest()},
        "checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "feature_cache": source_feature_cache,
        "datasets": source_datasets,
        "action_schema": {"version": "action-v1", "key": "test"},
        "observation_schema": {"version": "observation-v1"},
        "stop_candidate_schema": {"version": "stop-v1"},
        "sequence_feature_schema": {"columns": ["ema", "reachable_r"], "width": 2},
        "execution_protocol": {"data_source": "frames"},
    }
    (source_run / "run_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    output_config = b"saved: true\n"
    (path / "config.yaml").write_bytes(output_config)
    identity: dict[str, Any] = {field: {"value": field} for field in IDENTITY_FIELDS}
    identity.update(
        {
            "artifact_role": "historical_reanalysis",
            "historical_holdout_inspected": True,
            "evaluation_provenance": {
                "source": {"available": True, "head": "test-head", "dirty_files": []},
                "runtime": {
                    "python": "3.x",
                    "platform": "test",
                    "python_hash_seed_env": None,
                },
                "python_executable": "/test/python",
            },
            "legacy_sequence_columns_used": False,
            "legacy_action_schema_opt_in": False,
            "legacy_observation_schema_opt_in": False,
            "legacy_sequence_schema_opt_in": False,
            "legacy_stop_schema_opt_in": False,
            "legacy_data_provenance_opt_in": False,
            "legacy_action_layout_selected": None,
            "dataset_parity_disposition": "verified",
            "checkpoint_manifest_hash_disposition": "verified",
            "action_schema": source_manifest["action_schema"],
            "observation_schema": source_manifest["observation_schema"],
            "stop_candidate_schema": source_manifest["stop_candidate_schema"],
            "sequence_feature_schema": source_manifest["sequence_feature_schema"],
            "sequence_feature_schema_disposition": "verified",
            "training_data_source": "frames",
            "stop_schema_disposition": "versioned",
            "policy_action_mode": "deterministic",
            "execution_seed": 50,
            "source_run": str(source_run),
            "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "feature_cache_identity": source_feature_cache["identity"],
            "feature_cache_hash": source_feature_cache["sha256"],
            "evaluated_feature_cache_path": str(feature_cache),
            "feature_cache_file_sha256": hashlib.sha256(feature_cache.read_bytes()).hexdigest(),
            "feature_cache_disposition": "verified",
            "source_feature_cache": source_feature_cache,
            "source_datasets": source_datasets,
            "source_feature_cache_replay_path": None,
            "reanalysis_config_sha256": hashlib.sha256(output_config).hexdigest(),
        }
    )
    (path / "reanalysis_manifest.json").write_text(json.dumps(identity), encoding="utf-8")
    for split in SPLITS:
        split_path = path / split
        split_path.mkdir()
        for name in NUMERICAL_ARTIFACTS:
            (split_path / name).write_text(f"{split}:{name}\n", encoding="utf-8")


def test_reanalysis_comparator_accepts_identical_packages(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)

    report = compare_reanalyses(first, second)

    assert report["status"] == "passed"
    assert len(report["numerical_artifacts"]) == len(SPLITS) * len(NUMERICAL_ARTIFACTS)
    assert report["errors"] == []


def test_reanalysis_comparator_rejects_different_execution_seeds(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_seed"] = 51
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "reanalysis identity differs: execution_seed" in report["errors"]


def test_reanalysis_comparator_accepts_explicit_legacy_stop_schema(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    source_manifest_path = tmp_path / "source_run" / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.pop("stop_candidate_schema")
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    for run_dir in (first, second):
        manifest_path = run_dir / "reanalysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stop_candidate_schema"] = None
        manifest["stop_schema_disposition"] = "legacy_opt_in"
        manifest["legacy_stop_schema_opt_in"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "passed"
    assert report["errors"] == []


def test_reanalysis_comparator_rejects_missing_stop_schema_without_opt_in(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    source_manifest_path = tmp_path / "source_run" / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.pop("stop_candidate_schema")
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    for run_dir in (first, second):
        manifest_path = run_dir / "reanalysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stop_candidate_schema"] = None
        manifest["stop_schema_disposition"] = "legacy_opt_in"
        manifest["legacy_stop_schema_opt_in"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "first missing source stop-candidate schema lacks explicit opt-in" in report["errors"]


def test_reanalysis_comparator_rejects_changed_raw_trade_artifact(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    (second / "testing" / "trades.csv").write_text("changed\n", encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "numerical artifact differs: testing/trades.csv" in report["errors"]


def test_reanalysis_comparator_rejects_different_checkpoint_identity(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"] = "different-checkpoint"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "reanalysis identity differs: checkpoint_sha256" in report["errors"]


def test_reanalysis_comparator_rejects_different_evaluator_source_identity(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_provenance"]["source"]["head"] = "other-head"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "reanalysis identity differs: evaluation_provenance" in report["errors"]


def test_reanalysis_comparator_accepts_crossed_hash_seed_when_artifacts_match(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    for run_dir, seed in ((first, "1"), (second, "12")):
        manifest_path = run_dir / "reanalysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation_provenance"]["runtime"]["python_hash_seed_env"] = seed
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "passed"
    assert report["python_hash_seed_env"] == {"first": "1", "second": "12"}


def test_reanalysis_comparator_rejects_malformed_hash_seed_record(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_provenance"]["runtime"]["python_hash_seed_env"] = 12
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="PYTHONHASHSEED provenance is missing or invalid"):
        compare_reanalyses(first, second)


def test_reanalysis_comparator_rejects_cache_identity_detached_from_source(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    for run_dir in (first, second):
        manifest_path = run_dir / "reanalysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["feature_cache_hash"] = "same-but-not-source-hash"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "first evaluated feature cache sha256 differs from source" in report["errors"]


def test_reanalysis_comparator_rejects_source_dataset_manifest_drift(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    source_manifest_path = tmp_path / "source_run" / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["datasets"]["train_bars"]["sha256"] = "changed-source-data"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "first dataset hash differs from original: train_bars" in report["errors"]


def test_reanalysis_comparator_rejects_sequence_schema_drift(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sequence_feature_schema"] = {"columns": ["different"], "width": 1}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert any(
        "sequence-feature schema differs from original" in error for error in report["errors"]
    )


def test_reanalysis_comparator_rejects_changed_evaluated_feature_cache(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _write_reanalysis(first)
    _write_reanalysis(second)
    manifest_path = second / "reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_cache = Path(manifest["evaluated_feature_cache_path"])
    feature_cache = second / "features.parquet"
    feature_cache.write_bytes(original_cache.read_bytes())
    manifest["evaluated_feature_cache_path"] = str(feature_cache)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    feature_cache.write_bytes(b"changed-cache")

    report = compare_reanalyses(first, second)

    assert report["status"] == "failed"
    assert "second evaluated feature cache does not match its recorded hash" in report["errors"]
