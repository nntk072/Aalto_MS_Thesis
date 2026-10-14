"""Reproducibility manifest helpers for immutable training artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash column names, index and typed cell values deterministically."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in frame.columns]).encode())
    row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype=np.uint64
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def dataset_record(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe and fingerprint one concrete dataset split."""
    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "start": str(frame.index.min()) if len(frame) else None,
        "end": str(frame.index.max()) if len(frame) else None,
        "sha256": dataframe_sha256(frame),
    }


def array_sha256(array: np.ndarray[Any, Any]) -> str:
    """Hash an array with its dtype and shape included."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_state(repository: Path) -> dict[str, Any]:
    """Capture the source commit and dirty path list when Git is available."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "head": None, "dirty_paths": []}
    dirty_files: list[dict[str, Any]] = []
    for line in status:
        relative = line[3:] if len(line) >= 4 else ""
        path = repository / relative
        if path.is_dir():
            listing = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    relative,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.splitlines()
            relatives = sorted(listing)
        else:
            relatives = [relative]
        for dirty_relative in relatives:
            dirty_path = repository / dirty_relative
            record: dict[str, Any] = {"path": dirty_relative}
            if dirty_path.is_file():
                record["sha256"] = file_sha256(dirty_path)
            dirty_files.append(record)
    return {
        "available": True,
        "head": head,
        "dirty_files": sorted(dirty_files, key=lambda item: str(item["path"])),
    }


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in ("numpy", "pandas", "torch", "stable-baselines3", "gymnasium"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        import torch

        versions["cuda_available"] = str(bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            versions["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        versions["cuda_available"] = "unknown"
    versions["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return versions


def build_training_run_manifest(
    *,
    repository: Path,
    config_path: Path,
    datasets: dict[str, pd.DataFrame],
    dataset_records: dict[str, dict[str, Any]] | None = None,
    split: dict[str, str],
    seed: int,
    completed_steps: int,
    action_schema: dict[str, Any] | None,
    observation_schema: dict[str, Any] | None = None,
    feature_cache_path: Path | None,
    feature_cache_hash: str | None,
    checkpoint_path: Path | None,
    execution_protocol: dict[str, Any],
    metrics_protocol: dict[str, Any],
    artifact_paths: dict[str, str],
    training_architecture: dict[str, Any] | None = None,
    training_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a source/data/config/checkpoint/runtime manifest for one trial."""
    resolved_dataset_records = dict(dataset_records or {})
    for name, frame in sorted(datasets.items()):
        resolved_dataset_records[name] = dataset_record(frame)
    required_split = {"train_end", "test_start"}
    if not required_split.issubset(split):
        raise ValueError(
            "run manifest split must include train_end and test_start; "
            f"missing {sorted(required_split.difference(split))}"
        )
    required_datasets = {"train_bars", "train_features", "test_bars", "test_features"}
    missing_datasets = required_datasets.difference(resolved_dataset_records)
    if missing_datasets:
        raise ValueError(
            "run manifest requires exact train/test bars and feature identities; "
            f"missing {sorted(missing_datasets)}"
        )
    for name in sorted(required_datasets):
        record = resolved_dataset_records[name]
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"run manifest dataset {name} requires a SHA-256 identity")
    if feature_cache_path is None or not feature_cache_path.is_file():
        raise FileNotFoundError(
            f"feature cache required for manifest is missing: {feature_cache_path}"
        )
    if not feature_cache_hash:
        raise ValueError("run manifest requires a feature cache identity")
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint required for manifest is missing: {checkpoint_path}")
    required_artifacts = {"checkpoint", "training_log", "trade_log"}
    missing_artifacts = required_artifacts.difference(artifact_paths)
    if missing_artifacts:
        raise ValueError(
            "run manifest requires checkpoint, training-log, and trade-log paths; "
            f"missing {sorted(missing_artifacts)}"
        )
    return {
        "manifest_version": 3,
        "repository": str(repository.resolve()),
        "source": _source_state(repository),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": file_sha256(config_path),
        },
        "datasets": resolved_dataset_records,
        "split": dict(split),
        "seed": int(seed),
        "completed_steps": int(completed_steps),
        "action_schema": action_schema,
        "observation_schema": observation_schema,
        "training_architecture": dict(training_architecture or {}),
        "training_protocol": dict(training_protocol or {}),
        "feature_fit_scope": "train_mask_only",
        "feature_cache": {
            "path": None if feature_cache_path is None else str(feature_cache_path.resolve()),
            "identity": None if feature_cache_path is None else feature_cache_path.name,
            "sha256": feature_cache_hash,
        },
        "checkpoint": {
            "path": None if checkpoint_path is None else str(checkpoint_path.resolve()),
            "sha256": (file_sha256(checkpoint_path) if checkpoint_path is not None else None),
        },
        "execution_protocol": dict(execution_protocol),
        "metrics_protocol": dict(metrics_protocol),
        "runtime": {
            "versions": _runtime_versions(),
            "python_executable": sys.executable,
        },
        "artifact_paths": dict(artifact_paths),
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically write a JSON manifest so interrupted writes are not valid artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
