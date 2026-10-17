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
from omegaconf import DictConfig, OmegaConf

STOP_CANDIDATE_SCHEMA_VERSION = "rq1-stop-menu-v1"
SEQUENCE_FEATURE_SCHEMA_VERSION = "sequence-feature-columns-v1"


def build_sequence_feature_schema(columns: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Describe the ordered numeric feature coordinates supplied to the policy."""
    names = [str(column) for column in columns]
    if not names or len(names) != len(set(names)):
        raise ValueError("sequence feature schema requires non-empty unique column names")
    payload = json.dumps(names, separators=(",", ":")).encode("utf-8")
    return {
        "version": SEQUENCE_FEATURE_SCHEMA_VERSION,
        "width": len(names),
        "columns": names,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_sequence_feature_schema(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> str:
    """Fail closed on missing or reordered model sequence coordinates."""
    if actual is None:
        if allow_legacy:
            return "legacy_unverified"
        raise ValueError(
            "source run manifest lacks the sequence feature schema; pass "
            "--allow-legacy-sequence-schema only for an explicitly unverified replay"
        )
    if actual != expected:
        raise ValueError("evaluated sequence feature schema differs from the training manifest")
    return "verified"


def build_stop_candidate_schema(cfg: DictConfig) -> dict[str, Any]:
    """Describe the effective structural stop-candidate contract in a run.

    This is stored both in the run manifest and the SB3 checkpoint. The feature
    cache identity still carries the full feature configuration and data hash;
    this compact record makes the stop-menu compatibility boundary explicit.
    """
    structure = OmegaConf.select(cfg, "features.structure", default={})
    if not isinstance(structure, (DictConfig, dict)):
        structure = {}
    enabled = bool(structure.get("enabled", False))
    count = int(structure.get("swing_candidate_count", 1)) if enabled else 1
    if not 1 <= count <= 3:
        raise ValueError("features.structure.swing_candidate_count must be between 1 and 3")
    timeframes = structure.get(
        "timeframes",
        OmegaConf.select(cfg, "features.structure_timeframes", default=("M5", "M15", "H1")),
    )
    if isinstance(timeframes, str):
        timeframes = (timeframes,)
    if not timeframes:
        timeframes = ("M5", "M15", "H1")
    point_size = OmegaConf.select(cfg, "costs.point_size", default=None)
    price_increment = structure.get("candidate_price_increment")
    if price_increment is None:
        price_increment = point_size
    resolved_features_raw = OmegaConf.to_container(cfg.features, resolve=True)
    resolved_features: dict[str, Any] = {}
    if isinstance(resolved_features_raw, dict):
        resolved_features = {str(key): value for key, value in resolved_features_raw.items()}
    resolved_strategy = OmegaConf.to_container(cfg.strategy, resolve=True)
    feature_source_keys = (
        "structure",
        "htf_timeframes",
        "include_pd_context",
        "include_po3",
        "include_fvg_ifvg",
        "fvg_timeframes",
        "ifvg_mtf",
        "include_po3_full",
        "include_strategy_state",
        "include_session_ohlc",
        "po3_htf",
        "po3_ltf",
        "po3_reference_timeframes",
    )
    feature_stop_sources = {
        key: resolved_features.get(key) for key in feature_source_keys if key in resolved_features
    }
    # Entry-level stop/target construction receives every configured HTF key
    # (the environment starts with these defaults and then merges encoder
    # windows). Record that effective menu so a custom timeframe cannot change
    # stop candidates while appearing schema-compatible.
    from quant_rl.envs.observation import DEFAULT_HTF_WINDOWS

    stop_menu_timeframes = dict(DEFAULT_HTF_WINDOWS)
    encoder_windows = OmegaConf.select(cfg, "encoder.windows", default={})
    if encoder_windows:
        for timeframe, window in dict(encoder_windows).items():
            stop_menu_timeframes[str(timeframe)] = int(window)

    return {
        "schema_version": STOP_CANDIDATE_SCHEMA_VERSION,
        "strategy_name": str(OmegaConf.select(cfg, "strategy.name", default="baseline")),
        "menu": (
            "recent_confirmed_unswept_mtf_swings_v1"
            if count > 1
            else "legacy_last_accepted_swing_v1"
        ),
        "enabled": enabled,
        "candidate_count_per_side_per_timeframe": count,
        "timeframes": [str(tf).upper() for tf in timeframes],
        "stop_menu_timeframes": [str(tf).upper() for tf in stop_menu_timeframes],
        "confirmation": "completed_source_bar_close",
        "staleness": "drop_swept_or_after_count_newer_distinct_confirmations",
        "candidate_families": [
            "strategy_specific_ifvg_fvg",
            "multi_htf_confirmed_swings",
            "prior_day_week_levels",
            "deviation_levels",
        ],
        "ifvg_candidate_sources": [
            "ifvg_far_edge",
            "ifvg_through_zone_extreme_when_observed",
        ],
        "feature_stop_sources": feature_stop_sources,
        "strategy_config": resolved_strategy,
        "deduplication": {
            "rule": "newest_confirmed_price_level_per_side",
            "price_increment": None if price_increment is None else float(price_increment),
        },
        "stop_buffer_points": float(OmegaConf.select(cfg, "env.sl_buffer_pts", default=0.0)),
        "minimum_distance": {
            "absolute_points": float(OmegaConf.select(cfg, "risk.min_sl_points", default=0.0)),
            "atr_multiple": float(OmegaConf.select(cfg, "risk.min_sl_atr_mult", default=0.0)),
        },
        "manipulation_filter": str(
            OmegaConf.select(cfg, "strategy.manipulation_filter", default="against")
        ),
        "menu_order": "ascending_distance_stable_source_insertion_ties",
        "selection_rule": "policy_fraction_selects_candidate_by_distance_with_stable_ties",
    }


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


def build_evaluation_provenance(repository: Path) -> dict[str, Any]:
    """Capture the code tree and runtime that produced a deterministic replay."""
    return {
        "source": _source_state(repository.resolve()),
        "runtime": _runtime_versions(),
        "python_executable": sys.executable,
    }


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
    sequence_feature_schema: dict[str, Any] | None = None,
    stop_candidate_schema: dict[str, Any] | None = None,
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
    if not stop_candidate_schema:
        raise ValueError("run manifest requires an explicit stop-candidate schema")
    if not isinstance(sequence_feature_schema, dict):
        raise ValueError("run manifest requires an explicit sequence feature schema")
    sequence_feature_schema = build_sequence_feature_schema(
        sequence_feature_schema.get("columns", [])
    )
    return {
        "manifest_version": 5,
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
        "sequence_feature_schema": sequence_feature_schema,
        "stop_candidate_schema": stop_candidate_schema,
        "training_architecture": dict(training_architecture or {}),
        "training_protocol": dict(training_protocol or {}),
        "feature_fit_scope": "train_mask_only",
        "feature_cache": {
            "path": None if feature_cache_path is None else str(feature_cache_path.resolve()),
            "identity": None if feature_cache_path is None else feature_cache_path.name,
            "sha256": feature_cache_hash,
            "file_sha256": (
                file_sha256(feature_cache_path) if feature_cache_path is not None else None
            ),
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
