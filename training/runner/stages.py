"""Typed stage results for the training preparation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingConfigurationStage:
    """Resolved command, domain config, device and training-risk callback."""

    args: Any
    cfg: Any
    device: Any
    lagrangian_cb: Any


@dataclass(slots=True)
class TrainingDataStage:
    """Prepared train/test inputs and their cache and provenance identities."""

    primary_ticks: Any
    primary_sym: str
    cache_dir: Path
    train_start: Any
    train_end: Any
    test_start: Any
    schedule_path: Path
    feature_columns: list[str]
    feat_cache: Path
    train_bars: Any
    train_feat: Any
    test_bars: Any
    test_feat: Any
    train_sec: Any
    test_sec: Any
    vae_model: Any
    env_vae: Any
    pre_ny_by_date: Any
    dataset_records: dict[str, Any]


@dataclass(slots=True)
class TrainingRunAssetsStage:
    """Output layout, worker specs, and bounded monitoring data for a run."""

    seeds: list[int]
    is_multi_seed: bool
    run_dir: Path
    model_dir: Path
    data_source: str
    obs_mmap: str | None
    bundle_path: Path | None
    test_bundle_path: Path | None
    train_env_spec: Any
    train_eval_spec: Any
    test_eval_spec: Any
    train_bar_count: int
    test_bar_count: int
    best_bars: Any
    best_features: Any
    dashboard_bars: Any
    dashboard_features: Any


@dataclass(slots=True)
class TrainingPreparation:
    """Composition of the independently prepared training workflow stages."""

    configuration: TrainingConfigurationStage
    data: TrainingDataStage
    run_assets: TrainingRunAssetsStage


__all__ = [
    "TrainingConfigurationStage",
    "TrainingDataStage",
    "TrainingRunAssetsStage",
    "TrainingPreparation",
]
