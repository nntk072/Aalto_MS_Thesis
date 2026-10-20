"""Dataset and feature reconstruction for provenance-checked reanalysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from configuration.paths import repository_root
from evaluation.reanalysis.provenance import (
    _evaluation_dataset_records,
    _feature_build_scope,
    _load_exact_source_feature_cache,
    _reanalysis_backend_disposition,
    _validate_reanalysis_datasets,
    _validate_reanalysis_feature_cache,
)
from evaluation.reporting.run_manifest import file_sha256
from features import build_features
from features.context.policy_market import attach_policy_market_context
from market_data.loading.pipeline import build_tick_books, run_pipeline
from market_data.splits.definitions import (
    get_split_config,
    get_train_start_config,
    make_train_mask,
    split_bars,
    split_train_test,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReanalysisDatasets:
    primary_ticks: Any
    features: pd.DataFrame
    train_bars: pd.DataFrame
    test_bars: pd.DataFrame
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_secondary: pd.DataFrame | None
    test_secondary: pd.DataFrame | None
    train_end: str
    test_start: str
    feature_cache: Path
    feature_cache_disposition: str
    evaluated_feature_cache_hash: str
    evaluated_feature_cache_file_sha256: str
    source_cache_replay_path: Path | None
    dataset_records: dict[str, Any]
    dataset_parity_disposition: str
    training_data_source: str
    backend_parity_disposition: str


def reconstruct_reanalysis_datasets(
    *,
    args: Any,
    run_dir: Path,
    cfg: DictConfig,
    source_manifest: dict[str, Any],
) -> ReanalysisDatasets:
    """Load, rebuild, split, fingerprint and compare the exact evaluation inputs."""
    source_feature_cache = source_manifest.get("feature_cache")
    source_datasets = source_manifest.get("datasets")
    source_execution_protocol = source_manifest.get("execution_protocol")
    selected_datasets = ["training"] if args.train_only_screen else None
    data = run_pipeline(cfg, force=args.force, datasets=selected_datasets)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")
    primary_timeframe_bars = data[cfg.data.primary]
    secondary_timeframe_bars = data.get(cfg.data.secondary, {})
    primary_symbol = str(cfg.data.primary)
    primary_ticks = build_tick_books(
        cfg,
        symbols=[primary_symbol],
        datasets=selected_datasets,
    ).get(primary_symbol)

    cache_dir = Path(cfg.data.cache_dir)
    train_end, test_start = get_split_config(cfg)
    train_start = get_train_start_config(cfg)
    source_cache_replay_path: Path | None = None
    if args.source_feature_cache and args.train_only_screen:
        raise ValueError(
            "--train-only-screen rebuilds features from training inputs to stay within memory; "
            "omit --source-feature-cache"
        )
    if args.source_feature_cache:
        config_record = source_manifest.get("config")
        source_config_path = run_dir / "config.yaml"
        expected_config_hash = (
            config_record.get("sha256") if isinstance(config_record, dict) else None
        )
        if not isinstance(expected_config_hash, str) or not expected_config_hash:
            raise ValueError("exact source-cache replay requires a saved config SHA-256")
        if (
            not source_config_path.is_file()
            or file_sha256(source_config_path) != expected_config_hash
        ):
            raise ValueError("saved run config does not match its manifest SHA-256")
        source_cache_replay_path = Path(args.source_feature_cache).expanduser()
        if not source_cache_replay_path.is_absolute():
            source_cache_replay_path = repository_root(__file__) / source_cache_replay_path
        train_mask = make_train_mask(pd.DatetimeIndex(primary_m1.index), train_end, train_start)
        (
            features,
            feature_cache,
            evaluated_feature_cache_hash,
            evaluated_feature_cache_file_sha256,
        ) = _load_exact_source_feature_cache(source_feature_cache, source_cache_replay_path)
        feature_cache_disposition = "source_cache_replay_verified"
    else:
        train_mask, feature_cache = _feature_build_scope(
            primary_m1,
            cfg,
            cache_dir,
            str(cfg.data.primary),
            train_end,
            secondary=secondary_m1,
            train_start=train_start,
            timeframe_bars=primary_timeframe_bars,
            secondary_timeframe_bars=secondary_timeframe_bars,
        )
        features = build_features(
            primary_m1,
            secondary=secondary_m1,
            cfg=cfg,
            train_mask=train_mask,
            cache_path=feature_cache,
            timeframe_bars=primary_timeframe_bars,
            secondary_timeframe_bars=secondary_timeframe_bars,
        )
        if args.train_only_screen:
            hash_path = Path(f"{feature_cache}.hash")
            evaluated_feature_cache_hash = hash_path.read_text(encoding="utf-8").strip()
            evaluated_feature_cache_file_sha256 = file_sha256(feature_cache)
            feature_cache_disposition = "training_partition_rebuilt_train_feature_digest_checked"
        else:
            (
                feature_cache_disposition,
                evaluated_feature_cache_hash,
                evaluated_feature_cache_file_sha256,
            ) = _validate_reanalysis_feature_cache(
                source_feature_cache,
                feature_cache,
                allow_legacy=bool(args.allow_legacy_data_provenance),
            )

    schedule_path = Path(str(cfg.data.session_schedule_file))
    if not schedule_path.is_absolute():
        schedule_path = repository_root(__file__) / schedule_path
    features = attach_policy_market_context(
        primary_m1,
        features,
        primary_ticks,
        schedule_path=schedule_path,
        session_start=str(cfg.session.start),
        session_end=str(cfg.session.end),
    )

    train_bars, test_bars, train_features, test_features = split_train_test(
        primary_m1, features, train_end, test_start, train_start
    )
    train_secondary = test_secondary = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_secondary, test_secondary = split_bars(
            secondary_m1, train_end, test_start, train_start
        )
    evaluated_datasets = _evaluation_dataset_records(
        train_bars=train_bars,
        train_features=train_features,
        test_bars=test_bars,
        test_features=test_features,
        train_secondary=train_secondary,
        test_secondary=test_secondary,
        primary_ticks=primary_ticks,
    )
    dataset_parity_disposition = _validate_reanalysis_datasets(
        source_datasets,
        evaluated_datasets,
        allow_legacy=bool(args.allow_legacy_data_provenance),
        train_only=bool(args.train_only_screen),
    )
    training_data_source, backend_parity_disposition = _reanalysis_backend_disposition(
        source_execution_protocol,
        evaluation_backend="frame",
    )
    log.info(
        "Evaluating train=%d bars (≤%s)  test=%d bars (≥%s) under FTMO eval guardrails",
        len(train_bars),
        train_end,
        len(test_bars),
        test_start,
    )
    return ReanalysisDatasets(
        primary_ticks=primary_ticks,
        features=features,
        train_bars=train_bars,
        test_bars=test_bars,
        train_features=train_features,
        test_features=test_features,
        train_secondary=train_secondary,
        test_secondary=test_secondary,
        train_end=train_end,
        test_start=test_start,
        feature_cache=feature_cache,
        feature_cache_disposition=feature_cache_disposition,
        evaluated_feature_cache_hash=evaluated_feature_cache_hash,
        evaluated_feature_cache_file_sha256=evaluated_feature_cache_file_sha256,
        source_cache_replay_path=source_cache_replay_path,
        dataset_records=evaluated_datasets,
        dataset_parity_disposition=dataset_parity_disposition,
        training_data_source=training_data_source,
        backend_parity_disposition=backend_parity_disposition,
    )


__all__ = ["ReanalysisDatasets", "reconstruct_reanalysis_datasets"]
