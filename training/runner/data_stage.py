"""Focused training data preparation stage."""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from omegaconf import OmegaConf

from configuration.paths import repository_root
from evaluation.reporting.run_manifest import (
    array_sha256,
    dataset_record,
)
from features import (
    build_features,
    feature_cache_content_hash,
    feature_cache_path,
)
from features.context.policy_market import (
    attach_policy_market_context,
)
from market_data.loading.pipeline import build_tick_books, run_pipeline
from market_data.splits.definitions import (
    get_split_config,
    get_train_start_config,
    make_train_mask,
    split_bars,
    split_train_test,
)
from training.setup.data import (
    _tick_execution_split_coverage,
)
from training.setup.environments import (
    max_tick_quote_wait_ms,
)
from training.setup.environments import (
    randomized_fill_latency_ms as resolve_randomized_fill_latency_ms,
)
from training.setup.environments import (
    tick_execution_required as resolve_tick_execution_required,
)

from .stages import TrainingConfigurationStage, TrainingDataStage

log = logging.getLogger(__name__)


def _prepare_training_data(stage: TrainingConfigurationStage) -> TrainingDataStage:
    args = stage.args
    cfg = stage.cfg
    data = run_pipeline(cfg, force=args.force)
    primary_ticks = build_tick_books(cfg).get(str(cfg.data.primary))

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")
    primary_timeframe_bars = data[primary_sym]
    secondary_timeframe_bars = data.get(secondary_sym, {})

    cache_dir = Path(cfg.data.cache_dir)
    # Split dates first so z-score fit can be train-scoped (TI-4) and the
    # content-hash cache key includes that mask (TI-3).
    train_end, test_start = get_split_config(cfg)
    train_start = get_train_start_config(cfg)
    train_mask = make_train_mask(
        cast(pd.DatetimeIndex, primary_m1.index),
        train_end,
        train_start,
    )
    randomized_fill_latency_ms = resolve_randomized_fill_latency_ms(cfg)
    tick_execution_required = resolve_tick_execution_required(cfg)
    if tick_execution_required or randomized_fill_latency_ms > 0.0:
        execution_coverage = _tick_execution_split_coverage(
            primary_ticks,
            cast(pd.DatetimeIndex, primary_m1.index),
            train_mask,
            test_start,
            session_labels=(primary_m1["session"] if "session" in primary_m1.columns else None),
            max_latency_ns_exclusive=int(randomized_fill_latency_ms * 1_000_000),
            max_quote_wait_ms=max_tick_quote_wait_ms(cfg),
        )
        for split_name, report in execution_coverage.items():
            log.info(
                "Tick execution coverage [%s]: covered=%d/%d (%.4f%%)",
                split_name,
                report["covered_count"],
                report["order_count"],
                100.0 * report["coverage_fraction"],
            )
            if report["unavailable_count"]:
                raise SystemExit(
                    "Tick-only execution preflight failed for "
                    f"{split_name}: {report['unavailable_count']}/"
                    f"{report['order_count']} candidate order times lack a quote "
                    "that is available for every allowed latency draw within "
                    f"{max_tick_quote_wait_ms(cfg)} ms; first unavailable order "
                    f"time={report['first_unavailable_order_time']}. Training not started."
                )
    prebuilt_cache_value = OmegaConf.select(cfg, "data.prebuilt_feature_cache")
    if prebuilt_cache_value:
        feat_cache = Path(str(prebuilt_cache_value)).expanduser()
        if not feat_cache.is_absolute():
            feat_cache = repository_root(__file__) / feat_cache
        feat_cache = feat_cache.resolve()

        hash_path = Path(str(feat_cache) + ".hash")
        expected_cache_hash = feature_cache_content_hash(
            cfg,
            primary_m1,
            train_mask=train_mask,
            secondary=secondary_m1,
            timeframe_bars=primary_timeframe_bars,
            secondary_timeframe_bars=secondary_timeframe_bars,
        )
        if not feat_cache.is_file() or not hash_path.is_file():
            raise FileNotFoundError(
                f"Configured prebuilt feature cache or its .hash sidecar is missing: {feat_cache}"
            )
        actual_cache_hash = hash_path.read_text(encoding="utf-8").strip()
        if actual_cache_hash != expected_cache_hash:
            raise ValueError(
                "Configured prebuilt feature cache does not match the effective "
                f"data/config hash: {feat_cache} (expected {expected_cache_hash}, "
                f"found {actual_cache_hash})"
            )
        reference_mode = str(OmegaConf.select(cfg, "features.po3_reference_mode") or "m1_legacy")
        sweep_audit_path = Path(str(feat_cache) + ".htf_sweeps.parquet")
        if reference_mode == "htf_confirmed" and not sweep_audit_path.is_file():
            raise FileNotFoundError(
                "Configured prebuilt feature cache is missing its HTF sweep audit: "
                f"{sweep_audit_path}"
            )
        log.info("Using hash-validated prebuilt feature cache: %s", feat_cache)
    else:
        feat_cache = feature_cache_path(
            cache_dir,
            primary_sym,
            cfg,
            primary_m1,
            train_mask=train_mask,
            secondary=secondary_m1,
            timeframe_bars=primary_timeframe_bars,
            secondary_timeframe_bars=secondary_timeframe_bars,
        )
    features = build_features(
        primary_m1,
        secondary=secondary_m1,
        cfg=cfg,
        train_mask=train_mask,
        cache_path=feat_cache,
        timeframe_bars=primary_timeframe_bars,
        secondary_timeframe_bars=secondary_timeframe_bars,
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
    feature_columns = [str(column) for column in features.columns]

    # Split
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start, train_start
    )
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start, train_start)
    log.info(
        "Split: train=%d bars (<=%s)  test=%d bars (>=%s)",
        len(train_bars),
        train_end,
        len(test_bars),
        test_start,
    )

    # Slice for MVP: ~30 calendar days of the train split (full-day M1, not
    # the old 390 NY-only bars/day heuristic).
    if args.mvp and len(train_bars) > 0:
        cutoff = pd.Timestamp(train_bars.index[0]) + pd.Timedelta(days=30)
        n_keep = int((train_bars.index < cutoff).sum())
        if 0 < n_keep < len(train_bars):
            train_bars = train_bars.iloc[:n_keep]
            train_feat = train_feat.iloc[:n_keep]
            log.info("MVP: sliced training to %d bars (~30 calendar days)", len(train_bars))

    # Fingerprint the exact split passed to the environments before bundle mode
    # releases the frames. These records are reused for every seed in this run.
    dataset_records = {
        "train_bars": dataset_record(train_bars),
        "test_bars": dataset_record(test_bars),
        "train_features": dataset_record(train_feat),
        "test_features": dataset_record(test_feat),
    }
    if train_sec is not None:
        dataset_records["train_secondary"] = dataset_record(train_sec)
    if test_sec is not None:
        dataset_records["test_secondary"] = dataset_record(test_sec)
    if primary_ticks is not None:
        tick_digest = hashlib.sha256(
            "".join(
                array_sha256(array)
                for array in (primary_ticks._ts, primary_ticks._bid, primary_ticks._ask)
            ).encode()
        ).hexdigest()
        dataset_records["primary_ticks"] = {
            "rows": int(len(primary_ticks._ts)),
            "columns": ["timestamp_ns", "bid", "ask"],
            "start": (
                str(pd.to_datetime(primary_ticks._ts[0], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "end": (
                str(pd.to_datetime(primary_ticks._ts[-1], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "sha256": tick_digest,
        }

    vae_model = None
    env_vae = None
    pre_ny_by_date = None
    if args.use_vae:
        from learning.self_supervised.vae_pre_ny import (
            build_pre_ny_by_date,
            load_vae_from_checkpoint,
        )

        vae_model = load_vae_from_checkpoint(args.vae_path, config_path=args.vae_config)
        # Env keeps a CPU copy so SB3 moving the policy VAE to CUDA cannot break encode().
        env_vae = copy.deepcopy(vae_model).cpu().eval()
        pre_ny_by_date = build_pre_ny_by_date(
            primary_m1,
            seq_len=int(vae_model.encoder.seq_len),
            n_features=int(vae_model.encoder.n_features),
        )
        log.info(
            "VAE loaded from %s (%d pre-NY day sequences)",
            args.vae_path,
            len(pre_ny_by_date),
        )
    import gc

    gc.collect()
    log.info("Parent memory cleanup: released full-history pipeline objects")
    return TrainingDataStage(
        primary_ticks=primary_ticks,
        primary_sym=str(primary_sym),
        cache_dir=cache_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        schedule_path=schedule_path,
        feature_columns=feature_columns,
        feat_cache=feat_cache,
        train_bars=train_bars,
        train_feat=train_feat,
        test_bars=test_bars,
        test_feat=test_feat,
        train_sec=train_sec,
        test_sec=test_sec,
        vae_model=vae_model,
        env_vae=env_vae,
        pre_ny_by_date=pre_ny_by_date,
        dataset_records=dataset_records,
    )
