"""Prepare data, configuration, and environment assets for an RL training run."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from configuration.paths import repository_root

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import copy
import hashlib
import logging
from typing import Any, cast

import pandas as pd
from omegaconf import OmegaConf

from compute.environment.device import get_device, scale_training_cfg
from evaluation.reporting.export import build_run_dir
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
from market_data.loading.ticks import ticks_covering
from market_data.splits.definitions import (
    get_split_config,
    get_train_start_config,
    make_train_mask,
    split_bars,
    split_train_test,
)
from training.configuration.cli import _load_merged_config, parse_train_args
from training.runner.seed import _setup_rngs
from training.setup.callbacks import (
    _build_lagrangian_callback,
    _dashboard_block,
)
from training.setup.data import (
    _publish_obs_memmap,
    _tick_execution_split_coverage,
)
from training.setup.environments import (
    _max_tick_quote_wait_ms,
    _randomized_fill_latency_ms,
    _strategy_from_cfg,
    _tick_execution_required,
    _training_episodic,
)
from training.setup.models import (
    _validate_q5_resume_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfigurationStage:
    args: Any
    cfg: Any
    device: Any
    lagrangian_cb: Any


@dataclass(slots=True)
class TrainingDataStage:
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
class TrainingPreparation:
    args: Any
    cfg: Any
    device: Any
    lagrangian_cb: Any
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
    dataset_records: dict[str, Any]


def _prepare_training_configuration() -> TrainingConfigurationStage:
    args = parse_train_args()

    if args.use_vae and not args.vae_path:
        raise SystemExit("--use-vae requires --vae-path pointing at a trained VAE .pth")
    if args.walk_forward and args.use_vae:
        raise SystemExit(
            "Walk-forward with an externally fitted VAE has unverified fold provenance; "
            "provide a fold-scoped VAE fitting pipeline before enabling this combination"
        )

    _setup_rngs(args.seed)

    cfg = _load_merged_config(args)
    try:
        _validate_q5_resume_state(cfg, resume_checkpoint=args.resume_checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.arch is None:
        args.arch = str(cfg.agent.get("arch", "tcn"))
    elif getattr(args, "quality_variant", None) and args.arch != str(cfg.agent.get("arch", "tcn")):
        raise SystemExit(
            f"--arch={args.arch} conflicts with the selected quality variant encoder "
            f"{cfg.agent.get('arch', 'tcn')}"
        )

    device = get_device(cfg.get("device"))
    requested_n_envs = int(cfg.env.n_envs)
    quality_variant = str(cfg.get("quality_variant_id", ""))
    exact_n_envs = False
    preserve_ppo_geometry = False
    if quality_variant:
        from training.configuration.quality import quality_variant_resource_contract

        resource_contract = quality_variant_resource_contract(quality_variant)
        exact_n_envs = bool(resource_contract.get("exact_n_envs", False))
        preserve_ppo_geometry = exact_n_envs and "aggregate_rollout_steps" in resource_contract
    cfg.quality_execution.requested_n_envs = requested_n_envs
    cfg.quality_execution.exact_n_envs = exact_n_envs
    scale_training_cfg(
        cfg,
        device,
        args.arch,
        preserve_ppo_geometry=preserve_ppo_geometry,
    )
    cfg.quality_execution.effective_n_envs = int(cfg.env.n_envs)
    if quality_variant and exact_n_envs:
        from training.configuration.quality import validate_quality_resource_shape

        try:
            validate_quality_resource_shape(
                quality_variant,
                requested_n_envs=requested_n_envs,
                effective_n_envs=int(cfg.env.n_envs),
                aggregate_rollout_steps=int(cfg.env.n_envs * cfg.ppo.n_steps),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    log.info(
        "Using device: %s  n_envs=%s  ppo.batch_size=%s  ppo.n_steps=%s  n_epochs=%s",
        device,
        cfg.env.n_envs,
        cfg.ppo.batch_size,
        cfg.ppo.n_steps,
        cfg.ppo.n_epochs,
    )

    # Override for MVP mode
    if args.mvp:
        log.info("MVP mode: using first 30 days of training data")
        cfg.training.use_m1_only = True
        cfg.training.max_days = 30
    lagrangian_cb = _build_lagrangian_callback(cfg, algo=args.algo, mvp=args.mvp)
    if lagrangian_cb is not None:
        log.info(
            "Q5 primal-dual PPO enabled: episode_cost_budget=%.4f dual_lr=%.4f lambda_max=%.3f",
            lagrangian_cb.episode_cost_budget,
            lagrangian_cb.dual_learning_rate,
            lagrangian_cb.lambda_max,
        )
    return TrainingConfigurationStage(
        args=args, cfg=cfg, device=device, lagrangian_cb=lagrangian_cb
    )


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
    randomized_fill_latency_ms = _randomized_fill_latency_ms(cfg)
    tick_execution_required = _tick_execution_required(cfg)
    if tick_execution_required or randomized_fill_latency_ms > 0.0:
        execution_coverage = _tick_execution_split_coverage(
            primary_ticks,
            cast(pd.DatetimeIndex, primary_m1.index),
            train_mask,
            test_start,
            session_labels=(primary_m1["session"] if "session" in primary_m1.columns else None),
            max_latency_ns_exclusive=int(randomized_fill_latency_ms * 1_000_000),
            max_quote_wait_ms=_max_tick_quote_wait_ms(cfg),
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
                    f"{_max_tick_quote_wait_ms(cfg)} ms; first unavailable order "
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
        from models.auxiliary.vae_pre_ny import build_pre_ny_by_date, load_vae_from_checkpoint

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


def _prepare_run_assets(
    config_stage: TrainingConfigurationStage, data_stage: TrainingDataStage
) -> TrainingPreparation:
    args = config_stage.args
    cfg = config_stage.cfg
    device = config_stage.device
    lagrangian_cb = config_stage.lagrangian_cb
    primary_ticks = data_stage.primary_ticks
    primary_sym = data_stage.primary_sym
    cache_dir = data_stage.cache_dir
    train_start = data_stage.train_start
    train_end = data_stage.train_end
    test_start = data_stage.test_start
    schedule_path = data_stage.schedule_path
    feature_columns = data_stage.feature_columns
    feat_cache = data_stage.feat_cache
    train_bars = data_stage.train_bars
    train_feat = data_stage.train_feat
    test_bars = data_stage.test_bars
    test_feat = data_stage.test_feat
    train_sec = data_stage.train_sec
    test_sec = data_stage.test_sec
    vae_model = data_stage.vae_model
    env_vae = data_stage.env_vae
    pre_ny_by_date = data_stage.pre_ny_by_date
    dataset_records = data_stage.dataset_records
    # Setup output directory for model
    seeds = list(args.seeds) if args.seeds else [args.seed]
    if args.resume_checkpoint and (args.algo != "ppo" or len(seeds) != 1):
        raise SystemExit("--resume-checkpoint currently supports one PPO seed per run")
    is_multi_seed = len(seeds) > 1
    quality_variant = str(cfg.get("quality_variant_id", ""))
    if quality_variant:
        run_name = (
            f"quality_{quality_variant}_seeds_{args.arch}"
            if is_multi_seed
            else f"quality_{quality_variant}_seed{args.seed}_{args.arch}"
        )
    elif is_multi_seed:
        run_name = f"rl_train_seeds_{args.arch}"
    else:
        run_name = f"rl_train_seed{args.seed}_{args.arch}"
    run_dir = build_run_dir(args.out, run_name)
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Store the resolved config before any model work so interrupted trials
    # still retain the exact invocation settings beside partial artifacts.
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    data_source = str(cfg.env.get("data_source", "frames"))
    obs_mmap: str | None
    bundle_path: Path | None = None
    test_bundle_path: Path | None = None
    train_env_spec: Any | None = None
    train_eval_spec: Any | None = None
    test_eval_spec: Any | None = None
    train_bar_count = len(train_bars)
    test_bar_count = len(test_bars)
    best_bars: pd.DataFrame | None = None
    best_features: pd.DataFrame | None = None
    dashboard_bars: pd.DataFrame | None = None
    dashboard_features: pd.DataFrame | None = None
    if data_source == "bundle":
        from simulation.environment.bundles import build_env_bundle
        from simulation.environment.spec import EnvSpec

        if args.use_vae:
            raise ValueError(
                "env.data_source=bundle with --use-vae requires the VAE latent-table phase"
            )

        bundle_path = build_env_bundle(
            cache_dir,
            train_bars,
            train_feat,
            cfg,
            raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
            feature_config_hash=feat_cache.stem,
            tickbook=ticks_covering(primary_ticks, train_bars),
            pre_ny_by_date=pre_ny_by_date,
        )
        obs_mmap = str(bundle_path / "obs_features.npy")
        train_env_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(bundle_path),
            episodic=_training_episodic(cfg),
        )
        train_eval_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(bundle_path),
            episodic=False,
            evaluation=True,
        )
        test_bundle_path = build_env_bundle(
            cache_dir,
            test_bars,
            test_feat,
            cfg,
            raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
            feature_config_hash=feat_cache.stem,
            tickbook=ticks_covering(primary_ticks, test_bars),
            pre_ny_by_date=None,
        )
        test_eval_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(test_bundle_path),
            episodic=False,
            evaluation=True,
        )
        best_count = min(20_000, train_bar_count)
        # .copy() detaches these from the full train arrays so that the
        # `del train_bars, train_feat` below actually frees the base (a bare
        # .iloc slice would keep the whole 351k-row frame alive).
        best_bars = train_bars.iloc[:best_count].copy()
        best_features = train_feat.iloc[:best_count].copy()
        dash_cfg = _dashboard_block(cfg)
        dashboard_count = min(int(dash_cfg.get("eval_bars", 20_000)), train_bar_count)
        dashboard_bars = train_bars.iloc[:dashboard_count].copy()
        dashboard_features = train_feat.iloc[:dashboard_count].copy()
        log.info("Built shared environment bundle: %s", bundle_path)
    elif data_source == "frames":
        obs_mmap = _publish_obs_memmap(train_feat, train_bars, cfg, run_dir / "obs_features.npy")
    else:
        raise ValueError(f"env.data_source must be 'frames' or 'bundle', got {data_source!r}")
    runtime_train_bars = train_bars
    runtime_train_feat = train_feat
    runtime_test_bars = test_bars
    runtime_test_feat = test_feat
    if data_source == "bundle" and not args.walk_forward:
        runtime_train_bars = runtime_train_feat = None
        runtime_test_bars = runtime_test_feat = None
        import gc

        gc.collect()
        log.info("Released train/test frames; bundle workers now map the shared arrays")

    return TrainingPreparation(
        args=args,
        cfg=cfg,
        device=device,
        lagrangian_cb=lagrangian_cb,
        primary_ticks=primary_ticks,
        primary_sym=str(primary_sym),
        cache_dir=cache_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        schedule_path=schedule_path,
        feature_columns=feature_columns,
        feat_cache=feat_cache,
        train_bars=runtime_train_bars,
        train_feat=runtime_train_feat,
        test_bars=runtime_test_bars,
        test_feat=runtime_test_feat,
        train_sec=train_sec,
        test_sec=test_sec,
        vae_model=vae_model,
        env_vae=env_vae,
        pre_ny_by_date=pre_ny_by_date,
        seeds=[int(seed) for seed in seeds],
        is_multi_seed=is_multi_seed,
        run_dir=run_dir,
        model_dir=model_dir,
        data_source=data_source,
        obs_mmap=obs_mmap,
        bundle_path=bundle_path,
        test_bundle_path=test_bundle_path,
        train_env_spec=train_env_spec,
        train_eval_spec=train_eval_spec,
        test_eval_spec=test_eval_spec,
        train_bar_count=train_bar_count,
        test_bar_count=test_bar_count,
        best_bars=best_bars,
        best_features=best_features,
        dashboard_bars=dashboard_bars,
        dashboard_features=dashboard_features,
        dataset_records=dataset_records,
    )


def prepare_training_run() -> TrainingPreparation:
    """Resolve config, data/features, and run assets before any policy learning starts."""
    config_stage = _prepare_training_configuration()
    data_stage = _prepare_training_data(config_stage)
    return _prepare_run_assets(config_stage, data_stage)
