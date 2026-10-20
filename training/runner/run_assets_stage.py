"""Focused run assets preparation stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from evaluation.reporting.export import build_run_dir
from market_data.loading.ticks import ticks_covering
from training.setup.callbacks import (
    _dashboard_block,
)
from training.setup.data import (
    _publish_obs_memmap,
)
from training.setup.environments import (
    _training_episodic,
    strategy_from_cfg,
)

from .stages import (
    TrainingConfigurationStage,
    TrainingDataStage,
    TrainingPreparation,
    TrainingRunAssetsStage,
)

log = logging.getLogger(__name__)


def _prepare_run_assets(
    config_stage: TrainingConfigurationStage, data_stage: TrainingDataStage
) -> TrainingPreparation:
    args = config_stage.args
    cfg = config_stage.cfg
    primary_ticks = data_stage.primary_ticks
    cache_dir = data_stage.cache_dir
    feat_cache = data_stage.feat_cache
    train_bars = data_stage.train_bars
    train_feat = data_stage.train_feat
    test_bars = data_stage.test_bars
    test_feat = data_stage.test_feat
    pre_ny_by_date = data_stage.pre_ny_by_date
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
            raw_columns=list(strategy_from_cfg(cfg)[0].raw_columns),
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
            raw_columns=list(strategy_from_cfg(cfg)[0].raw_columns),
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

    data_stage.train_bars = runtime_train_bars
    data_stage.train_feat = runtime_train_feat
    data_stage.test_bars = runtime_test_bars
    data_stage.test_feat = runtime_test_feat
    run_assets = TrainingRunAssetsStage(
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
    )
    return TrainingPreparation(
        configuration=config_stage,
        data=data_stage,
        run_assets=run_assets,
    )
