"""Train PPO/SAC agent on structure-aware trading environment.

Trains a PPO or SAC policy to learn entry/exit timing and risk/reward parameter selection
using swing structure and SMT divergence features.

Usage
-----
    cd Aalto_MS_Thesis
    uv run python -m training.runner.training --seed 42
    uv run python -m training.runner.training --mvp --seed 42  # MVP: first 30 days
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

from training.runner.reporting_runtime import finalize_training_run
from training.runner.seed_runtime import SeedRunContext, run_seed_training
from training.runner.setup_runtime import prepare_training_run
from training.runner.walkforward_runtime import run_walk_forward_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    """Compose training setup, per-seed execution, reporting, and optional walk-forward validation."""
    prep = prepare_training_run()
    seed_context = SeedRunContext(
        args=prep.args,
        cfg=prep.cfg,
        device=prep.device,
        seeds=prep.seeds,
        is_multi_seed=prep.is_multi_seed,
        run_dir=prep.run_dir,
        model_dir=prep.model_dir,
        train_env_spec=prep.train_env_spec,
        train_bars=prep.train_bars,
        train_feat=prep.train_feat,
        env_vae=prep.env_vae,
        pre_ny_by_date=prep.pre_ny_by_date,
        obs_mmap=prep.obs_mmap,
        primary_ticks=prep.primary_ticks,
        lagrangian_cb=prep.lagrangian_cb,
        train_bar_count=prep.train_bar_count,
        best_bars=prep.best_bars,
        best_features=prep.best_features,
        bundle_path=prep.bundle_path,
        cache_dir=prep.cache_dir,
        feat_cache=prep.feat_cache,
        dashboard_bars=prep.dashboard_bars,
        dashboard_features=prep.dashboard_features,
        test_eval_spec=prep.test_eval_spec,
        test_bars=prep.test_bars,
        test_feat=prep.test_feat,
        train_eval_spec=prep.train_eval_spec,
        train_sec=prep.train_sec,
        test_sec=prep.test_sec,
        test_bundle_path=prep.test_bundle_path,
        dataset_records=prep.dataset_records,
        train_start=prep.train_start,
        train_end=prep.train_end,
        test_start=prep.test_start,
        data_source=prep.data_source,
        primary_sym=prep.primary_sym,
        schedule_path=prep.schedule_path,
        vae_model=prep.vae_model,
    )
    seed_results, timesteps = run_seed_training(seed_context)

    finalize_training_run(
        cfg=prep.cfg,
        args=prep.args,
        is_multi_seed=prep.is_multi_seed,
        seed_results=seed_results,
        timesteps=int(timesteps),
        train_bar_count=prep.train_bar_count,
        test_bar_count=prep.test_bar_count,
        run_dir=prep.run_dir,
        feat_cache=prep.feat_cache,
        feature_columns=prep.feature_columns,
        seeds=prep.seeds,
    )
    run_walk_forward_validation(
        args=prep.args,
        timesteps=int(timesteps),
        train_bars=prep.train_bars,
        train_sec=prep.train_sec,
        cfg=prep.cfg,
        cache_dir=prep.cache_dir,
        primary_sym=prep.primary_sym,
        primary_ticks=prep.primary_ticks,
        schedule_path=prep.schedule_path,
        feature_columns=prep.feature_columns,
        run_dir=prep.run_dir,
        env_vae=prep.env_vae,
        pre_ny_by_date=prep.pre_ny_by_date,
        device=prep.device,
        vae_model=prep.vae_model,
        seed=int(seed_results[-1]["seed"]),
    )
    log.info("Training complete. Run directory: %s", prep.run_dir)


if __name__ == "__main__":
    main()
