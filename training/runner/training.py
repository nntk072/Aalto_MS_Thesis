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
    config = prep.configuration
    data = prep.data
    assets = prep.run_assets
    seed_context = SeedRunContext(preparation=prep)
    seed_results, timesteps = run_seed_training(seed_context)

    finalize_training_run(
        cfg=config.cfg,
        args=config.args,
        is_multi_seed=assets.is_multi_seed,
        seed_results=seed_results,
        timesteps=int(timesteps),
        train_bar_count=assets.train_bar_count,
        test_bar_count=assets.test_bar_count,
        run_dir=assets.run_dir,
        feat_cache=data.feat_cache,
        feature_columns=data.feature_columns,
        seeds=assets.seeds,
    )
    run_walk_forward_validation(
        args=config.args,
        timesteps=int(timesteps),
        train_bars=data.train_bars,
        train_sec=data.train_sec,
        cfg=config.cfg,
        cache_dir=data.cache_dir,
        primary_sym=data.primary_sym,
        primary_ticks=data.primary_ticks,
        schedule_path=data.schedule_path,
        feature_columns=data.feature_columns,
        run_dir=assets.run_dir,
        env_vae=data.env_vae,
        pre_ny_by_date=data.pre_ny_by_date,
        device=config.device,
        vae_model=data.vae_model,
        seed=int(seed_results[-1]["seed"]),
    )
    log.info("Training complete. Run directory: %s", assets.run_dir)


if __name__ == "__main__":
    main()
