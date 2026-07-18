"""RL training entrypoint (wires SB3 PPO to TradingEnv).

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.train_rl          # requires encoder implementation
    python -m quant_rl.train.train_rl --stub   # uses a trivial MLP policy to verify env wiring
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

import numpy as np

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.features.build import build_features
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.backtest.costs import COST_US100

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_stub_agent(env, cfg):
    """Trivial MlpPolicy PPO – no custom encoder, used only to verify env wiring."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        raise ImportError("stable-baselines3 is required")

    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        verbose=1,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--stub", action="store_true", help="Use stub MLP policy (no encoder needed)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    env = TradingEnv(
        bars=primary_m1,
        features=features,
        obs_window=cfg.env.obs_window,
        cost_model=COST_US100,
        initial_balance=cfg.account.initial_balance,
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    if args.stub:
        log.info("Using stub MLP policy (env wiring test) …")
        model = _build_stub_agent(env, cfg)
    else:
        from quant_rl.models.agent import build_agent
        model = build_agent(env, cfg)

    log.info("Starting training for %d timesteps …", cfg.ppo.total_timesteps)
    model.learn(total_timesteps=cfg.ppo.total_timesteps)
    log.info("Training complete.")

    save_path = Path("models") / "ppo_trading"
    save_path.parent.mkdir(exist_ok=True)
    model.save(str(save_path))
    log.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
