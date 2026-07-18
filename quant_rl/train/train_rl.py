"""RL training entrypoint (wires SB3 PPO to TradingEnv).

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.train_rl                    # TCN encoder
    python -m quant_rl.train.train_rl arch=transformer   # Transformer encoder
    python -m quant_rl.train.train_rl --stub             # trivial MLP (env sanity check)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.features.build import build_features
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.backtest.costs import COST_US100

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_stub_agent(env, cfg):
    """Trivial MlpPolicy PPO – no custom encoder, used only to verify env wiring."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    return PPO(
        "MultiInputPolicy",
        vec_env,
        n_steps=min(cfg.ppo.n_steps, 512),
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        learning_rate=cfg.ppo.learning_rate,
        gamma=cfg.ppo.gamma,
        verbose=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--stub", action="store_true", help="Use stub MLP (no encoder needed)")
    parser.add_argument("--arch", default="tcn", choices=["tcn", "transformer"])
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
        log.info("Building stub MLP agent (env wiring test) …")
        model = _build_stub_agent(env, cfg)
        timesteps = 2048
    else:
        log.info("Building PPO + %s encoder …", args.arch.upper())
        from quant_rl.models.agent import build_agent
        model = build_agent(env, cfg, arch=args.arch)
        timesteps = cfg.ppo.total_timesteps

    log.info("Starting training for %d timesteps …", timesteps)
    model.learn(total_timesteps=timesteps)
    log.info("Training complete.")

    save_path = Path("models") / f"ppo_{args.arch if not args.stub else 'stub'}"
    save_path.parent.mkdir(exist_ok=True)
    model.save(str(save_path))
    log.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
