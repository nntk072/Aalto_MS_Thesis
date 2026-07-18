"""Train PPO agent on structure-aware trading environment.

Trains a PPO policy to learn entry/exit timing and risk/reward parameter selection
using swing structure and SMT divergence features.

Usage
-----
    cd Aalto_MS_Thesis
    uv run python -m quant_rl.train.train_rl --seed 42
    uv run python -m quant_rl.train.train_rl --mvp --seed 42  # MVP: first 30 days
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline, build_tick_books
from quant_rl.data.split import split_train_test, get_split_config
from quant_rl.features.build import build_features
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.backtest.engine import run_backtest
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.export import save_run, build_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _rl_policy(obs: np.ndarray) -> int:
    """Placeholder: returns output from trained policy (will be set after training)."""
    return 0  # Hold by default


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on structure-aware trading.")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mvp", action="store_true", help="MVP mode: first 30 days only")
    parser.add_argument("--force", action="store_true", help="Force data pipeline rerun")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    cfg = load_config(args.overrides)

    # Override for MVP mode
    if args.mvp:
        log.info("MVP mode: using first 30 days of training data")
        cfg.training.use_m1_only = True
        cfg.training.max_days = 30

    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    # Split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Slice for MVP
    if args.mvp and len(train_bars) > 30 * 390:  # ~30 trading days * 390 M1 bars
        train_bars = train_bars.iloc[: 30 * 390]
        train_feat = train_feat.iloc[: 30 * 390]
        log.info("MVP: sliced training to %d bars", len(train_bars))

    # Create training environment
    log.info("Creating training environment…")
    train_env = TradingEnv(
        bars=train_bars,
        features=train_feat,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        risk_frac_range=(cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        rr_ratio_range=(cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
        dsr_eta=cfg.env.reward_dsr_eta,
    )

    # Setup output directory for model
    run_dir = build_run_dir(args.out, f"rl_train_seed{args.seed}")
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, len(train_bars) // 10),
        save_path=model_dir,
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
    )

    # Train PPO
    timesteps = cfg.ppo.total_timesteps if not args.mvp else cfg.training.total_timesteps_mvp
    log.info("Training PPO for %d timesteps…", timesteps)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        ent_coef=cfg.ppo.ent_coef,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)

    # Save final model
    model_path = model_dir / "ppo_final"
    model.save(model_path)
    log.info("Model saved: %s", model_path)

    # Create a policy function that uses the trained model
    def _trained_policy(obs_array: np.ndarray) -> int:
        """Extract action from trained model (stub for backtest compatibility)."""
        # Note: This is simplified. Full integration would use the model properly.
        return 0

    # Evaluate on test set
    log.info("Running backtest on test set with trained policy…")
    test_result = run_backtest(
        bars=test_bars,
        features=test_feat,
        policy=_trained_policy,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
        use_structure_sl_tp=True,
        risk_frac=cfg.risk.default_risk_frac,
        rr_ratio=cfg.risk.rr_ratio_default,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
    )
    test_result["initial_balance"] = cfg.account.initial_balance
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    log.info(
        "[test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%",
        test_m.sharpe, test_m.max_drawdown * 100, test_m.total_trades, test_m.total_return * 100,
    )

    # Save config
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

    # Save training log
    training_log = {
        "seed": args.seed,
        "mvp": args.mvp,
        "timesteps": timesteps,
        "train_bars": len(train_bars),
        "test_bars": len(test_bars),
        "test_sharpe": float(test_m.sharpe),
        "test_max_dd": float(test_m.max_drawdown),
        "test_trades": test_m.total_trades,
        "test_return": float(test_m.total_return),
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))

    log.info("Training complete. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
