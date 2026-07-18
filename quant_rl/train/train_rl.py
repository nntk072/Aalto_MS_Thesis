"""PPO training loop with structure-aware SL/TP and risk sizing.

Usage:
    cd Aalto_MS_Thesis
    python -m quant_rl.train.train_rl --mvp
    python -m quant_rl.train.train_rl --seed=42
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline, build_tick_books
from quant_rl.data.split import split_train_test, get_split_config
from quant_rl.features.build import build_features
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.backtest.engine import run_backtest
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.export import save_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def make_env(bars: pd.DataFrame, features: pd.DataFrame, cfg) -> TradingEnv:
    """Factory to create a single training environment."""
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        max_loss_per_trade=cfg.backtest.validation.max_loss_per_trade_usd or 100.0,
        contract_size=cfg.account.contract_size if hasattr(cfg.account, 'contract_size') else 1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policy with structure SL/TP")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mvp", action="store_true", help="MVP mode: short training on minimal data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym   = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1    = data[primary_sym]["M1"]
    secondary_m1  = data.get(secondary_sym, {}).get("M1")

    cache_dir  = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features   = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    
    # MVP mode: use only recent data for faster iteration
    if args.mvp:
        max_days = cfg.training.get("max_days_mvp", 30)
        if max_days:
            cutoff = pd.Timestamp(train_end) - pd.Timedelta(days=max_days)
            mask = train_bars.index >= cutoff
            train_bars = train_bars[mask]
            train_feat = train_feat[mask]
        log.info("MVP mode: training on %d bars", len(train_bars))

    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Create vectorized training environment
    def make_train_env():
        return make_env(train_bars, train_feat, cfg)

    env = DummyVecEnv([make_train_env])

    # PPO training
    timesteps = (cfg.training.get("total_timesteps_mvp", 8192) if args.mvp 
                 else cfg.ppo.total_timesteps)
    log.info("PPO training: %d timesteps (MVP=%s)", timesteps, args.mvp)

    model = PPO(
        "MultiInputPolicy",
        env,
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

    try:
        model.learn(total_timesteps=timesteps)
    except KeyboardInterrupt:
        log.info("Training interrupted by user")

    # Save trained policy
    model_path = Path(args.out) / f"ppo_model_seed{args.seed}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    log.info("Model saved to %s", model_path)

    # Evaluate on test set using trained policy
    def trained_policy(obs: np.ndarray) -> int:
        # Convert obs to env-compatible format if needed
        action, _ = model.predict(obs, deterministic=True)
        return int(action[0]) if isinstance(action, np.ndarray) else int(action)

    max_loss_per_trade = cfg.backtest.validation.max_loss_per_trade_usd
    take_profit_per_trade = cfg.backtest.validation.take_profit_per_trade_usd
    contract_size = cfg.account.contract_size if hasattr(cfg.account, 'contract_size') else 1.0
    default_risk_frac = cfg.risk.default_risk_frac if hasattr(cfg, 'risk') else 0.01

    tick_books = build_tick_books(cfg, force=args.force)
    primary_ticks = tick_books.get(cfg.data.primary)
    test_ticks = primary_ticks.slice(
        pd.Timestamp(test_start + " 00:00:00", tz=cfg.data.tz),
        pd.Timestamp("2099-01-01", tz=cfg.data.tz),
    ) if primary_ticks is not None else None

    log.info("Evaluating trained policy on test set ...")
    test_result = run_backtest(
        bars=test_bars,
        features=test_feat,
        policy=trained_policy,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        max_loss_per_trade_usd=max_loss_per_trade,
        take_profit_per_trade_usd=take_profit_per_trade,
        tickbook=test_ticks,
        use_structure_sl_tp=True,
        contract_size=contract_size,
        default_risk_frac=default_risk_frac,
    )
    test_result["initial_balance"] = cfg.account.initial_balance

    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    log.info("[Test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%",
             test_m.sharpe, test_m.max_drawdown * 100, test_m.total_trades,
             test_m.total_return * 100)


if __name__ == "__main__":
    main()
