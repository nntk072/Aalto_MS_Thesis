"""Evaluate a saved PPO model on backtest data.

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.run_eval --model outputs/<run>/model/ppo_tcn_mvp
    python -m quant_rl.train.run_eval --model models/ppo_tcn --no-save
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
from quant_rl.data.split import split_train_test, get_split_config
from quant_rl.features.build import build_features
from quant_rl.backtest.engine import run_backtest
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.export import save_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--model", required=True, help="Path to saved SB3 model (without .zip)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    cfg  = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym   = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1    = data[primary_sym]["M1"]
    secondary_m1  = data.get(secondary_sym, {}).get("M1")

    cache_dir  = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features   = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    # Date-based split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Load model
    try:
        from stable_baselines3 import PPO
    except ImportError:
        log.error("stable-baselines3 is required: pip install stable-baselines3")
        sys.exit(1)

    model_path = Path(args.model)
    log.info("Loading model from %s …", model_path)
    model = PPO.load(str(model_path))

    obs_window = cfg.env.obs_window

    def _make_policy(feat_array: np.ndarray):
        from quant_rl.backtest.account import AccountState
        _acc = AccountState(initial_balance=cfg.account.initial_balance)

        def policy(obs_seq: np.ndarray) -> int:
            obs_dict = {
                "seq":     obs_seq[np.newaxis].astype(np.float32),
                "account": _acc.to_array()[np.newaxis].astype(np.float32),
            }
            action, _ = model.predict(obs_dict, deterministic=True)
            return {0: 0, 1: 1, 2: -1}.get(int(action), 0)

        return policy

    def _run(bars, feats, label: str) -> tuple[dict, object]:
        log.info("Evaluating model [%s] …", label)
        result = run_backtest(
            bars=bars, features=feats,
            policy=_make_policy(feats.values.astype(np.float32)),
            obs_window=obs_window,
            initial_balance=cfg.account.initial_balance,
        )
        result["initial_balance"] = cfg.account.initial_balance
        m = calculate_metrics(
            result["equity"], trades=result["trades"],
            n_sessions=result.get("n_sessions", 1),
            n_breach_sessions=result.get("n_breach_sessions", 0),
        )
        log.info(
            "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%",
            label, m.sharpe, m.max_drawdown * 100, m.total_trades, m.total_return * 100,
        )
        return result, m

    train_result, train_m = _run(train_bars, train_feat, "training")
    test_result,  test_m  = _run(test_bars,  test_feat,  "testing")

    if not args.no_save:
        model_name = model_path.stem
        run_dir = save_run(
            out_dir=args.out, name=f"eval_{model_name}",
            train_result=train_result, train_metrics=train_m, train_bars=train_bars,
            test_result=test_result,   test_metrics=test_m,   test_bars=test_bars,
            cfg=cfg,
        )
        log.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--model", required=True, help="Path to saved SB3 model (without .zip)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--out", default="outputs")
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

    # Load model
    try:
        from stable_baselines3 import PPO
    except ImportError:
        log.error("stable-baselines3 is required: pip install stable-baselines3")
        sys.exit(1)

    model_path = Path(args.model)
    log.info("Loading model from %s …", model_path)
    model = PPO.load(str(model_path))

    # Build obs dict for the policy — we need a TradingEnv to get the obs space
    from quant_rl.envs.trading_env import TradingEnv

    env = TradingEnv(
        bars=primary_m1,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        max_episode_steps=cfg.env.max_episode_steps,
    )

    # Build a stateful policy using the SB3 model
    obs_window = cfg.env.obs_window
    feat_array = features.values.astype(np.float32)
    feat_array = np.nan_to_num(feat_array, nan=0.0)

    # We need the account state for obs construction — use a simple wrapper
    from quant_rl.backtest.account import AccountState

    _tmp_acc = AccountState(initial_balance=cfg.account.initial_balance)
    _step = [0]

    def policy(obs_seq: np.ndarray) -> int:
        account_obs = _tmp_acc.to_array()
        obs_dict = {
            "seq": obs_seq[np.newaxis],                            # [1, T, F]
            "account": account_obs[np.newaxis].astype(np.float32), # [1, 5]
        }
        action, _ = model.predict(obs_dict, deterministic=True)
        # map Discrete(3) {0,1,2} → {0, 1, -1}
        action_map = {0: 0, 1: 1, 2: -1}
        return action_map.get(int(action), 0)

    log.info("Running eval backtest …")
    result = run_backtest(
        bars=primary_m1,
        features=features,
        policy=policy,
        obs_window=obs_window,
        initial_balance=cfg.account.initial_balance,
    )
    result["initial_balance"] = cfg.account.initial_balance

    equity = result["equity"]
    trades = result["trades"]
    m = calculate_metrics(
        equity, trades=trades,
        n_sessions=result.get("n_sessions", 1),
        n_breach_sessions=result.get("n_breach_sessions", 0),
    )

    log.info(
        "Sharpe=%.3f  Sortino=%.3f  MaxDD=%.2f%%  Trades=%d  Breaches=%d/%d  Return=%.2f%%",
        m.sharpe, m.sortino, m.max_drawdown * 100, m.total_trades,
        result.get("n_breach_sessions", 0), result.get("n_sessions", 1),
        m.total_return * 100,
    )

    if not args.no_save:
        model_name = Path(args.model).stem
        run_dir = save_run(
            result, m, out_dir=args.out, name=f"eval_{model_name}",
            bars=primary_m1, cfg=cfg,
        )
        log.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
