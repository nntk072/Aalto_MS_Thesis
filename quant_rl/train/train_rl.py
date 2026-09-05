"""Train PPO/SAC agent on structure-aware trading environment.

Trains a PPO or SAC policy to learn entry/exit timing and risk/reward parameter selection
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
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, split_train_test
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.eval.export import build_run_dir, save_run
from quant_rl.eval.rollout import evaluate_model
from quant_rl.evaluation import calculate_metrics
from quant_rl.features.build import build_features
from quant_rl.models.agent import build_agent
from quant_rl.train.auxiliary_training import AuxiliaryTrainerCallback
from quant_rl.train.callbacks import BestCheckpointEvalCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def make_env(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Any,
    *,
    algo: str,
    reward: str,
    episodic: bool = True,
) -> TradingEnv:
    continuous_actions = algo == "sac"
    use_sweep_reward = reward == "sweep"
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        risk_frac_range=(cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        rr_ratio_range=(cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=int(cfg.env.get("max_episode_steps", 1000)),
        episodic=episodic,
        continuous_actions=continuous_actions,
        use_sweep_reward=use_sweep_reward,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on structure-aware trading.")
    parser.add_argument(
        "--config",
        default=None,
        help="Base config YAML (default: quant_rl/config/default.yaml). "
        "Lets this entrypoint load config/features_*_mtf.yaml variant configs.",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mvp", action="store_true", help="MVP mode: first 30 days only")
    parser.add_argument("--force", action="store_true", help="Force data pipeline rerun")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="RL algorithm")
    parser.add_argument(
        "--arch", choices=["tcn", "gru", "transformer"], default="tcn", help="Encoder architecture"
    )
    parser.add_argument("--reward", choices=["dsr", "sweep"], default="dsr", help="Reward function")
    parser.add_argument(
        "--use-vae", action="store_true", help="Use VAE feature extractor (not yet implemented)"
    )
    parser.add_argument("--wandb", action="store_true", help="Log run metrics to Weights & Biases")
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run purged walk-forward validation"
    )
    parser.add_argument("--wf-splits", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--purge-bars", type=int, default=60, help="Purge bars for walk-forward")
    parser.add_argument(
        "--embargo-bars", type=int, default=20, help="Embargo bars for walk-forward"
    )
    parser.add_argument(
        "--wf-steps",
        type=int,
        default=None,
        help="Timesteps per WF fold (default: reuse main timesteps)",
    )
    args = parser.parse_args()

    if args.use_vae:
        raise NotImplementedError(
            "VAE feature extractor exists in quant_rl/models/vae.py but is not wired "
            "into this training entrypoint; out of scope for this thesis. "
            "See scripts/train_vae.py to train it standalone."
        )

    np.random.seed(args.seed)
    import random

    random.seed(args.seed)
    # SB3 seeds its own generator but does *not* seed torch.manual_seed — the
    # caller's responsibility. Network init/dropout draw from torch's global
    # RNG, so without this line two --seed 42 runs are not reproducible.
    torch.manual_seed(args.seed)

    cfg = load_config(args.overrides, config_path=args.config)

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
    feat_cache = cache_dir / f"{primary_sym}_features_v4_po3causal.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    # Split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info(
        "Split: train=%d bars (<=%s)  test=%d bars (>=%s)",
        len(train_bars),
        train_end,
        len(test_bars),
        test_start,
    )

    # Slice for MVP
    if args.mvp and len(train_bars) > 30 * 390:
        train_bars = train_bars.iloc[: 30 * 390]
        train_feat = train_feat.iloc[: 30 * 390]
        log.info("MVP: sliced training to %d bars", len(train_bars))

    # Create training environment
    log.info("Creating training environment...")
    train_env = make_env(train_bars, train_feat, cfg, algo=args.algo, reward=args.reward)

    # Setup output directory for model
    run_dir = build_run_dir(args.out, f"rl_train_seed{args.seed}")
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, len(train_bars) // 10),
        save_path=str(model_dir),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
    )

    # Train agent
    timesteps = cfg.ppo.total_timesteps if not args.mvp else cfg.training.total_timesteps_mvp
    log.info("Training %s for %d timesteps...", args.algo.upper(), timesteps)

    model = build_agent(train_env, cfg, arch=args.arch, algo=args.algo)

    aux_cb = None
    aux_cfg = getattr(cfg, "auxiliary", None)
    if aux_cfg is not None and float(aux_cfg.get("aux_weight", 0.0)) > 0.0:
        aux_cb = AuxiliaryTrainerCallback(
            prediction_horizon=int(aux_cfg.get("prediction_horizon", 5)),
            aux_weight=float(aux_cfg.get("aux_weight", 0.1)),
            lr=float(aux_cfg.get("lr", 1e-4)),
            grad_steps=int(aux_cfg.get("grad_steps", 4)),
            batch_windows=int(aux_cfg.get("batch_windows", 256)),
        )
        log.info(
            "Auxiliary loss enabled: aux_weight=%.3f horizon=%d",
            aux_cb.aux_weight,
            aux_cb.prediction_horizon,
        )

    callbacks: list[CheckpointCallback | AuxiliaryTrainerCallback | BestCheckpointEvalCallback] = [
        c for c in (checkpoint_callback, aux_cb) if c is not None
    ]

    # Best-checkpoint eval: every N rollouts, evaluate on a fresh copy of the
    # training env (episodic=False so guardrail breaches don't kill the run)
    # and save the best policy to model_dir/best_model. PPO's final save is
    # rarely the best one; this gives us a "best-so-far" snapshot for the
    # final test evaluation.
    best_eval_freq = max(1, cfg.ppo.n_steps)
    best_cb = BestCheckpointEvalCallback(
        eval_env_factory=lambda: make_env(
            train_bars,
            train_feat,
            cfg,
            algo=args.algo,
            reward=args.reward,
            episodic=False,
        ),
        eval_freq=best_eval_freq,
        best_model_path=model_dir / "ppo_best",
    )
    callbacks.append(best_cb)

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model
    model_path = model_dir / "ppo_final"
    model.save(model_path)
    log.info("Model saved: %s", model_path)

    # Evaluate the trained model on the test set
    log.info("Evaluating trained model on test set...")
    test_result = evaluate_model(
        model,
        bars=test_bars,
        features=test_feat,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        risk_frac_range=(cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        rr_ratio_range=(cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=int(cfg.env.get("max_episode_steps", 1000)),
        continuous_actions=(args.algo == "sac"),
        use_sweep_reward=(args.reward == "sweep"),
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
        test_m.sharpe,
        test_m.max_drawdown * 100,
        test_m.n_trades,
        test_m.total_return_pct,
    )

    # Export test artifacts so the RL run includes the same plots as other runners.
    save_run(
        run_dir=run_dir,
        test_result=test_result,
        test_metrics=test_m,
        test_bars=test_bars,
        cfg=cfg,
        save_plots=getattr(cfg.output, "save_plots", True),
        save_html=getattr(cfg.output, "save_html", True),
        save_csv=getattr(cfg.output, "save_csv", True),
        dpi=getattr(cfg.output, "dpi", 150),
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
        "algo": args.algo,
        "arch": args.arch,
        "reward": args.reward,
        "timesteps": timesteps,
        "train_bars": len(train_bars),
        "test_bars": len(test_bars),
        "test_sharpe": float(test_m.sharpe),
        "test_max_dd": float(test_m.max_drawdown),
        "test_trades": test_m.n_trades,
        "test_return": float(test_m.total_return_pct),
        # Gate G3 checks "zero kill-switch breaches"; the count is computed by
        # evaluate_model but was not being written out.
        "test_breaches": test_result.get("n_breach_sessions", 0),
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))

    # Optional wandb logging — the sweep (config/wandb_sweep.yaml) reads
    # these same keys when it launches this entrypoint.
    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit("--wandb requires the wandb package: pip install wandb")
        wandb.init(
            project="aalto-liquidity-sweep",
            name=run_dir.name,
            config={
                "algo": args.algo,
                "arch": args.arch,
                "reward": args.reward,
                "seed": args.seed,
                "mvp": args.mvp,
                "timesteps": timesteps,
            },
        )
        wandb.log(
            {
                "sharpe": training_log["test_sharpe"],
                "test_sharpe": training_log["test_sharpe"],
                "test_max_dd": training_log["test_max_dd"],
                "test_trades": training_log["test_trades"],
                "test_return": training_log["test_return"],
                "test_breaches": training_log["test_breaches"],
            }
        )
        wandb.finish()

    # Walk-forward validation (additive, does not alter the single-split flow)
    if args.walk_forward:
        log.info("Running walk-forward validation with %d splits...", args.wf_splits)
        from quant_rl.evaluation.walkforward import purged_walk_forward

        wf_steps = args.wf_steps or timesteps
        wf_results = []

        for split in purged_walk_forward(
            len(train_bars),
            n_splits=args.wf_splits,
            purge_bars=args.purge_bars,
            embargo_bars=args.embargo_bars,
        ):
            fold_train_bars = train_bars.iloc[split.train_idx]
            fold_train_feat = train_feat.iloc[split.train_idx]
            fold_test_bars = train_bars.iloc[split.test_idx]
            fold_test_feat = train_feat.iloc[split.test_idx]

            log.info(
                "WF fold %d: train=%d test=%d",
                split.fold,
                len(fold_train_bars),
                len(fold_test_bars),
            )

            fold_env = make_env(
                fold_train_bars, fold_train_feat, cfg, algo=args.algo, reward=args.reward
            )
            fold_model = build_agent(fold_env, cfg, arch=args.arch, algo=args.algo)
            fold_model.learn(total_timesteps=wf_steps, callback=None, progress_bar=False)

            fold_result = evaluate_model(
                fold_model,
                bars=fold_test_bars,
                features=fold_test_feat,
                obs_window=cfg.env.obs_window,
                initial_balance=cfg.account.initial_balance,
                risk_frac_range=(
                    cfg.risk.default_risk_frac * 0.5,
                    cfg.risk.default_risk_frac * 2.0,
                ),
                rr_ratio_range=(cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
                swing_buffer_pts=cfg.risk.swing_buffer_pts,
                contract_size=cfg.account.contract_size,
                max_loss_per_trade_usd=cfg.backtest.validation.max_loss_per_trade_usd,
                dsr_eta=cfg.env.reward_dsr_eta,
                max_episode_steps=int(cfg.env.get("max_episode_steps", 1000)),
                continuous_actions=(args.algo == "sac"),
                use_sweep_reward=(args.reward == "sweep"),
            )
            fold_m = calculate_metrics(
                fold_result["equity"],
                trades=fold_result["trades"],
                n_sessions=fold_result.get("n_sessions", 1),
                n_breach_sessions=fold_result.get("n_breach_sessions", 0),
            )
            wf_results.append(
                {
                    "fold": split.fold,
                    "train_bars": len(fold_train_bars),
                    "test_bars": len(fold_test_bars),
                    "sharpe": float(fold_m.sharpe),
                    "max_drawdown": float(fold_m.max_drawdown),
                    "total_return_pct": float(fold_m.total_return_pct),
                    "n_trades": fold_m.n_trades,
                }
            )

        wf_sharpes = [r["sharpe"] for r in wf_results]
        wf_summary = {
            "splits": args.wf_splits,
            "purge_bars": args.purge_bars,
            "embargo_bars": args.embargo_bars,
            "wf_steps": wf_steps,
            "folds": wf_results,
            "mean_sharpe": float(np.mean(wf_sharpes)) if wf_sharpes else None,
            "std_sharpe": float(np.std(wf_sharpes)) if wf_sharpes else None,
        }
        (run_dir / "walk_forward.json").write_text(json.dumps(wf_summary, indent=2))
        log.info(
            "Walk-forward complete. Mean Sharpe=%.3f std=%.3f",
            wf_summary["mean_sharpe"],
            wf_summary["std_sharpe"],
        )

    log.info("Training complete. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
