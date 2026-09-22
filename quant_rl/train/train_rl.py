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
import copy
import json
import logging
from datetime import datetime
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, make_train_mask, split_bars, split_train_test
from quant_rl.envs.distribution_reward import DistributionReward
from quant_rl.envs.po3_reward import PO3Reward
from quant_rl.envs.strategies import (
    BaselineStrategy,
    DistributionStrategy,
    PO3IFVGStrategy,
    TradingStrategy,
)
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.eval.export import build_run_dir, save_run
from quant_rl.eval.rollout import evaluate_model
from quant_rl.evaluation import calculate_metrics
from quant_rl.features.build import build_features, feature_cache_path
from quant_rl.models.agent import build_agent
from quant_rl.train.auxiliary_training import AuxiliaryTrainerCallback
from quant_rl.train.callbacks import (
    BestCheckpointEvalCallback,
    ClipLogStdCallback,
    PeriodicCheckpointCallback,
    ProgressLoggerCallback,
)
from quant_rl.utils.device import get_device, scale_training_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_STRATEGY_CONFIGS = {
    "po3_ifvg": "config/idea1_po3_ifvg.yaml",
    "distribution": "config/idea2_distribution.yaml",
}


def _strategy_risk_ranges(cfg: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Risk/RR ranges for TradingEnv; strategy_actions uses strategy.risk YAML."""
    if bool(cfg.env.get("strategy_actions", False)):
        risk_cfg = getattr(cfg, "strategy", None)
        risk_block = risk_cfg.get("risk", {}) if risk_cfg is not None else {}
        rfr = list(risk_block.get("risk_frac_range", [0.005, 0.01]))
        rr = list(risk_block.get("rr_range", [1.5, 5.0]))
        return (float(rfr[0]), float(rfr[1])), (float(rr[0]), float(rr[1]))
    return (
        (cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        (cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
    )


def _max_loss_per_trade(cfg: Any) -> float:
    """Per-trade USD cap: FTMO risk_per_trade when strategy_actions, else backtest."""
    if bool(cfg.env.get("strategy_actions", False)):
        return float(cfg.ftmo.risk_per_trade_limit)
    return float(cfg.backtest.validation.max_loss_per_trade_usd)


def _guardrail_kwargs(cfg: Any) -> dict[str, float]:
    """PPO training kill-switches, including the $7k-from-peak training cap."""
    return {
        "daily_loss_limit": float(cfg.ftmo.daily_loss_limit),
        "max_loss_limit": float(cfg.ftmo.max_loss_limit),
        "risk_per_trade_limit": float(cfg.ftmo.risk_per_trade_limit),
        "soft_daily_loss_limit": float(cfg.ftmo.get("soft_daily_loss_limit", 2000.0)),
        "soft_max_loss_limit": float(cfg.ftmo.get("soft_max_loss_limit", 5000.0)),
        "trailing_dd_limit": float(cfg.ftmo.get("trailing_dd_limit", 0.07)),
        "soft_trailing_dd_limit": float(cfg.ftmo.get("soft_trailing_dd_limit", 0.0)),
    }


def _eval_guardrail_kwargs(cfg: Any) -> dict[str, float]:
    """FTMO rules for train/test rollouts: daily $5k + max $10k from initial."""
    kwargs = _guardrail_kwargs(cfg)
    kwargs["trailing_dd_limit"] = 0.0
    kwargs["soft_trailing_dd_limit"] = 0.0
    return kwargs


def _periodic_checkpoint_callback(
    cfg: DictConfig, model_dir: Path
) -> PeriodicCheckpointCallback | None:
    """Save numbered + latest zips every ``ppo.checkpoint_freq`` env steps."""
    freq = int(cfg.ppo.get("checkpoint_freq", 1_000_000))
    if freq <= 0:
        log.info("Periodic checkpoints disabled (checkpoint_freq=%s)", freq)
        return None
    log.info("Periodic checkpoints every %d env steps → %s", freq, model_dir)
    return PeriodicCheckpointCallback(save_freq=freq, save_path=model_dir)


def _max_episode_steps(cfg: Any) -> int | None:
    """Parse ``env.max_episode_steps``; YAML ``null`` → full-year episode."""
    raw = cfg.env.get("max_episode_steps", None)
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "null", "none"):
        return None
    return int(raw)


def _direction_summary(trades: Any) -> dict[str, int]:
    """Count open directions from an evaluate_model trade log."""
    if trades is None or getattr(trades, "empty", True):
        return {"long": 0, "short": 0, "opens": 0}
    opens = trades[trades["type"] == "open"] if "type" in trades.columns else trades
    if opens.empty or "direction" not in opens.columns:
        return {"long": 0, "short": 0, "opens": 0}
    dirs = opens["direction"].astype(int)
    return {
        "long": int((dirs == 1).sum()),
        "short": int((dirs == -1).sum()),
        "opens": int(len(opens)),
    }


def _strategy_from_cfg(cfg: Any) -> tuple[Any, Any, float]:
    """Build (strategy, strategy_reward, strategy_weight) from merged config.

    Baseline (Idea 3, P2) keeps the no-op strategy and no alignment reward
    unless ``env.strategy_actions`` is explicitly enabled by a variant config.
    """
    strat_cfg = getattr(cfg, "strategy", None)
    if strat_cfg is None or not bool(cfg.env.get("strategy_actions", False)):
        return BaselineStrategy(), None, 0.0

    name = str(strat_cfg.get("name", "baseline"))
    enforce_gate = bool(strat_cfg.get("entry", {}).get("enforce_gate", False))
    reward_cfg = strat_cfg.get("reward", {})
    weight = float(reward_cfg.get("strategy_weight", 0.0)) if reward_cfg else 0.0

    reward: PO3Reward | DistributionReward | None = None
    strategy: TradingStrategy
    if name == "po3_ifvg":
        strategy = PO3IFVGStrategy(enforce_gate=enforce_gate)
        reward = PO3Reward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            manipulation_penalty=float(reward_cfg.get("manipulation_penalty", 0.02)),
            invalid_ifvg_penalty=float(reward_cfg.get("invalid_ifvg_penalty", 0.01)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
        )
    elif name == "distribution":
        strategy = DistributionStrategy(enforce_gate=enforce_gate)
        reward_cfg = strat_cfg.get("reward", {}) or {}
        reward = DistributionReward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
        )
    else:
        strategy = BaselineStrategy()
        reward = None
        weight = 0.0
    return strategy, reward, weight


def _publish_obs_memmap(features: pd.DataFrame, cfg: Any, path: Path) -> str | None:
    """Write the model observation matrix once so workers mmap it read-only.

    ``n_envs <= 1`` keeps the private ``to_numpy`` copy. The pandas frame is
    still passed in; this only removes the extra float32 matrix per worker.
    """
    if int(cfg.env.get("n_envs", 1)) <= 1:
        return None
    from quant_rl.features.build import select_obs_columns

    strategy, _, _ = _strategy_from_cfg(cfg)
    frame = select_obs_columns(features, strategy.raw_columns)
    array = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path)


def make_env(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Any,
    *,
    algo: str,
    reward: str,
    episodic: bool = True,
    use_vae: bool = False,
    vae: Any | None = None,
    pre_ny_by_date: dict[Any, Any] | None = None,
    obs_features_mmap: str | None = None,
) -> TradingEnv:
    continuous_actions = algo == "sac"
    use_sweep_reward = reward == "sweep"
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        guardrail_kwargs=_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=_max_episode_steps(cfg),
        episodic=episodic,
        continuous_actions=continuous_actions,
        use_sweep_reward=use_sweep_reward,
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        use_vae=use_vae,
        vae=vae,
        pre_ny_by_date=pre_ny_by_date,
        obs_features_mmap=obs_features_mmap,
    )


def parse_train_args() -> argparse.Namespace:
    """Parse CLI arguments for the RL training entrypoint."""
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
        "--strategy",
        choices=["baseline", "po3_ifvg", "distribution"],
        default="baseline",
        help="RL strategy variant (Agent.md: Idea 1 = po3_ifvg, Idea 2 = distribution, Idea 3 = baseline).",
    )
    parser.add_argument(
        "--use-vae",
        action="store_true",
        help="Condition the policy on a frozen VAE narrative latent (requires --vae-path)",
    )
    parser.add_argument(
        "--vae-path",
        default=None,
        help="Path to a trained VAE state_dict (.pth from scripts/train_vae.py)",
    )
    parser.add_argument(
        "--vae-config",
        default="config/vae.yaml",
        help="VAE architecture YAML (must match the checkpoint)",
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
    return parser.parse_args()


def _setup_rngs(seed: int) -> None:
    """Seed numpy, random, and torch for reproducible training."""
    np.random.seed(seed)
    import random

    random.seed(seed)
    # SB3 seeds its own generator but does *not* seed torch.manual_seed — the
    # caller's responsibility. Network init/dropout draw from torch's global
    # RNG, so without this line two --seed 42 runs are not reproducible.
    torch.manual_seed(seed)


def _load_merged_config(args: argparse.Namespace) -> DictConfig:
    """Load base config and merge strategy variant if requested."""
    cfg = load_config(args.overrides, config_path=args.config)
    # Merge the strategy variant config (Idea 1/2) on top of the base config.
    # Baseline (Idea 3) leaves cfg untouched.
    if args.strategy in _STRATEGY_CONFIGS:
        variant_path = _STRATEGY_CONFIGS[args.strategy]
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(variant_path)))
        log.info("Merged strategy config: %s", variant_path)
    elif args.strategy != "baseline":
        raise ValueError(f"unknown strategy: {args.strategy}")
    return cfg


def _build_training_log(
    *,
    seed: int,
    mvp: bool,
    algo: str,
    arch: str,
    reward: str,
    timesteps: int,
    train_bars: int,
    test_bars: int,
    test_m: Any,
    test_result: dict[str, Any],
    strategy: str = "baseline",
    strategy_actions: bool = False,
) -> dict[str, Any]:
    """Build the training_log.json dict from run results."""
    return {
        "seed": seed,
        "mvp": mvp,
        "algo": algo,
        "arch": arch,
        "reward": reward,
        "strategy": strategy,
        "strategy_actions": strategy_actions,
        "timesteps": timesteps,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "test_sharpe": float(test_m.sharpe),
        "test_max_dd": float(test_m.max_drawdown),
        "test_trades": test_m.n_trades,
        "test_return": float(test_m.total_return_pct),
        # Gate G3 checks "zero kill-switch breaches"; the count is computed by
        # evaluate_model but was not being written out.
        "test_breaches": test_result.get("n_breach_sessions", 0),
        "test_n_sessions": int(test_result.get("n_sessions", 0)),
        "test_days_traded": int(test_result.get("days_traded", 0)),
        "test_survived_full_year": bool(test_result.get("survived_full_year", False)),
        "test_max_trailing_dd": float(test_result.get("max_trailing_dd", test_m.max_drawdown)),
        "test_fail_time": (
            str(test_result["fail_time"]) if test_result.get("fail_time") is not None else None
        ),
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    args = parse_train_args()

    if args.use_vae and not args.vae_path:
        raise SystemExit("--use-vae requires --vae-path pointing at a trained VAE .pth")

    _setup_rngs(args.seed)

    cfg = _load_merged_config(args)

    device = get_device(cfg.get("device"))
    scale_training_cfg(cfg, device, args.arch)
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

    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    # Split dates first so z-score fit can be train-scoped (TI-4) and the
    # content-hash cache key includes that mask (TI-3).
    train_end, test_start = get_split_config(cfg)
    train_mask = make_train_mask(
        cast(pd.DatetimeIndex, primary_m1.index),
        train_end,
    )
    feat_cache = feature_cache_path(cache_dir, primary_sym, cfg, primary_m1, train_mask=train_mask)
    features = build_features(
        primary_m1,
        secondary=secondary_m1,
        cfg=cfg,
        train_mask=train_mask,
        cache_path=feat_cache,
    )

    # Split
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start)
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

    vae_model = None
    env_vae = None
    pre_ny_by_date = None
    if args.use_vae:
        from quant_rl.models.vae_pre_ny import build_pre_ny_by_date, load_vae_from_checkpoint

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

    # Setup output directory for model
    run_dir = build_run_dir(args.out, f"rl_train_seed{args.seed}_{args.arch}")
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    obs_mmap = _publish_obs_memmap(train_feat, cfg, run_dir / "obs_features.npy")

    # Create training environment
    log.info("Creating training environment...")
    train_env = make_env(
        train_bars,
        train_feat,
        cfg,
        algo=args.algo,
        reward=args.reward,
        use_vae=args.use_vae,
        vae=env_vae,
        pre_ny_by_date=pre_ny_by_date,
        obs_features_mmap=obs_mmap,
    )

    checkpoint_callback = _periodic_checkpoint_callback(cfg, model_dir)

    # Train agent
    timesteps = cfg.ppo.total_timesteps if not args.mvp else cfg.training.total_timesteps_mvp
    log.info("Training %s for %d timesteps...", args.algo.upper(), timesteps)

    model = build_agent(
        train_env,
        cfg,
        arch=args.arch,
        algo=args.algo,
        device=device,
        use_vae=args.use_vae,
        vae=vae_model,
        env_fn=partial(
            make_env,
            train_bars,
            train_feat,
            cfg,
            algo=args.algo,
            reward=args.reward,
            use_vae=args.use_vae,
            vae=env_vae,
            pre_ny_by_date=pre_ny_by_date,
            obs_features_mmap=obs_mmap,
        ),
    )

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

    callbacks: list[
        PeriodicCheckpointCallback
        | AuxiliaryTrainerCallback
        | BestCheckpointEvalCallback
        | ProgressLoggerCallback
        | ClipLogStdCallback
    ] = [c for c in (checkpoint_callback, aux_cb) if c is not None]

    # Best-checkpoint eval: every N rollouts, evaluate on a fresh copy of the
    # training env (episodic=False so guardrail breaches don't kill the run)
    # and save the best policy to model_dir/best_model. PPO's final save is
    # rarely the best one; this gives us a "best-so-far" snapshot for the
    # final test evaluation.
    best_eval_freq = max(1, timesteps)  # eval once at end — avoids CPU-only stalls during training
    best_cb = BestCheckpointEvalCallback(
        eval_env_factory=lambda: make_env(
            train_bars,
            train_feat,
            cfg,
            algo=args.algo,
            reward=args.reward,
            episodic=False,
            use_vae=args.use_vae,
            vae=env_vae,
            pre_ny_by_date=pre_ny_by_date,
        ),
        eval_freq=best_eval_freq,
        best_model_path=model_dir / "ppo_best",
    )
    callbacks.append(best_cb)

    progress_log = model_dir / "training_log.csv"
    callbacks.append(ProgressLoggerCallback(log_path=progress_log))
    if isinstance(train_env.action_space, spaces.Box):
        callbacks.append(
            ClipLogStdCallback(
                log_std_min=float(cfg.ppo.get("log_std_min", -0.7)),
                log_std_max=float(cfg.ppo.get("log_std_max", 0.0)),
            )
        )

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save final model
    model_path = model_dir / "ppo_final"
    model.save(model_path)
    log.info("Model saved: %s", model_path)

    # Learning-curve / loss charts from the SB3 progress CSV.
    try:
        from quant_rl.eval.training_plots import save_training_plots

        save_training_plots(
            progress_log,
            out_dir=model_dir,
            dpi=getattr(cfg.output, "dpi", 150),
            save_html=getattr(cfg.output, "save_html", True),
        )
    except Exception as exc:
        log.warning("Training progress plots skipped: %s", exc)

    # Evaluate the trained model on the held-out test set (out-of-sample).
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    strategy_actions = bool(cfg.env.get("strategy_actions", False))
    sl_buffer_pts = float(cfg.env.get("sl_buffer_pts", 0.0))
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    eval_common = dict(
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        guardrail_kwargs=_eval_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        continuous_actions=(args.algo == "sac"),
        use_sweep_reward=(args.reward == "sweep"),
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        use_vae=args.use_vae,
        vae=env_vae,
        strategy=strategy,
        strategy_actions=strategy_actions,
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        sl_buffer_pts=sl_buffer_pts,
        pre_ny_by_date=pre_ny_by_date,
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
    )
    log.info("Evaluating trained model on test set...")
    test_result = evaluate_model(
        model,
        bars=test_bars,
        features=test_feat,
        max_episode_steps=None,
        **eval_common,
    )
    test_result["initial_balance"] = cfg.account.initial_balance
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
    )
    test_dir_sum = _direction_summary(test_result["trades"])
    log.info(
        "[test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  "
        "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
        test_m.sharpe,
        test_m.max_drawdown * 100,
        test_m.n_trades,
        test_m.total_return_pct,
        test_result.get("survived_full_year"),
        test_result.get("days_traded", 0),
        test_result.get("n_sessions", 0),
        test_result.get("fail_time"),
        test_m.max_drawdown * 100,
        test_dir_sum,
        test_result.get("entry_diag", {}),
    )

    # In-sample evaluation on the training split so the run dir carries the same
    # training/ + testing/ artifact layout as the baseline runners.
    log.info("Evaluating trained model on training set (in-sample)...")
    train_result = evaluate_model(
        model,
        bars=train_bars,
        features=train_feat,
        max_episode_steps=None,
        **eval_common,
    )
    train_result["initial_balance"] = cfg.account.initial_balance
    train_m = calculate_metrics(
        train_result["equity"],
        trades=train_result["trades"],
        n_sessions=train_result.get("n_sessions", 1),
        n_breach_sessions=train_result.get("n_breach_sessions", 0),
    )
    train_dir_sum = _direction_summary(train_result["trades"])
    log.info(
        "[train] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  "
        "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
        train_m.sharpe,
        train_m.max_drawdown * 100,
        train_m.n_trades,
        train_m.total_return_pct,
        train_result.get("survived_full_year"),
        train_result.get("days_traded", 0),
        train_result.get("n_sessions", 0),
        train_result.get("fail_time"),
        train_m.max_drawdown * 100,
        train_dir_sum,
        train_result.get("entry_diag", {}),
    )
    (run_dir / "direction_summary.json").write_text(
        json.dumps({"train": train_dir_sum, "test": test_dir_sum}, indent=2)
    )

    # Export both splits so the RL run produces the same artifact layout as the
    # other runners: training/ + testing/ trees with plots, CSVs, and metrics.
    save_run(
        run_dir=run_dir,
        train_result=train_result,
        train_metrics=train_m,
        train_bars=train_bars,
        train_secondary=train_sec,
        test_result=test_result,
        test_metrics=test_m,
        test_bars=test_bars,
        test_secondary=test_sec,
        cfg=cfg,
        save_plots=getattr(cfg.output, "save_plots", True),
        save_html=getattr(cfg.output, "save_html", True),
        save_csv=getattr(cfg.output, "save_csv", True),
        dpi=getattr(cfg.output, "dpi", 150),
    )

    # Thesis data/EDA pack at the run root (coverage, returns, features, PO3).
    try:
        from quant_rl.eval.data_plots import write_data_eda

        write_data_eda(
            run_dir / "data",
            primary_m1,
            features,
            train_end=train_end,
            test_start=test_start,
            dpi=getattr(cfg.output, "dpi", 150),
        )
    except Exception as exc:
        log.warning("Run data EDA skipped: %s", exc)

    # Save config
    if cfg is not None:
        try:
            (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

    # Save training log
    training_log = _build_training_log(
        seed=args.seed,
        mvp=args.mvp,
        algo=args.algo,
        arch=args.arch,
        reward=args.reward,
        timesteps=timesteps,
        train_bars=len(train_bars),
        test_bars=len(test_bars),
        test_m=test_m,
        test_result=test_result,
        strategy="baseline"
        if not strategy_actions
        else str(cfg.strategy.get("name", args.strategy)),
        strategy_actions=strategy_actions,
    )
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

            fold_mmap = _publish_obs_memmap(
                fold_train_feat, cfg, run_dir / f"obs_features_fold{split.fold}.npy"
            )
            fold_env = make_env(
                fold_train_bars,
                fold_train_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                obs_features_mmap=fold_mmap,
            )
            fold_model = build_agent(
                fold_env,
                cfg,
                arch=args.arch,
                algo=args.algo,
                device=device,
                use_vae=args.use_vae,
                vae=vae_model,
                env_fn=partial(
                    make_env,
                    fold_train_bars,
                    fold_train_feat,
                    cfg,
                    algo=args.algo,
                    reward=args.reward,
                    use_vae=args.use_vae,
                    vae=env_vae,
                    pre_ny_by_date=pre_ny_by_date,
                    obs_features_mmap=fold_mmap,
                ),
            )
            fold_model.learn(total_timesteps=wf_steps, callback=None, progress_bar=False)

            fold_result = evaluate_model(
                fold_model,
                bars=fold_test_bars,
                features=fold_test_feat,
                obs_window=cfg.env.obs_window,
                initial_balance=cfg.account.initial_balance,
                guardrail_kwargs=_eval_guardrail_kwargs(cfg),
                risk_frac_range=_strategy_risk_ranges(cfg)[0],
                rr_ratio_range=_strategy_risk_ranges(cfg)[1],
                swing_buffer_pts=cfg.risk.swing_buffer_pts,
                contract_size=cfg.account.contract_size,
                max_loss_per_trade_usd=_max_loss_per_trade(cfg),
                dsr_eta=cfg.env.reward_dsr_eta,
                max_episode_steps=_max_episode_steps(cfg),
                continuous_actions=(args.algo == "sac"),
                use_sweep_reward=(args.reward == "sweep"),
                block_overnight=bool(cfg.env.get("block_overnight", True)),
                eod_risk=dict(cfg.env.get("eod_risk", {})),
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
                min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
                max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
                entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
                reward_mode=str(cfg.env.get("reward_mode", "dsr")),
                entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
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
