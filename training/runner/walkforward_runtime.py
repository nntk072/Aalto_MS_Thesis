from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from learning.rl.agent import build_agent

from evaluation import calculate_metrics, evaluation_periods_per_year
from features import build_walkforward_fold_features
from features.context.policy_market import attach_policy_market_context
from market_data.loading.ticks import ticks_covering
from market_data.splits.walkforward import purged_walk_forward
from training.setup.data import _publish_obs_memmap
from training.setup.environments import (
    _evaluate_final_split,
    build_eval_provider,
    eval_guardrail_kwargs,
    fill_delay_ms,
    make_env,
    max_episode_steps,
    max_loss_per_trade,
    strategy_from_cfg,
    strategy_risk_ranges,
)

log = logging.getLogger(__name__)


def run_walk_forward_validation(
    *,
    args: Any,
    timesteps: int,
    train_bars: pd.DataFrame,
    train_sec: pd.DataFrame | None,
    cfg: Any,
    cache_dir: Path,
    primary_sym: str,
    primary_ticks: Any,
    schedule_path: Path,
    feature_columns: list[str],
    run_dir: Path,
    env_vae: Any,
    pre_ny_by_date: Any,
    device: Any,
    vae_model: Any,
    seed: int,
) -> None:
    """Run optional purged walk-forward validation without altering the main split flow."""
    if not args.walk_forward:
        return

    log.info("Running walk-forward validation with %d splits...", args.wf_splits)
    wf_steps = args.wf_steps or timesteps
    wf_results: list[dict[str, float | int | str]] = []

    for split in purged_walk_forward(
        len(train_bars),
        n_splits=args.wf_splits,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    ):
        fold_train_bars = train_bars.iloc[split.train_idx]
        fold_test_bars = train_bars.iloc[split.test_idx]
        fold_train_feat, fold_test_feat, fold_cache = build_walkforward_fold_features(
            train_bars,
            train_sec,
            cfg,
            fold_train_bars,
            fold_test_bars,
            cache_dir,
            str(primary_sym),
        )
        fold_train_feat = attach_policy_market_context(
            fold_train_bars,
            fold_train_feat,
            primary_ticks,
            schedule_path=schedule_path,
            session_start=str(cfg.session.start),
            session_end=str(cfg.session.end),
        )
        fold_test_feat = attach_policy_market_context(
            fold_test_bars,
            fold_test_feat,
            primary_ticks,
            schedule_path=schedule_path,
            session_start=str(cfg.session.start),
            session_end=str(cfg.session.end),
        )
        if list(fold_train_feat.columns) != feature_columns:
            raise RuntimeError(f"Walk-forward fold {split.fold} changed feature schema")

        log.info(
            "WF fold %d: train=%d test=%d",
            split.fold,
            len(fold_train_bars),
            len(fold_test_bars),
        )

        fold_mmap = _publish_obs_memmap(
            fold_train_feat,
            fold_train_bars,
            cfg,
            run_dir / f"obs_features_fold{split.fold}.npy",
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
            tickbook=ticks_covering(primary_ticks, fold_train_bars),
            arch=args.arch,
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
                tickbook=ticks_covering(primary_ticks, fold_train_bars),
                arch=args.arch,
            ),
            seed=int(seed),
        )
        fold_model.learn(total_timesteps=wf_steps, callback=None, progress_bar=False)

        provider = build_eval_provider(
            fold_model,
            cfg=cfg,
            args=args,
            env_vae=env_vae,
            pre_ny_by_date=pre_ny_by_date,
            arch=args.arch,
        )

        fold_result = _evaluate_final_split(
            provider,
            bars=fold_test_bars,
            features=fold_test_feat,
            obs_window=cfg.env.obs_window,
            initial_balance=cfg.account.initial_balance,
            guardrail_kwargs=eval_guardrail_kwargs(cfg),
            risk_frac_range=strategy_risk_ranges(cfg)[0],
            rr_ratio_range=strategy_risk_ranges(cfg)[1],
            swing_buffer_pts=cfg.risk.swing_buffer_pts,
            sl_mode=str(cfg.risk.get("sl_mode", "fixed")),
            contract_size=cfg.account.contract_size,
            max_loss_per_trade_usd=max_loss_per_trade(cfg),
            dsr_eta=cfg.env.reward_dsr_eta,
            max_episode_steps=max_episode_steps(cfg),
            continuous_actions=(args.algo == "sac"),
            use_sweep_reward=(args.reward == "sweep"),
            block_overnight=bool(cfg.env.get("block_overnight", True)),
            eod_risk=dict(cfg.env.get("eod_risk", {})),
            strategy=strategy_from_cfg(cfg)[0],
            strategy_actions=bool(cfg.env.get("strategy_actions", False)),
            entry_state_machine=bool(cfg.env.get("entry_state_machine", False)),
            candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
            arm_requires_retest=bool(cfg.env.get("arm_requires_retest", False)),
            entry_state_observation=bool(cfg.env.get("entry_state_observation", True)),
            strict_research_eligibility=bool(cfg.env.get("strict_research_eligibility", False)),
            use_vae=args.use_vae,
            vae=env_vae,
            pre_ny_by_date=pre_ny_by_date,
            min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
            min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
            max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
            entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
            reward_mode=str(cfg.env.get("reward_mode", "dsr")),
            entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
            agent_direction_control=bool(cfg.env.get("agent_direction_control", False)),
            direction_override_threshold=float(cfg.env.get("direction_override_threshold", 0.0)),
            allow_agent_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
            sl_mode_defer_threshold=float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
            allow_agent_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
            tp_mode_defer_threshold=float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
            allow_multi_tp=bool(cfg.env.get("allow_multi_tp", False)),
            allow_simplex=bool(cfg.env.get("allow_simplex", True)),
            tp_breakeven_alpha=float(cfg.env.get("tp_breakeven_alpha", 0.5)),
            tickbook=ticks_covering(primary_ticks, fold_test_bars),
            fill_delay_ms=fill_delay_ms(cfg),
        )
        fold_m = calculate_metrics(
            fold_result["equity"],
            trades=fold_result["trades"],
            n_sessions=fold_result.get("n_sessions", 1),
            n_breach_sessions=fold_result.get("n_breach_sessions", 0),
            periods_per_year=evaluation_periods_per_year(cfg),
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
                "feature_cache_name": fold_cache.name,
                "normalization_mode": "frozen_fold",
            }
        )

    wf_sharpes = [float(result["sharpe"]) for result in wf_results]
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
