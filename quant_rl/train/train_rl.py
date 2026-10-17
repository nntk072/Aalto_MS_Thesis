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
import hashlib
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from quant_rl.backtest.costs import CostModel
from quant_rl.config import apply_overrides, load_config
from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.session import ny_session_mask
from quant_rl.data.split import (
    get_split_config,
    make_train_mask,
    split_bars,
    split_train_test,
)
from quant_rl.data.ticks import TickBook, tick_execution_coverage, ticks_covering
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
from quant_rl.eval.rollout import BAR_CLOSE_EXPOSURE_MEASURE, evaluate_model
from quant_rl.eval.run_manifest import (
    array_sha256,
    build_sequence_feature_schema,
    build_stop_candidate_schema,
    build_training_run_manifest,
    dataset_record,
    file_sha256,
    write_run_manifest,
)
from quant_rl.evaluation import calculate_metrics, evaluation_periods_per_year
from quant_rl.features.build import (
    build_features,
    build_walkforward_fold_features,
    feature_cache_content_hash,
    feature_cache_path,
)
from quant_rl.models.agent import build_agent
from quant_rl.models.ppo_policy import install_post_update_std_clip
from quant_rl.train.ablation_utils import load_variant_config, merge_variant_cfg  # noqa: E402
from quant_rl.train.auxiliary_training import AuxiliaryTrainerCallback
from quant_rl.train.callbacks import (
    BestCheckpointEvalCallback,
    ClipLogStdCallback,
    EarlyAbortCallback,
    EpisodeEquityCallback,
    LagrangianRiskCallback,
    LearningDiagnosticsCallback,
    PeriodicCheckpointCallback,
    ProgressLoggerCallback,
    save_ppo_checkpoint,
)
from quant_rl.train.dashboard import TrainingDashboardCallback, ppo_log_interval
from quant_rl.train.debug_trace import install_flight_recorder
from quant_rl.train.equity_gate import assess_train_equity
from quant_rl.utils.device import get_device, scale_training_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

_STRATEGY_CONFIGS = {
    "po3_ifvg": _CONFIG_DIR / "2_po3_ifvg.yaml",
    "distribution": _CONFIG_DIR / "3_distribution.yaml",
}

_RUN_LOG_NAME = "train.log"


def _attach_run_log(run_dir: Path) -> Path | None:
    """Mirror console logs into ``run_dir/train.log``.

    Adds a file handler to the root logger without removing console handlers,
    so the transcript lives beside the run's model and eval artifacts. A
    second call for the same directory reuses the existing handler instead of
    duplicating records or truncating the file. An unwritable directory
    returns ``None`` rather than aborting training over logging.
    """
    name = str(run_dir / _RUN_LOG_NAME)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == name:
            return Path(name)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(name)
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return Path(name)


def _initialize_value_head(model: Any, *, resumed: bool) -> None:
    """Zero the critic head only for a new run, never for a loaded checkpoint."""
    if resumed:
        return
    value_net = getattr(model.policy, "value_net", None)
    if value_net is None:
        return
    with torch.no_grad():
        for param in value_net.parameters():
            param.zero_()


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


def _training_episodic(cfg: Any) -> bool:
    """Resolve whether risk breaches terminate training episodes.

    ``full_split`` keeps the environment cursor moving through the chronological
    training split after a risk breach; the environment latches the year failure
    and blocks trading according to the existing guardrail semantics. The
    episode then ends at the split boundary, so PPO cannot silently train only
    on repeated early-year prefixes. ``risk_terminated`` preserves the prior
    episodic behavior for explicitly requested legacy runs.
    """
    training = cfg.get("training", {}) if hasattr(cfg, "get") else getattr(cfg, "training", {})
    mode = str(training.get("episode_mode", "full_split")).strip().lower()
    if mode == "full_split":
        if _max_episode_steps(cfg) is not None:
            raise ValueError(
                "training.episode_mode='full_split' requires env.max_episode_steps=null"
            )
        return False
    if mode == "risk_terminated":
        return True
    raise ValueError(
        f"training.episode_mode must be 'full_split' or 'risk_terminated', got {mode!r}"
    )


def _build_lagrangian_callback(cfg: Any, *, algo: str, mvp: bool) -> Any | None:
    """Build the Q5 survival-objective callback only for annual PPO episodes."""
    constrained = cfg.get("constrained_rl", {})
    if not bool(constrained.get("enabled", False)):
        return None
    mode = str(cfg.get("training", {}).get("episode_mode", "full_split"))
    if algo != "ppo":
        raise ValueError("constrained_rl is implemented for PPO only")
    if mode != "full_split" or _training_episodic(cfg):
        raise ValueError("constrained_rl requires training.episode_mode='full_split'")
    if mvp:
        raise ValueError(
            "constrained_rl requires the complete chronological train split; MVP is invalid"
        )
    return LagrangianRiskCallback(
        episode_cost_budget=float(constrained.get("episode_cost_budget", 1.0)),
        dual_learning_rate=float(constrained.get("dual_learning_rate", 0.1)),
        lambda_initial=float(constrained.get("lambda_initial", 0.0)),
        lambda_max=float(constrained.get("lambda_max", 100.0)),
        gamma=float(cfg.ppo.gamma),
    )


def _validate_q5_resume_state(cfg: Any, *, resume_checkpoint: str | Path | None) -> None:
    """Reject Q5 resume until PPO checkpoints also restore full-split env state.

    PPO checkpoints preserve the learned multiplier, but the training CLI builds
    fresh environments when resuming. That would combine a partial episode's
    saved risk cost with a new episode starting at the first training bar.
    """
    constrained = cfg.get("constrained_rl", {})
    if resume_checkpoint and bool(constrained.get("enabled", False)):
        raise ValueError(
            "Q5 constrained_rl cannot resume from a PPO-only checkpoint: the "
            "full-split training environment/account state is not restored"
        )


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
    manipulation_filter = str(strat_cfg.get("manipulation_filter", "against"))
    enforce_gate = bool(strat_cfg.get("entry", {}).get("enforce_gate", False))
    reward_cfg = strat_cfg.get("reward", {})
    weight = float(reward_cfg.get("strategy_weight", 0.0)) if reward_cfg else 0.0

    reward: PO3Reward | DistributionReward | None = None
    strategy: TradingStrategy
    if name == "po3_ifvg":
        ifvg_cfg = strat_cfg.get("ifvg", {}) or {}
        strategy = PO3IFVGStrategy(
            enforce_gate=enforce_gate,
            require_price_retest=bool(ifvg_cfg.get("require_price_retest", True)),
            entry_mode=str(strat_cfg.get("entry", {}).get("mode", "legacy")),
            candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
            variant_id=str(strat_cfg.get("variant_id", "po3_ifvg_legacy")),
        )
        strategy.manipulation_filter = manipulation_filter
        reward = PO3Reward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            manipulation_penalty=float(reward_cfg.get("manipulation_penalty", 0.02)),
            invalid_ifvg_penalty=float(reward_cfg.get("invalid_ifvg_penalty", 0.01)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
        )
    elif name == "distribution":
        previous_session_pd_gate = bool(
            strat_cfg.get("entry", {}).get("previous_session_pd_gate", False)
        )
        if previous_session_pd_gate and not bool(
            cfg.features.get("include_previous_session_pd", False)
        ):
            raise ValueError(
                "strategy.entry.previous_session_pd_gate requires "
                "features.include_previous_session_pd=true"
            )
        strategy = DistributionStrategy(
            enforce_gate=enforce_gate,
            entry_mode=str(strat_cfg.get("entry", {}).get("mode", "legacy")),
            candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
            previous_session_pd_gate=previous_session_pd_gate,
            variant_id=str(strat_cfg.get("variant_id", "distribution_legacy")),
        )
        strategy.manipulation_filter = manipulation_filter
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


def _publish_obs_memmap(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: Any,
    path: Path,
) -> str | None:
    """Write the model observation matrix once so workers mmap it read-only.

    ``n_envs <= 1`` keeps the private ``to_numpy`` copy. The pandas frame is
    still passed in; this only removes the extra float32 matrix per worker.
    """
    if int(cfg.env.get("n_envs", 1)) <= 1:
        return None
    strategy, _, _ = _strategy_from_cfg(cfg)
    from quant_rl.envs.env_bundle_builder import observation_features

    array, _ = observation_features(bars, features, strategy.raw_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path)


def _early_abort_callback(cfg: Any) -> EarlyAbortCallback | None:
    """Build the mid-training stop, or ``None`` when the config disables it."""
    training = cfg.get("training") if hasattr(cfg, "get") else None
    block: Any = {}
    if training is not None:
        raw = training.get("early_abort", {})
        if raw is not None:
            block = raw
    if not bool(block.get("enabled", True)):
        return None
    return EarlyAbortCallback(
        min_timesteps=int(block.get("min_timesteps", 5_000_000)),
        window=int(block.get("window", 4)),
        stop_on_nonfinite=bool(block.get("stop_on_nonfinite", True)),
        stop_on_no_trades=bool(block.get("stop_on_no_trades", False)),
        stop_on_equity_gate=bool(block.get("stop_on_equity_gate", False)),
    )


def _dashboard_block(cfg: Any) -> Any:
    """``training.dashboard`` config, or an empty mapping when it is absent."""
    training = cfg.get("training") if hasattr(cfg, "get") else None
    if training is None:
        return {}
    raw = training.get("dashboard", {})
    return {} if raw is None else raw


def _build_eval_common(
    cfg: Any,
    args: Any | None = None,
    env_vae: Any | None = None,
    pre_ny_by_date: dict[Any, Any] | None = None,
    arch: str | None = None,
) -> tuple[dict[str, Any], Any, Any, float, bool]:
    """Kwargs shared by the in-loop dashboard slice and the final eval."""
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    strategy_actions = bool(cfg.env.get("strategy_actions", False))
    sl_buffer_pts = float(cfg.env.get("sl_buffer_pts", 0.0))
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    use_mtf, mtf_windows = _mtf_settings(cfg, arch)
    eval_common = {
        "obs_window": cfg.env.obs_window,
        "initial_balance": cfg.account.initial_balance,
        "cost_model": _cost_model(cfg),
        "guardrail_kwargs": _eval_guardrail_kwargs(cfg),
        "risk_frac_range": risk_frac_range,
        "rr_ratio_range": rr_ratio_range,
        "sl_mode": str(cfg.risk.get("sl_mode", "fixed")),
        "tp_mode": str(cfg.risk.get("tp_mode", "fixed")),
        "tp_reference": str(cfg.risk.get("tp_reference", "structural")),
        "swing_buffer_pts": cfg.risk.swing_buffer_pts,
        "contract_size": cfg.account.contract_size,
        "max_loss_per_trade_usd": _max_loss_per_trade(cfg),
        "dsr_eta": cfg.env.reward_dsr_eta,
        "continuous_actions": (args.algo == "sac") if args is not None else False,
        "use_sweep_reward": (args.reward == "sweep") if args is not None else False,
        "block_overnight": bool(cfg.env.get("block_overnight", True)),
        "eod_risk": dict(cfg.env.get("eod_risk", {})),
        "use_vae": getattr(args, "use_vae", False) if args is not None else False,
        "vae": env_vae,
        "strategy": strategy,
        "strategy_actions": strategy_actions,
        "entry_state_machine": bool(cfg.env.get("entry_state_machine", False)),
        "candidate_max_age_bars": int(cfg.env.get("candidate_max_age_bars", 5)),
        "arm_requires_retest": bool(cfg.env.get("arm_requires_retest", False)),
        "entry_state_observation": bool(cfg.env.get("entry_state_observation", True)),
        "strict_research_eligibility": bool(cfg.env.get("strict_research_eligibility", False)),
        "strategy_reward": strategy_reward,
        "strategy_weight": strategy_weight,
        "sl_buffer_pts": sl_buffer_pts,
        "pre_ny_by_date": pre_ny_by_date,
        "min_sl_atr_mult": float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        "min_sl_points": float(cfg.risk.get("min_sl_points", 0.0)),
        "max_entries_per_session": int(cfg.env.get("max_entries_per_session", 0)),
        "open_manipulation_bars": int(cfg.env.get("open_manipulation_bars", 10)),
        "entry_cooldown_bars": int(cfg.env.get("entry_cooldown_bars", 0)),
        "reward_mode": str(cfg.env.get("reward_mode", "dsr")),
        "fixed_risk_usd": float(cfg.env.get("fixed_risk_usd", 0.0)),
        "risk_mode": str(cfg.env.get("risk_mode", "dynamic")),
        "observation_schema_version": str(
            cfg.env.get("observation_schema_version", "account-state-v1")
        ),
        "observation_age_scale_bars": float(cfg.env.get("observation_age_scale_bars", 60.0)),
        "entry_intensity_threshold": float(cfg.env.get("entry_intensity_threshold", 0.0)),
        "peak_trailing_dd_limit": float(cfg.env.get("peak_trailing_dd_limit", 0.0)),
        "agent_direction_control": bool(cfg.env.get("agent_direction_control", False)),
        "direction_override_threshold": float(cfg.env.get("direction_override_threshold", 0.0)),
        "fill_delay_ms": _fill_delay_ms(cfg),
        "randomized_fill_latency_ms": _randomized_fill_latency_ms(cfg),
        "tick_execution_required": _tick_execution_required(cfg),
        "max_tick_quote_wait_ms": _max_tick_quote_wait_ms(cfg),
        "execution_seed": (
            int(getattr(args, "seed"))
            if args is not None and getattr(args, "seed", None) is not None
            else None
        ),
        "allow_multi_tp": bool(cfg.env.get("allow_multi_tp", False)),
        "allow_simplex": bool(cfg.env.get("allow_simplex", True)),
        "allow_ema_exit": bool(cfg.env.get("allow_ema_exit", True)),
        "allow_agent_tp_mode": bool(cfg.env.get("allow_agent_tp_mode", False)),
        "allow_agent_sl_mode": bool(cfg.env.get("allow_agent_sl_mode", True)),
        "sl_mode_defer_threshold": float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
        "tp_mode_defer_threshold": float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
        "tp_breakeven_alpha": float(cfg.env.get("tp_breakeven_alpha", 0.5)),
        "risk_floor": float(cfg.env.get("risk_floor", 0.0005)),
        "mtf": use_mtf,
        "mtf_windows": mtf_windows,
    }
    return eval_common, strategy, strategy_reward, strategy_weight, strategy_actions


def _evaluate_final_split(model: Any, **kwargs: Any) -> dict[str, Any]:
    """Evaluate a reported split with deterministic policy actions.

    The training equity gate and the report-only test must share the campaign's
    deterministic evaluation contract. Walk-forward fold metrics are also used
    for model selection and therefore use the same deterministic contract. The
    periodic dashboard monitor remains a separate stochastic progress diagnostic.
    """
    kwargs["deterministic"] = True
    return evaluate_model(model, **kwargs)


def _log_effective_ppo(model: Any, cfg: Any) -> None:
    """Print the PPO settings the run actually uses, after the n_envs split."""
    n_steps = int(model.n_steps)
    n_envs = int(model.n_envs)
    aggregate_rollout_steps = n_steps * n_envs
    batch_size = int(cfg.ppo.batch_size)
    minibatches_per_epoch = (aggregate_rollout_steps + batch_size - 1) // batch_size
    requested_n_envs = int(cfg.env.get("n_envs", 1))
    if n_envs != requested_n_envs:
        raise RuntimeError(
            f"effective PPO n_envs={n_envs} does not match resolved config "
            f"n_envs={requested_n_envs}"
        )
    strategy_name = (
        str(cfg.strategy.get("name", "strategy"))
        if bool(cfg.env.get("strategy_actions", False))
        else "baseline"
    )
    log.info(
        "PPO effective: logical_n_envs=%d sb3_n_steps_per_env=%d "
        "aggregate_rollout_steps=%d batch_size=%d n_epochs=%s minibatches_per_epoch=%d "
        "updates_per_rollout=%d lr=%s "
        "gamma=%s clip=%s ent_coef=%s log_std=[%s,%s] reward_mode=%s strategy=%s",
        n_envs,
        n_steps,
        aggregate_rollout_steps,
        batch_size,
        cfg.ppo.n_epochs,
        minibatches_per_epoch,
        minibatches_per_epoch * int(cfg.ppo.n_epochs),
        cfg.ppo.learning_rate,
        cfg.ppo.gamma,
        cfg.ppo.clip_range,
        model.ent_coef,
        cfg.ppo.get("log_std_min", -0.7),
        cfg.ppo.get("log_std_max", 0.0),
        cfg.env.get("reward_mode", "dsr"),
        strategy_name,
    )


def _training_architecture_provenance(model: Any, cfg: Any, arch: str) -> dict[str, Any]:
    """Describe the actual encoder, policy heads, and vectorization used."""
    extractor = model.policy.features_extractor
    env = model.get_env()
    remotes = getattr(env, "remotes", None)
    physical_workers = len(remotes) if remotes is not None else 1
    head_cfg = cfg.agent.get("head_arch", {})
    encoder_spec = {
        "arch": arch,
        "encoder_class": type(extractor).__name__,
        "encoder_channels": getattr(extractor, "channels", None),
        "encoder_dilations": list(extractor.dilations) if hasattr(extractor, "dilations") else None,
        "encoder_kernel_size": getattr(extractor, "kernel_size", None),
        "encoder_latent_dim": int(getattr(extractor, "latent_dim", 128)),
        "encoder_parameters": {
            key: getattr(extractor, key)
            for key in ("hidden_size", "num_layers", "d_model", "nhead", "dim_feedforward")
            if hasattr(extractor, key)
        },
    }
    return {
        **encoder_spec,
        "quality_variant_id": str(cfg.get("quality_variant_id", "")) or None,
        "actor_architecture": str(cfg.ppo.get("actor_arch", "shared")),
        "semantic_head_width": int(cfg.ppo.get("semantic_head_width", 64)),
        "policy_head_architecture": [int(width) for width in head_cfg.get(arch, [256, 128])],
        "policy_parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "configured_vector_backend": str(cfg.env.get("vector_backend", "subproc")),
        "vector_backend": type(env).__name__,
        "logical_n_envs": int(model.n_envs),
        "sb3_n_steps_per_env": int(model.n_steps),
        "aggregate_rollout_steps": int(model.n_envs * model.n_steps),
        "batch_size": int(cfg.ppo.batch_size),
        "n_epochs": int(cfg.ppo.n_epochs),
        "minibatches_per_epoch": int(
            (model.n_envs * model.n_steps + int(cfg.ppo.batch_size) - 1) // int(cfg.ppo.batch_size)
        ),
        "updates_per_rollout": int(
            (
                (model.n_envs * model.n_steps + int(cfg.ppo.batch_size) - 1)
                // int(cfg.ppo.batch_size)
            )
            * int(cfg.ppo.n_epochs)
        ),
        "physical_worker_count": int(physical_workers),
    }


def _fill_delay_ms(cfg: Any) -> int:
    """Milliseconds between the crossing tick and the SL/TP fill."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 0
    try:
        return int(execution.get("fill_delay_ms", 0))
    except Exception:
        return 0


def _randomized_fill_latency_ms(cfg: Any) -> float:
    """Maximum exclusive randomized tick latency; zero means immediate tick."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 0.0
    try:
        return float(execution.get("randomized_fill_latency_ms", 0.0))
    except Exception:
        return 0.0


def _tick_execution_required(cfg: Any) -> bool:
    """Whether the effective config forbids candle-priced execution fills."""
    costs = cfg.get("costs", {}) if hasattr(cfg, "get") else {}
    return bool(costs.get("use_tick_execution", False)) if hasattr(costs, "get") else False


def _max_tick_quote_wait_ms(cfg: Any) -> int:
    """Maximum permitted delay from sampled arrival to the next tick quote."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 60_000
    try:
        return int(execution.get("max_tick_quote_wait_ms", 60_000))
    except Exception:
        return 60_000


def _tick_execution_split_schedule(
    index: pd.DatetimeIndex,
    train_mask: pd.Series,
    test_start: str,
    *,
    session_labels: pd.Series | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the observed NY decisions and next-bar quote anchors by split.

    ``TradingEnv`` advances through NY-eligible bars and resolves each step's
    execution/valuation quote at the following market bar. Requiring coverage
    for every overnight M1 row is unrelated to this environment's execution
    schedule and can reject otherwise complete NY-session tick data. The
    returned timestamps describe the observed-bar schedule only; they do not
    define which intervals the CFD provider was expected to publish.
    """
    if session_labels is None:
        ny_mask = ny_session_mask(index).to_numpy(dtype=bool)
    else:
        if not session_labels.index.equals(index):
            raise ValueError("session_labels index must match tick-preflight index")
        ny_mask = session_labels.astype(str).eq("ny").to_numpy(dtype=bool)

    train_mask_values = np.asarray(train_mask.to_numpy(), dtype=bool)
    if len(train_mask_values) != len(index) or not train_mask.index.equals(index):
        raise ValueError("train_mask index must match tick-preflight index")
    test_start_ts = pd.Timestamp(test_start)
    if index.tz is not None:
        if test_start_ts.tzinfo is None:
            test_start_ts = test_start_ts.tz_localize(index.tz)
        else:
            test_start_ts = test_start_ts.tz_convert(index.tz)

    # Match TradingEnv._step_untraced: every eligible NY step consumes the
    # following M1 market quote (clamped to the final available row).
    next_market = np.minimum(np.arange(len(index), dtype=np.int64) + 1, len(index) - 1)
    train_positions = np.flatnonzero(train_mask_values & ny_mask)
    test_positions = np.flatnonzero((index >= test_start_ts) & ny_mask)
    return {
        "train": {
            "decision_positions": train_positions,
            "decision_times": index[train_positions],
            "quote_anchor_positions": next_market[train_positions],
            "quote_anchor_times": index[next_market[train_positions]],
        },
        "test": {
            "decision_positions": test_positions,
            "decision_times": index[test_positions],
            "quote_anchor_positions": next_market[test_positions],
            "quote_anchor_times": index[next_market[test_positions]],
        },
    }


def _tick_execution_split_coverage(
    tickbook: TickBook | None,
    index: pd.DatetimeIndex,
    train_mask: pd.Series,
    test_start: str,
    *,
    session_labels: pd.Series | None = None,
    max_latency_ns_exclusive: int,
    max_quote_wait_ms: int,
) -> dict[str, dict[str, Any]]:
    """Preflight tick quotes at the timestamps actually used by NY env steps."""
    schedule = _tick_execution_split_schedule(
        index, train_mask, test_start, session_labels=session_labels
    )
    options = {
        "max_latency_ns_exclusive": max_latency_ns_exclusive,
        "max_quote_wait_ms": max_quote_wait_ms,
    }
    return {
        split: tick_execution_coverage(tickbook, rows["quote_anchor_times"], **options)
        for split, rows in schedule.items()
    }


def _mtf_settings(cfg: Any, arch: str | None) -> tuple[bool, dict[str, int]]:
    """``mtf`` is on only when the selected architecture is the multi-window encoder."""
    from quant_rl.envs.observation import DEFAULT_HTF_WINDOWS

    chosen = arch
    if chosen is None:
        agent = cfg.get("agent") if hasattr(cfg, "get") else None
        chosen = str(agent.get("arch", "tcn")) if agent is not None else "tcn"
    windows = dict(DEFAULT_HTF_WINDOWS)
    encoder = cfg.get("encoder") if hasattr(cfg, "get") else None
    raw = encoder.get("windows") if encoder is not None and hasattr(encoder, "get") else None
    if raw:
        for key, value in dict(raw).items():
            windows[str(key)] = int(value)
    return str(chosen) == "mtf", windows


def _cost_model(cfg: Any) -> CostModel:
    """Build the broker ``CostModel`` from ``costs.*``.

    ``make_env`` used to leave the env on its ``COST_US100`` default, so the
    ``costs.slippage_points: 5`` YAML value never reached fills. Spreads come
    from the primary-symbol YAML key; slippage is quoted in price points.
    """
    costs = cfg.get("costs", {}) if hasattr(cfg, "get") else {}
    if costs is None:
        costs = {}
    get = costs.get if hasattr(costs, "get") else lambda _k, _d=None: _d
    primary = str(cfg.data.primary) if hasattr(cfg, "data") else "US100.cash"
    spread_key = "spread_us500" if "US500" in primary else "spread_us100"
    return CostModel(
        spread_points=float(get(spread_key, 0.6)),
        slippage_points=float(get("slippage_points", 0.0)),
        commission_per_lot=float(get("commission_per_lot", 0.0)),
        point_size=float(get("point_size", 0.01)),
    )


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
    tickbook: TickBook | None = None,
    arch: str | None = None,
) -> TradingEnv:
    continuous_actions = algo == "sac"
    use_sweep_reward = reward == "sweep"
    use_mtf, mtf_windows = _mtf_settings(cfg, arch)
    strategy, strategy_reward, strategy_weight = _strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = _strategy_risk_ranges(cfg)
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        cost_model=_cost_model(cfg),
        guardrail_kwargs=_guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        sl_mode=str(cfg.risk.get("sl_mode", "fixed")),
        tp_mode=str(cfg.risk.get("tp_mode", "fixed")),
        tp_reference=str(cfg.risk.get("tp_reference", "structural")),
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=_max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=_max_episode_steps(cfg),
        episodic=episodic,
        continuous_actions=continuous_actions,
        use_sweep_reward=use_sweep_reward,
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        entry_state_machine=bool(cfg.env.get("entry_state_machine", False)),
        candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
        arm_requires_retest=bool(cfg.env.get("arm_requires_retest", False)),
        entry_state_observation=bool(cfg.env.get("entry_state_observation", True)),
        observation_schema_version=str(
            cfg.env.get("observation_schema_version", "account-state-v1")
        ),
        observation_age_scale_bars=float(cfg.env.get("observation_age_scale_bars", 60.0)),
        strict_research_eligibility=bool(cfg.env.get("strict_research_eligibility", False)),
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        open_manipulation_bars=int(cfg.env.get("open_manipulation_bars", 10)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        fixed_risk_usd=float(cfg.env.get("fixed_risk_usd", 0.0)),
        risk_mode=str(cfg.env.get("risk_mode", "dynamic")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        peak_trailing_dd_limit=float(cfg.env.get("peak_trailing_dd_limit", 0.0)),
        agent_direction_control=bool(cfg.env.get("agent_direction_control", False)),
        direction_override_threshold=float(cfg.env.get("direction_override_threshold", 0.0)),
        allow_agent_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
        sl_mode_defer_threshold=float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
        allow_agent_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
        tp_mode_defer_threshold=float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
        allow_multi_tp=bool(cfg.env.get("allow_multi_tp", False)),
        allow_simplex=bool(cfg.env.get("allow_simplex", True)),
        allow_ema_exit=bool(cfg.env.get("allow_ema_exit", True)),
        tp_breakeven_alpha=float(cfg.env.get("tp_breakeven_alpha", 0.5)),
        risk_floor=float(cfg.env.get("risk_floor", 0.0005)),
        tickbook=tickbook,
        tick_execution_required=_tick_execution_required(cfg),
        fill_delay_ms=_fill_delay_ms(cfg),
        randomized_fill_latency_ms=_randomized_fill_latency_ms(cfg),
        max_tick_quote_wait_ms=_max_tick_quote_wait_ms(cfg),
        use_vae=use_vae,
        vae=vae,
        pre_ny_by_date=pre_ny_by_date,
        obs_features_mmap=obs_features_mmap,
        mtf=use_mtf,
        mtf_windows=mtf_windows,
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seed list (default: single --seed)",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Variant name from config/experiments.yaml (overrides --strategy if set in variant)",
    )
    parser.add_argument(
        "--quality-variant",
        default=None,
        help="New staged variant from config/quality_experiments.yaml",
    )
    parser.add_argument("--mvp", action="store_true", help="MVP mode: first 30 days only")
    parser.add_argument("--force", action="store_true", help="Force data pipeline rerun")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Continue PPO optimizer/policy state from an intact periodic checkpoint",
    )
    parser.add_argument(
        "--allow-legacy-observation-schema",
        action="store_true",
        help="Explicitly resume an unversioned account-state-v1 checkpoint",
    )
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="RL algorithm")
    parser.add_argument(
        "--arch",
        choices=["tcn", "lite_tcn", "gru", "transformer", "mtf"],
        default=None,
        help="Encoder architecture",
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
    parser.add_argument(
        "--allow-locked-oos-for-selection",
        action="store_true",
        help="Rank on locked-OOS Sharpe. Only use if you knowingly tune on the holdout.",
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
    """Load base config and merge strategy variant if requested.

    CLI ``overrides`` apply *after* the strategy overlay merge: the Idea 1
    variant pins ``env.reward_mode: pnl``, which used to clobber an explicit
    ``env.reward_mode=rr`` override and silently ran every RR run as PnL.
    """
    cfg = load_config(None, config_path=args.config)
    # Merge the strategy variant config (Idea 1/2) on top of the base config.
    # Baseline (Idea 3) leaves cfg untouched.
    if args.strategy in _STRATEGY_CONFIGS:
        variant_path = _STRATEGY_CONFIGS[args.strategy]
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(variant_path)))
        log.info("Merged strategy config: %s", variant_path)
    elif args.strategy != "baseline":
        raise ValueError(f"unknown strategy: {args.strategy}")

    # Merge variant flags from experiments.yaml (--variant).
    if getattr(args, "variant", None):
        variant_dict = load_variant_config(args.variant)
        variant_strategy = str(variant_dict.get("strategy", args.strategy))
        include_pd = bool(variant_dict.get("include_pd_context", False))
        cfg = merge_variant_cfg(
            cfg,
            strategy=variant_strategy,
            include_pd_context=include_pd,
            variant=variant_dict,
        )
        log.info(
            "Merged variant: %s (strategy=%s include_pd=%s)",
            args.variant,
            variant_strategy,
            include_pd,
        )

    quality_variant = getattr(args, "quality_variant", None)
    if quality_variant:
        if getattr(args, "variant", None) or args.strategy != "baseline":
            raise ValueError(
                "--quality-variant cannot be combined with historical --variant or a "
                "non-baseline --strategy"
            )
        from quant_rl.train.quality_registry import apply_quality_variant

        cfg = apply_quality_variant(cfg, str(quality_variant))
        log.info("Applied staged RL quality variant: %s", quality_variant)

    return apply_overrides(cfg, args.overrides)


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
    entry_diag: dict[str, Any] | None = None,
    entry_state_diagnostics: dict[str, Any] | None = None,
    action_space_width: int | None = None,
    observation_account_dim: int | None = None,
    observation_space_shapes: dict[str, list[int]] | None = None,
    action_schema: dict[str, Any] | None = None,
    observation_schema: dict[str, Any] | None = None,
    env_flags: dict[str, Any] | None = None,
    oos_role: str = "final_report_only",
) -> dict[str, Any]:
    """Build the training_log.json dict from run results."""
    from quant_rl.train.ablation_utils import compute_tier1_funnel

    funnel = compute_tier1_funnel(entry_diag, entry_state_diagnostics)
    log_dict: dict[str, Any] = {
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
        "test_breaches": test_result.get("n_breach_sessions", 0),
        "test_n_sessions": int(test_result.get("n_sessions", 0)),
        "test_days_traded": int(test_result.get("days_traded", 0)),
        "test_survived_full_year": bool(test_result.get("survived_full_year", False)),
        "test_completed_data_split": bool(test_result.get("completed_data_split", False)),
        "test_episode_end_reason": test_result.get("episode_end_reason"),
        "test_max_trailing_dd": float(test_result.get("max_trailing_dd", test_m.max_drawdown)),
        "test_fail_time": (
            str(test_result["fail_time"]) if test_result.get("fail_time") is not None else None
        ),
        "timestamp": datetime.now().isoformat(),
    }
    log_dict.update(funnel)
    if action_space_width is not None:
        log_dict["action_space_width"] = action_space_width
    if observation_account_dim is not None:
        log_dict["observation_account_dim"] = observation_account_dim
    if observation_space_shapes is not None:
        log_dict["observation_space_shapes"] = observation_space_shapes
    if action_schema is not None:
        log_dict["action_schema"] = action_schema
    if observation_schema is not None:
        log_dict["observation_schema"] = observation_schema
    if env_flags is not None:
        log_dict["env_flags"] = env_flags
    log_dict["oos_role"] = oos_role
    return log_dict


def _action_space_width(env: Any) -> int | None:
    """Return the action space width for logging."""
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return None
    shape = getattr(action_space, "shape", None)
    if shape:
        return int(shape[0])
    if hasattr(action_space, "n"):
        return int(action_space.n)
    return None


def _action_schema_manifest(env: Any) -> dict[str, Any] | None:
    """Return declared coordinate names and flags, never infer from width."""
    schema = getattr(env, "action_schema", None)
    to_dict = getattr(schema, "to_dict", None)
    if not callable(to_dict):
        return None
    result = to_dict()
    return result if isinstance(result, dict) else None


def _observation_schema_manifest(env: Any) -> dict[str, Any] | None:
    """Return explicit account observation coordinate meanings."""
    schema = getattr(env, "observation_schema", None)
    to_dict = getattr(schema, "to_dict", None)
    if not callable(to_dict):
        return None
    result = to_dict()
    return result if isinstance(result, dict) else None


def _sequence_feature_schema_manifest(env: Any) -> dict[str, Any]:
    """Return ordered model sequence coordinates from the effective environment."""
    columns = getattr(env, "_obs_columns", None)
    if not isinstance(columns, (list, tuple)):
        raise RuntimeError("training environment does not expose sequence feature columns")
    return build_sequence_feature_schema([str(column) for column in columns])


def _training_protocol_manifest(
    *, model: Any, cfg: Any, args: Any, seed: int, requested_steps: int
) -> dict[str, Any]:
    """Persist the effective PPO/reward/risk protocol for later reproduction."""
    n_envs = int(model.n_envs)
    n_steps = int(model.n_steps)
    batch_size = int(cfg.ppo.batch_size)
    minibatches = (n_envs * n_steps + batch_size - 1) // batch_size
    quality_variant = str(cfg.get("quality_variant_id", ""))
    quality_lineage = None
    if quality_variant:
        from quant_rl.train.quality_registry import quality_variant_metadata

        quality_lineage = quality_variant_metadata(quality_variant)
    ftmo = OmegaConf.to_container(cfg.get("ftmo", {}), resolve=True)
    return {
        "quality_variant": quality_lineage,
        "constrained_rl": {
            "enabled": bool(cfg.get("constrained_rl", {}).get("enabled", False)),
            "method": "primal_dual_scalarized_return_ppo",
            "cost": "gamma_discounted_sum_of_max_daily_max_loss_trailing_normalized_risk_increments_per_episode",
            "episode_cost_budget": float(
                cfg.get("constrained_rl", {}).get("episode_cost_budget", 1.0)
            ),
            "dual_learning_rate": float(
                cfg.get("constrained_rl", {}).get("dual_learning_rate", 0.1)
            ),
            "lambda_initial": float(cfg.get("constrained_rl", {}).get("lambda_initial", 0.0)),
            "lambda_max": float(cfg.get("constrained_rl", {}).get("lambda_max", 100.0)),
            "cost_discount_gamma": float(cfg.ppo.gamma),
        },
        "seed": int(seed),
        "worker_seeds": [int(seed + index) for index in range(n_envs)],
        "requested_steps": int(requested_steps),
        "completed_steps": int(model.num_timesteps),
        "data_source": str(cfg.env.get("data_source", "frames")),
        "walk_forward": {
            "enabled": bool(getattr(args, "walk_forward", False)),
            "folds": int(getattr(args, "wf_splits", 0)),
            "purge_bars": int(getattr(args, "purge_bars", 0)),
            "embargo_bars": int(getattr(args, "embargo_bars", 0)),
        },
        "observation_schema_version": str(
            cfg.env.get("observation_schema_version", "account-state-v1")
        ),
        "observation_age_scale_bars": float(cfg.env.get("observation_age_scale_bars", 60.0)),
        "action_distribution": str(cfg.ppo.get("policy_distribution", "clamped_diagonal_gaussian")),
        "reward": {
            "mode": str(cfg.env.get("reward_mode", "dsr")),
            "dsr_eta": float(cfg.env.get("reward_dsr_eta", 0.01)),
            "fixed_risk_usd": float(cfg.env.get("fixed_risk_usd", 0.0)),
        },
        "risk": {
            "mode": str(cfg.env.get("risk_mode", "dynamic")),
            "ftmo": ftmo,
            "risk_per_trade_limit": float(cfg.get("ftmo", {}).get("risk_per_trade_limit", 1000.0)),
        },
        "ppo": {
            "requested_n_envs": cfg.get("quality_execution", {}).get("requested_n_envs", n_envs),
            "effective_n_envs": cfg.get("quality_execution", {}).get("effective_n_envs", n_envs),
            "exact_n_envs_required": bool(
                cfg.get("quality_execution", {}).get("exact_n_envs", False)
            ),
            "logical_n_envs": n_envs,
            "n_steps_per_env": n_steps,
            "aggregate_rollout_steps": int(n_envs * n_steps),
            "batch_size": batch_size,
            "epochs": int(cfg.ppo.n_epochs),
            "minibatches_per_epoch": minibatches,
            "updates_per_rollout": int(minibatches * int(cfg.ppo.n_epochs)),
            "learning_rate": float(cfg.ppo.learning_rate),
            "gamma": float(cfg.ppo.gamma),
            "gae_lambda": float(cfg.ppo.gae_lambda),
            "clip_range": float(cfg.ppo.clip_range),
            "entropy_coefficient": float(model.ent_coef),
            "actor_architecture": str(cfg.ppo.get("actor_arch", "shared")),
        },
        "environment": {
            "vector_backend": str(cfg.env.get("vector_backend", "subproc")),
            "strategy_actions": bool(cfg.env.get("strategy_actions", False)),
            "entry_state_machine": bool(cfg.env.get("entry_state_machine", False)),
            "entry_state_observation": bool(cfg.env.get("entry_state_observation", True)),
        },
    }


def _observation_account_dim(env: Any) -> int:
    """Return the account observation dimension from the env's observation space."""
    obs_space = getattr(env, "observation_space", None)
    if obs_space is None:
        return 0
    account_space = getattr(obs_space, "spaces", {}).get("account")
    if account_space is None:
        return 0
    return int(account_space.shape[0])


def _observation_space_shapes(env: Any) -> dict[str, list[int]]:
    """Return every observation branch shape for the run manifest."""
    obs_space = getattr(env, "observation_space", None)
    spaces_by_key = getattr(obs_space, "spaces", {})
    return {
        str(key): [int(size) for size in space.shape]
        for key, space in spaces_by_key.items()
        if hasattr(space, "shape")
    }


def _save_seed_artifacts(
    seed_dir: Path,
    seed: int,
    cfg: Any,
    train_result: dict[str, Any],
    test_result: dict[str, Any],
    train_m: Any,
    test_m: Any,
) -> None:
    """Write per-seed artifacts for multi-seed mode."""
    train_trades = train_result.get("trades")
    test_trades = test_result.get("trades")
    trade_log = {
        "train": train_trades.to_dict(orient="records") if train_trades is not None else [],
        "test": test_trades.to_dict(orient="records") if test_trades is not None else [],
    }
    (seed_dir / "trade_log.json").write_text(json.dumps(trade_log, indent=2, default=str))
    entry_diag = {
        "train": train_result.get("entry_diag", {}),
        "test": test_result.get("entry_diag", {}),
    }
    (seed_dir / "entry_diag.json").write_text(json.dumps(entry_diag, indent=2, default=str))
    entry_state = {
        "train": train_result.get("entry_state_diagnostics", {}),
        "test": test_result.get("entry_state_diagnostics", {}),
    }
    (seed_dir / "entry_state.json").write_text(json.dumps(entry_state, indent=2, default=str))
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        (seed_dir / "config.json").write_text(json.dumps(config_dict, indent=2, default=str))
    except Exception:
        pass


def _write_evaluation_artifacts(
    *,
    run_dir: Path,
    seed: int,
    completed_timesteps: int,
    train_result: dict[str, Any],
    test_result: dict[str, Any],
    train_metrics: Any,
    test_metrics: Any,
    train_gate: dict[str, Any],
    account_currency: str,
) -> None:
    """Write an explicit train/test evaluation report with unambiguous roles."""
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    def split_payload(role: str, result: dict[str, Any], metrics: Any) -> dict[str, Any]:
        return {
            "role": role,
            "policy_action_mode": "deterministic",
            "seed": int(seed),
            "completed_training_timesteps": int(completed_timesteps),
            "account_currency": str(account_currency),
            "metrics": asdict(metrics),
            "activity_status": "ACTIVE" if int(metrics.n_trades) > 0 else "INACTIVE",
            "n_sessions": int(result.get("n_sessions", 0)),
            "days_traded": int(result.get("days_traded", 0)),
            "survived_full_year": bool(result.get("survived_full_year", False)),
            "completed_data_split": bool(result.get("completed_data_split", False)),
            "episode_end_reason": result.get("episode_end_reason"),
            "fail_time": (
                str(result["fail_time"]) if result.get("fail_time") is not None else None
            ),
            "max_trailing_dd": float(result.get("max_trailing_dd", metrics.max_drawdown)),
            "breach_events": int(result.get("n_breach_sessions", 0)),
            "entry_diagnostics": result.get("entry_diag", {}),
        }

    train_payload = split_payload("in_sample_training_evaluation", train_result, train_metrics)
    test_payload = split_payload("report_only_test_evaluation", test_result, test_metrics)
    test_payload["oos_role"] = (
        "previously_inspected_holdout; report-only; not pristine confirmation"
    )
    train_payload["equity_gate"] = train_gate
    (evaluation_dir / "train_in_sample.json").write_text(
        json.dumps(train_payload, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )
    (evaluation_dir / "test_report_only.json").write_text(
        json.dumps(test_payload, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "seed": int(seed),
        "completed_training_timesteps": int(completed_timesteps),
        "training_in_sample": train_payload,
        "test_report_only": test_payload,
        "train_equity_gate": train_gate,
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )

    def line(label: str, payload: dict[str, Any]) -> str:
        metrics = payload["metrics"]
        return (
            f"| {label} | {metrics['total_return_pct']:+.2f}% | "
            f"{metrics['total_pnl']:+.2f} {account_currency} | "
            f"{metrics['max_drawdown']:.2%} | "
            f"{metrics['sharpe']:.3f} | {metrics['profit_factor']:.3f} | "
            f"{metrics['win_rate']:.1%} | {metrics['n_trades']} |"
        )

    report = [
        "# Evaluation results",
        "",
        f"Seed: {seed}  ",
        f"Training steps completed: {completed_timesteps:,}",
        "",
        "The dashboard's periodic train-prefix monitor is a progress check. These are the two final replays.",
        "Both final replays use deterministic policy actions.",
        "",
        f"| Evaluation | Return | Net PnL ({account_currency}) | Max drawdown | Sharpe | Profit factor | Win rate | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        line("Train (in-sample)", train_payload),
        line("Test (report-only; previously inspected)", test_payload),
        "",
        f"Train equity gate: {'PASS' if train_gate.get('ok') else 'FAIL'}",
        "",
        "The test split is report-only and must not be described as pristine confirmatory evidence.",
        "",
    ]
    (evaluation_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")


def _aggregate_seed_results(
    *,
    seed_results: list[dict[str, Any]],
    seed: int,
    mvp: bool,
    algo: str,
    arch: str,
    reward: str,
    timesteps: int,
    train_bars: int,
    test_bars: int,
    strategy: str,
    strategy_actions: bool,
    cfg: Any,
    args: Any,
    run_dir: Path,
) -> dict[str, Any]:
    """Aggregate per-seed results into a multi-seed training_log.json."""
    from quant_rl.train.ablation_utils import compute_tier1_funnel

    ok_seeds = [r for r in seed_results if r["test_m"].n_trades > 0]
    n_seeds_ok = len(ok_seeds)

    scalar_metrics = [
        "sharpe",
        "sortino",
        "max_drawdown",
        "total_return_pct",
        "profit_factor",
        "win_rate",
        "avg_trade",
        "turnover",
        "n_trades",
    ]

    aggregated: dict[str, Any] = {
        "seeds": [r["seed"] for r in seed_results],
        "n_seeds": len(seed_results),
        "n_seeds_ok": n_seeds_ok,
        "algo": algo,
        "arch": arch,
        "reward": reward,
        "strategy": strategy,
        "strategy_actions": strategy_actions,
        "mvp": mvp,
        "timesteps": timesteps,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "timestamp": datetime.now().isoformat(),
    }

    for metric in scalar_metrics:
        test_vals = [float(getattr(r["test_m"], metric)) for r in seed_results]
        train_vals = [float(getattr(r["train_m"], metric)) for r in seed_results]
        if test_vals:
            aggregated[f"test_{metric}_mean"] = round(sum(test_vals) / len(test_vals), 4)
            aggregated[f"test_{metric}_std"] = round(float(np.std(test_vals)), 4)
            aggregated[f"test_{metric}_seeds"] = test_vals
        if train_vals:
            aggregated[f"train_{metric}_mean"] = round(sum(train_vals) / len(train_vals), 4)
            aggregated[f"train_{metric}_std"] = round(float(np.std(train_vals)), 4)

    all_ok = all(r["train_gate"]["ok"] for r in seed_results)
    aggregated["train_equity_ok"] = all_ok
    aggregated["train_end_equity"] = float(seed_results[-1]["train_gate"]["end_equity"])
    aggregated["train_equity_slope"] = float(seed_results[-1]["train_gate"]["slope"])
    aggregated["train_equity_reason"] = str(seed_results[-1]["train_gate"]["reason"])

    abort_reasons = [r["abort_reason"] for r in seed_results if r["abort_reason"]]
    aggregated["early_abort_reason"] = abort_reasons[0] if abort_reasons else ""
    aggregated["timesteps_completed"] = seed_results[-1]["model_num_timesteps"]

    first = seed_results[0]
    funnel = compute_tier1_funnel(
        first["test_result"].get("entry_diag"),
        first["test_result"].get("entry_state_diagnostics"),
    )
    aggregated.update(funnel)

    aggregated["oos_role"] = (
        "selection_acknowledged"
        if getattr(args, "allow_locked_oos_for_selection", False)
        else "final_report_only"
    )

    env_flags = {
        key: bool(cfg.env.get(key, False))
        for key in (
            "strategy_actions",
            "allow_agent_sl_mode",
            "allow_agent_tp_mode",
            "allow_multi_tp",
            "allow_simplex",
            "entry_state_machine",
            "entry_state_observation",
            "arm_requires_retest",
        )
    }
    aggregated["env_flags"] = env_flags

    action_widths = [
        r.get("action_space_width") for r in seed_results if r.get("action_space_width") is not None
    ]
    obs_dims = [
        r.get("observation_account_dim")
        for r in seed_results
        if r.get("observation_account_dim") is not None
    ]
    if action_widths:
        aggregated["action_space_width"] = action_widths[0]
    schemas = [r.get("action_schema") for r in seed_results if r.get("action_schema")]
    if schemas:
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError("multi-seed run produced inconsistent action schemas")
        aggregated["action_schema"] = schemas[0]
    if obs_dims:
        aggregated["observation_account_dim"] = obs_dims[0]
    observation_shapes = [
        r.get("observation_space_shapes")
        for r in seed_results
        if r.get("observation_space_shapes") is not None
    ]
    if observation_shapes:
        aggregated["observation_space_shapes"] = observation_shapes[0]
    observation_schemas = [
        result.get("observation_schema")
        for result in seed_results
        if result.get("observation_schema") is not None
    ]
    if observation_schemas:
        if any(schema != observation_schemas[0] for schema in observation_schemas[1:]):
            raise ValueError("multi-seed run produced inconsistent observation schemas")
        aggregated["observation_schema"] = observation_schemas[0]

    return aggregated


def main() -> None:
    args = parse_train_args()

    if args.use_vae and not args.vae_path:
        raise SystemExit("--use-vae requires --vae-path pointing at a trained VAE .pth")
    if args.walk_forward and args.use_vae:
        raise SystemExit(
            "Walk-forward with an externally fitted VAE has unverified fold provenance; "
            "provide a fold-scoped VAE fitting pipeline before enabling this combination"
        )

    _setup_rngs(args.seed)

    cfg = _load_merged_config(args)
    try:
        _validate_q5_resume_state(cfg, resume_checkpoint=args.resume_checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.arch is None:
        args.arch = str(cfg.agent.get("arch", "tcn"))
    elif getattr(args, "quality_variant", None) and args.arch != str(cfg.agent.get("arch", "tcn")):
        raise SystemExit(
            f"--arch={args.arch} conflicts with the selected quality variant encoder "
            f"{cfg.agent.get('arch', 'tcn')}"
        )

    device = get_device(cfg.get("device"))
    requested_n_envs = int(cfg.env.n_envs)
    quality_variant = str(cfg.get("quality_variant_id", ""))
    exact_n_envs = False
    preserve_ppo_geometry = False
    if quality_variant:
        from quant_rl.train.quality_registry import quality_variant_resource_contract

        resource_contract = quality_variant_resource_contract(quality_variant)
        exact_n_envs = bool(resource_contract.get("exact_n_envs", False))
        preserve_ppo_geometry = exact_n_envs and "aggregate_rollout_steps" in resource_contract
    cfg.quality_execution.requested_n_envs = requested_n_envs
    cfg.quality_execution.exact_n_envs = exact_n_envs
    scale_training_cfg(
        cfg,
        device,
        args.arch,
        preserve_ppo_geometry=preserve_ppo_geometry,
    )
    cfg.quality_execution.effective_n_envs = int(cfg.env.n_envs)
    if quality_variant and exact_n_envs:
        from quant_rl.train.quality_registry import validate_quality_resource_shape

        try:
            validate_quality_resource_shape(
                quality_variant,
                requested_n_envs=requested_n_envs,
                effective_n_envs=int(cfg.env.n_envs),
                aggregate_rollout_steps=int(cfg.env.n_envs * cfg.ppo.n_steps),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
    lagrangian_cb = _build_lagrangian_callback(cfg, algo=args.algo, mvp=args.mvp)
    if lagrangian_cb is not None:
        log.info(
            "Q5 primal-dual PPO enabled: episode_cost_budget=%.4f dual_lr=%.4f lambda_max=%.3f",
            lagrangian_cb.episode_cost_budget,
            lagrangian_cb.dual_learning_rate,
            lagrangian_cb.lambda_max,
        )

    data = run_pipeline(cfg, force=args.force)
    primary_ticks = build_tick_books(cfg).get(str(cfg.data.primary))

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
    randomized_fill_latency_ms = _randomized_fill_latency_ms(cfg)
    tick_execution_required = _tick_execution_required(cfg)
    if tick_execution_required or randomized_fill_latency_ms > 0.0:
        execution_coverage = _tick_execution_split_coverage(
            primary_ticks,
            cast(pd.DatetimeIndex, primary_m1.index),
            train_mask,
            test_start,
            session_labels=(primary_m1["session"] if "session" in primary_m1.columns else None),
            max_latency_ns_exclusive=int(randomized_fill_latency_ms * 1_000_000),
            max_quote_wait_ms=_max_tick_quote_wait_ms(cfg),
        )
        for split_name, report in execution_coverage.items():
            log.info(
                "Tick execution coverage [%s]: covered=%d/%d (%.4f%%)",
                split_name,
                report["covered_count"],
                report["order_count"],
                100.0 * report["coverage_fraction"],
            )
            if report["unavailable_count"]:
                raise SystemExit(
                    "Tick-only execution preflight failed for "
                    f"{split_name}: {report['unavailable_count']}/"
                    f"{report['order_count']} candidate order times lack a quote "
                    "that is available for every allowed latency draw within "
                    f"{_max_tick_quote_wait_ms(cfg)} ms; first unavailable order "
                    f"time={report['first_unavailable_order_time']}. Training not started."
                )
    prebuilt_cache_value = OmegaConf.select(cfg, "data.prebuilt_feature_cache")
    if prebuilt_cache_value:
        feat_cache = Path(str(prebuilt_cache_value)).expanduser()
        if not feat_cache.is_absolute():
            feat_cache = Path(__file__).resolve().parents[2] / feat_cache
        feat_cache = feat_cache.resolve()

        hash_path = Path(str(feat_cache) + ".hash")
        expected_cache_hash = feature_cache_content_hash(
            cfg,
            primary_m1,
            train_mask=train_mask,
            secondary=secondary_m1,
        )
        if not feat_cache.is_file() or not hash_path.is_file():
            raise FileNotFoundError(
                f"Configured prebuilt feature cache or its .hash sidecar is missing: {feat_cache}"
            )
        actual_cache_hash = hash_path.read_text(encoding="utf-8").strip()
        if actual_cache_hash != expected_cache_hash:
            raise ValueError(
                "Configured prebuilt feature cache does not match the effective "
                f"data/config hash: {feat_cache} (expected {expected_cache_hash}, "
                f"found {actual_cache_hash})"
            )
        reference_mode = str(OmegaConf.select(cfg, "features.po3_reference_mode") or "m1_legacy")
        sweep_audit_path = Path(str(feat_cache) + ".htf_sweeps.parquet")
        if reference_mode == "htf_confirmed" and not sweep_audit_path.is_file():
            raise FileNotFoundError(
                "Configured prebuilt feature cache is missing its HTF sweep audit: "
                f"{sweep_audit_path}"
            )
        log.info("Using hash-validated prebuilt feature cache: %s", feat_cache)
    else:
        feat_cache = feature_cache_path(
            cache_dir,
            primary_sym,
            cfg,
            primary_m1,
            train_mask=train_mask,
            secondary=secondary_m1,
        )
    features = build_features(
        primary_m1,
        secondary=secondary_m1,
        cfg=cfg,
        train_mask=train_mask,
        cache_path=feat_cache,
    )
    feature_columns = [str(column) for column in features.columns]

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

    # Fingerprint the exact split passed to the environments before bundle mode
    # releases the frames. These records are reused for every seed in this run.
    dataset_records = {
        "train_bars": dataset_record(train_bars),
        "test_bars": dataset_record(test_bars),
        "train_features": dataset_record(train_feat),
        "test_features": dataset_record(test_feat),
    }
    if train_sec is not None:
        dataset_records["train_secondary"] = dataset_record(train_sec)
    if test_sec is not None:
        dataset_records["test_secondary"] = dataset_record(test_sec)
    if primary_ticks is not None:
        tick_digest = hashlib.sha256(
            "".join(
                array_sha256(array)
                for array in (primary_ticks._ts, primary_ticks._bid, primary_ticks._ask)
            ).encode()
        ).hexdigest()
        dataset_records["primary_ticks"] = {
            "rows": int(len(primary_ticks._ts)),
            "columns": ["timestamp_ns", "bid", "ask"],
            "start": (
                str(pd.to_datetime(primary_ticks._ts[0], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "end": (
                str(pd.to_datetime(primary_ticks._ts[-1], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "sha256": tick_digest,
        }

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
    seeds = list(args.seeds) if args.seeds else [args.seed]
    if args.resume_checkpoint and (args.algo != "ppo" or len(seeds) != 1):
        raise SystemExit("--resume-checkpoint currently supports one PPO seed per run")
    is_multi_seed = len(seeds) > 1
    quality_variant = str(cfg.get("quality_variant_id", ""))
    if quality_variant:
        run_name = (
            f"quality_{quality_variant}_seeds_{args.arch}"
            if is_multi_seed
            else f"quality_{quality_variant}_seed{args.seed}_{args.arch}"
        )
    elif is_multi_seed:
        run_name = f"rl_train_seeds_{args.arch}"
    else:
        run_name = f"rl_train_seed{args.seed}_{args.arch}"
    run_dir = build_run_dir(args.out, run_name)
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Store the resolved config before any model work so interrupted trials
    # still retain the exact invocation settings beside partial artifacts.
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    data_source = str(cfg.env.get("data_source", "frames"))
    obs_mmap: str | None
    bundle_path: Path | None = None
    test_bundle_path: Path | None = None
    train_env_spec: Any | None = None
    train_eval_spec: Any | None = None
    test_eval_spec: Any | None = None
    train_bar_count = len(train_bars)
    test_bar_count = len(test_bars)
    best_bars: pd.DataFrame | None = None
    best_features: pd.DataFrame | None = None
    dashboard_bars: pd.DataFrame | None = None
    dashboard_features: pd.DataFrame | None = None
    if data_source == "bundle":
        from quant_rl.envs.env_bundle_builder import build_env_bundle
        from quant_rl.envs.env_spec import EnvSpec

        if args.use_vae:
            raise ValueError(
                "env.data_source=bundle with --use-vae requires the VAE latent-table phase"
            )

        bundle_path = build_env_bundle(
            cache_dir,
            train_bars,
            train_feat,
            cfg,
            raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
            feature_config_hash=feat_cache.stem,
            tickbook=ticks_covering(primary_ticks, train_bars),
            pre_ny_by_date=pre_ny_by_date,
        )
        obs_mmap = str(bundle_path / "obs_features.npy")
        train_env_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(bundle_path),
            episodic=_training_episodic(cfg),
        )
        train_eval_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(bundle_path),
            episodic=False,
            evaluation=True,
        )
        test_bundle_path = build_env_bundle(
            cache_dir,
            test_bars,
            test_feat,
            cfg,
            raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
            feature_config_hash=feat_cache.stem,
            tickbook=ticks_covering(primary_ticks, test_bars),
            pre_ny_by_date=None,
        )
        test_eval_spec = EnvSpec.from_cfg(
            cfg,
            algo=args.algo,
            reward=args.reward,
            arch=args.arch,
            bundle_dir=str(test_bundle_path),
            episodic=False,
            evaluation=True,
        )
        best_count = min(20_000, train_bar_count)
        # .copy() detaches these from the full train arrays so that the
        # `del train_bars, train_feat` below actually frees the base (a bare
        # .iloc slice would keep the whole 351k-row frame alive).
        best_bars = train_bars.iloc[:best_count].copy()
        best_features = train_feat.iloc[:best_count].copy()
        dash_cfg = _dashboard_block(cfg)
        dashboard_count = min(int(dash_cfg.get("eval_bars", 20_000)), train_bar_count)
        dashboard_bars = train_bars.iloc[:dashboard_count].copy()
        dashboard_features = train_feat.iloc[:dashboard_count].copy()
        log.info("Built shared environment bundle: %s", bundle_path)
    elif data_source == "frames":
        obs_mmap = _publish_obs_memmap(train_feat, train_bars, cfg, run_dir / "obs_features.npy")
    else:
        raise ValueError(f"env.data_source must be 'frames' or 'bundle', got {data_source!r}")

    # Release full-history feature pipeline objects after the train/test split is ready.
    del data, features, primary_m1, secondary_m1
    import gc

    gc.collect()
    log.info("Parent memory cleanup: released full-history pipeline objects")
    if data_source == "bundle" and not args.walk_forward:
        del train_bars, train_feat, test_bars, test_feat
        gc.collect()
        log.info("Released train/test frames; bundle workers now map the shared arrays")

    # Seed loop: train + eval for each seed, collect results
    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        _setup_rngs(seed)
        phase_timings_seconds: dict[str, float] = {}
        log.info("=== Seed %d / %d ===", seed, len(seeds))

        # Per-seed output dirs
        if is_multi_seed:
            seed_dir = run_dir / "seeds" / f"seed{seed}"
            seed_model_dir = seed_dir / "model"
            seed_model_dir.mkdir(parents=True, exist_ok=True)
        else:
            seed_dir = run_dir
            seed_model_dir = model_dir

        # Create training environment
        log.info("Creating training environment...")
        if train_env_spec is not None:
            from quant_rl.envs.worker import make_bundle_env

            train_env = make_bundle_env(train_env_spec)
            env_factory = partial(make_bundle_env, train_env_spec)
        else:
            train_env = make_env(
                train_bars,
                train_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                episodic=_training_episodic(cfg),
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                obs_features_mmap=obs_mmap,
                tickbook=ticks_covering(primary_ticks, train_bars),
                arch=args.arch,
            )
            env_factory = partial(
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
                tickbook=ticks_covering(primary_ticks, train_bars),
                arch=args.arch,
            )

        checkpoint_callback = _periodic_checkpoint_callback(cfg, seed_model_dir)

        # Train agent
        timesteps = cfg.ppo.total_timesteps if not args.mvp else cfg.training.total_timesteps_mvp
        log.info("Training %s for %d timesteps...", args.algo.upper(), timesteps)

        if args.resume_checkpoint:
            from stable_baselines3 import PPO

            from quant_rl.models.agent import _build_vector_env, validate_model_architecture

            resume_path = Path(args.resume_checkpoint)
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            resume_env = _build_vector_env(train_env, cfg, env_factory)
            model = PPO.load(resume_path, env=resume_env, device=device)
            validate_model_architecture(model, args.arch, use_vae=args.use_vae)
            from quant_rl.models.agent import validate_model_observation_schema

            validate_model_observation_schema(
                model,
                train_env.observation_schema,
                allow_legacy_v1=bool(args.allow_legacy_observation_schema),
            )
            if int(model.num_timesteps) > int(timesteps):
                raise ValueError(
                    f"resume checkpoint has {model.num_timesteps} steps, exceeding requested {timesteps}"
                )
            learning_timesteps = int(timesteps) - int(model.num_timesteps)
            log.info(
                "Resuming PPO from %s at %d/%d steps; %d remain",
                resume_path,
                model.num_timesteps,
                timesteps,
                learning_timesteps,
            )
        else:
            model = build_agent(
                train_env,
                cfg,
                arch=args.arch,
                algo=args.algo,
                device=device,
                use_vae=args.use_vae,
                vae=vae_model,
                env_fn=env_factory,
                seed=int(seed),
            )
            learning_timesteps = int(timesteps)
        stop_candidate_schema = build_stop_candidate_schema(cfg)
        checkpoint_stop_schema = getattr(model, "stop_candidate_schema", None)
        if args.resume_checkpoint and checkpoint_stop_schema != stop_candidate_schema:
            raise ValueError(
                "resume checkpoint stop-candidate schema is missing or differs from the current "
                "run config; do not resume under a silently changed stop menu"
            )
        setattr(model, "stop_candidate_schema", stop_candidate_schema)
        _log_effective_ppo(model, cfg)

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
            | LearningDiagnosticsCallback
            | LagrangianRiskCallback
            | ClipLogStdCallback
            | EpisodeEquityCallback
            | EarlyAbortCallback
            | TrainingDashboardCallback
        ] = [c for c in (lagrangian_cb, checkpoint_callback, aux_cb) if c is not None]

        callbacks.append(EpisodeEquityCallback(log_path=seed_model_dir / "episode_equity.csv"))
        if bool(cfg.get("quality_diagnostics", {}).get("enabled", False)):
            callbacks.append(
                LearningDiagnosticsCallback(log_path=seed_model_dir / "learning_diagnostics.csv")
            )
        abort_cb = _early_abort_callback(cfg)
        if abort_cb is not None:
            callbacks.append(abort_cb)

        # Best checkpoint: one short prefix of the train calendar, scored by
        # end equity. Full replay risk checks run after training; PnL and trade
        # counts are reported as metrics and do not stop this callback.
        short_n = min(20_000, train_bar_count)
        short_bars = best_bars if best_bars is not None else train_bars.iloc[:short_n]
        short_feat = best_features if best_features is not None else train_feat.iloc[:short_n]
        if bundle_path is not None and train_env_spec is not None:
            from quant_rl.envs.env_bundle_builder import build_env_bundle
            from quant_rl.envs.env_spec import EnvSpec
            from quant_rl.envs.worker import make_bundle_env

            short_bundle = build_env_bundle(
                cache_dir,
                short_bars,
                short_feat,
                cfg,
                raw_columns=list(_strategy_from_cfg(cfg)[0].raw_columns),
                feature_config_hash=feat_cache.stem,
                tickbook=ticks_covering(primary_ticks, short_bars),
                pre_ny_by_date=None,
            )
            best_eval_spec = EnvSpec.from_cfg(
                cfg,
                algo=args.algo,
                reward=args.reward,
                arch=args.arch,
                bundle_dir=str(short_bundle),
                episodic=False,
                evaluation=True,
            )
            best_eval_factory = partial(make_bundle_env, best_eval_spec)
        else:
            best_eval_factory = partial(
                make_env,
                short_bars,
                short_feat,
                cfg,
                algo=args.algo,
                reward=args.reward,
                episodic=False,
                use_vae=args.use_vae,
                vae=env_vae,
                pre_ny_by_date=pre_ny_by_date,
                tickbook=ticks_covering(primary_ticks, short_bars),
                arch=args.arch,
            )

        best_eval_freq = max(1, timesteps)  # once at the end
        best_cb = BestCheckpointEvalCallback(
            eval_env_factory=best_eval_factory,
            eval_freq=best_eval_freq,
            best_model_path=seed_model_dir / "ppo_best",
        )
        callbacks.append(best_cb)

        progress_log = seed_model_dir / "training_log.csv"
        callbacks.append(ProgressLoggerCallback(log_path=progress_log))
        if isinstance(train_env.action_space, spaces.Box):
            callbacks.append(
                ClipLogStdCallback(
                    log_std_min=float(cfg.ppo.get("log_std_min", -0.7)),
                    log_std_max=float(cfg.ppo.get("log_std_max", 0.0)),
                )
            )

        eval_common, _strategy, _strategy_reward, _strategy_weight, strategy_actions = (
            _build_eval_common(cfg, args, env_vae, pre_ny_by_date, arch=args.arch)
        )
        dash = _dashboard_block(cfg)
        if bool(dash.get("enabled", True)):
            eval_bars = int(dash.get("eval_bars", 20_000))
            slice_n = min(eval_bars, train_bar_count)
            dash_bars = dashboard_bars if dashboard_bars is not None else train_bars.iloc[:slice_n]
            dash_feat = (
                dashboard_features if dashboard_features is not None else train_feat.iloc[:slice_n]
            )

            def _dashboard_eval() -> dict[str, Any]:
                result = evaluate_model(
                    model,
                    bars=dash_bars,
                    features=dash_feat,
                    max_episode_steps=None,
                    deterministic=False,
                    tickbook=ticks_covering(primary_ticks, dash_bars),
                    **eval_common,
                )
                metrics = calculate_metrics(
                    result["equity"],
                    trades=result["trades"],
                    n_sessions=result.get("n_sessions", 1),
                    n_breach_sessions=result.get("n_breach_sessions", 0),
                    periods_per_year=evaluation_periods_per_year(cfg),
                )
                n_steps = max(1, int(result.get("n_steps", 1)))
                from quant_rl.eval.decision_diversity import (
                    format_diversity_text,
                    thresholds_from_mapping,
                )

                diversity = format_diversity_text(
                    result["trades"],
                    thresholds_from_mapping(dash.get("diversity")),
                )
                return {
                    "evaluated_bars": int(len(dash_bars)),
                    "evaluation_split": "training_prefix_monitor",
                    "return_pct": float(metrics.total_return_pct),
                    "total_pnl": float(metrics.total_pnl),
                    "account_currency": str(cfg.account.currency),
                    "sharpe": float(metrics.sharpe),
                    "sortino": float(metrics.sortino),
                    "max_drawdown": float(metrics.max_drawdown),
                    "profit_factor": float(metrics.profit_factor),
                    "win_rate": float(metrics.win_rate),
                    "avg_trade": float(metrics.avg_trade),
                    "n_trades": float(metrics.n_trades),
                    "turnover": float(metrics.turnover),
                    "reward_mean": float(result.get("reward_sum", 0.0)) / n_steps,
                    "diversity_text": diversity,
                }

            callbacks.append(
                TrainingDashboardCallback(
                    total_timesteps=timesteps,
                    eval_every=int(dash.get("eval_every", 100_000)),
                    eval_fn=_dashboard_eval,
                    log_path=seed_model_dir / "dashboard_log.csv",
                    log_every=int(dash.get("log_every", 100_000)),
                )
            )

        log_every = int(dash.get("log_every", 100_000))
        log_interval = ppo_log_interval(
            n_steps=int(model.n_steps),
            n_envs=int(model.n_envs),
            log_every=log_every,
        )
        log.info("SB3 log every %d rollouts (~%d env steps)", log_interval, log_every)
        if isinstance(train_env.action_space, spaces.Box):
            install_post_update_std_clip(
                model,
                float(cfg.ppo.get("log_std_min", -0.7)),
                float(cfg.ppo.get("log_std_max", 0.0)),
            )
        debug_cfg = cfg.training.get("debug", {})
        if bool(debug_cfg.get("enabled", False)):
            install_flight_recorder(model, seed_dir, thresholds=debug_cfg)
        _initialize_value_head(model, resumed=bool(args.resume_checkpoint))
        phase_started = time.perf_counter()
        if learning_timesteps > 0:
            model.learn(
                total_timesteps=learning_timesteps,
                callback=cast(Any, callbacks),
                log_interval=log_interval,
                reset_num_timesteps=not bool(args.resume_checkpoint),
            )
        elif args.resume_checkpoint:
            best_cb.model = model
            best_cb._run_eval()
        phase_timings_seconds["ppo_learn"] = time.perf_counter() - phase_started
        log.info("PHASE_TIMING ppo_learn_seconds=%.3f", phase_timings_seconds["ppo_learn"])
        abort_reason = abort_cb.reason if abort_cb is not None else None
        if abort_reason:
            (seed_dir / "early_abort.json").write_text(
                json.dumps(
                    {"reason": abort_reason, "timesteps": int(model.num_timesteps)},
                    indent=2,
                )
            )
            log.error("EARLY_ABORT %s at timesteps=%s", abort_reason, model.num_timesteps)

        # Save final model
        model_path = seed_model_dir / "ppo_final"
        save_ppo_checkpoint(model, model_path)
        log.info("Model saved: %s", model_path)

        # Learning-curve / loss charts from the SB3 progress CSV.
        try:
            from quant_rl.eval.training_plots import save_dashboard_plots, save_training_plots

            save_training_plots(
                progress_log,
                out_dir=seed_model_dir,
                dpi=getattr(cfg.output, "dpi", 150),
                save_html=getattr(cfg.output, "save_html", False),
            )
            save_dashboard_plots(
                seed_model_dir / "dashboard_log.csv",
                out_dir=seed_model_dir,
                dpi=getattr(cfg.output, "dpi", 150),
                save_html=getattr(cfg.output, "save_html", False),
            )
        except Exception as exc:
            log.warning("Training progress plots skipped: %s", exc)

        # Final report-only held-out evaluation uses deterministic policy actions.
        phase_started = time.perf_counter()
        log.info("Evaluating trained model on test set...")
        if test_eval_spec is not None:
            from quant_rl.envs.worker import make_bundle_env

            test_eval_env = make_bundle_env(test_eval_spec)
            empty_bars = pd.DataFrame()
            empty_features = pd.DataFrame()
            test_ticks = None
        else:
            test_eval_env = None
            empty_bars = test_bars
            empty_features = test_feat
            test_ticks = ticks_covering(primary_ticks, test_bars)
        test_result = _evaluate_final_split(
            model,
            bars=empty_bars,
            features=empty_features,
            max_episode_steps=None,
            tickbook=test_ticks,
            env=test_eval_env,
            **eval_common,
        )
        test_result["initial_balance"] = cfg.account.initial_balance
        test_m = calculate_metrics(
            test_result["equity"],
            trades=test_result["trades"],
            n_sessions=test_result.get("n_sessions", 1),
            n_breach_sessions=test_result.get("n_breach_sessions", 0),
            periods_per_year=evaluation_periods_per_year(cfg),
        )
        test_dir_sum = _direction_summary(test_result["trades"])
        log.info(
            "[test/report-only] NetPnL=%.2f %s  Return=%.2f%%  "
            "MaxDD=%.2f%%  Sharpe=%.3f  ProfitFactor=%.3f  WinRate=%.1f%%  Trades=%d  "
            "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
            test_m.total_pnl,
            str(cfg.account.currency),
            test_m.total_return_pct,
            test_m.max_drawdown * 100,
            test_m.sharpe,
            test_m.profit_factor,
            test_m.win_rate * 100,
            test_m.n_trades,
            test_result.get("survived_full_year"),
            test_result.get("days_traded", 0),
            test_result.get("n_sessions", 0),
            test_result.get("fail_time"),
            test_m.max_drawdown * 100,
            test_dir_sum,
            test_result.get("entry_diag", {}),
        )
        phase_timings_seconds["test_evaluation"] = time.perf_counter() - phase_started
        log.info(
            "PHASE_TIMING test_evaluation_seconds=%.3f",
            phase_timings_seconds["test_evaluation"],
        )

        # In-sample evaluation on the training split so the run dir carries the same
        # training/ + testing/ artifact layout as the baseline runners.
        phase_started = time.perf_counter()
        log.info("Evaluating trained model on training set (in-sample)...")
        if train_eval_spec is not None:
            from quant_rl.envs.worker import make_bundle_env

            train_eval_env = make_bundle_env(train_eval_spec)
            empty_bars = pd.DataFrame()
            empty_features = pd.DataFrame()
            train_ticks = None
        else:
            train_eval_env = None
            empty_bars = train_bars
            empty_features = train_feat
            train_ticks = ticks_covering(primary_ticks, train_bars)
        train_result = _evaluate_final_split(
            model,
            bars=empty_bars,
            features=empty_features,
            max_episode_steps=None,
            tickbook=train_ticks,
            env=train_eval_env,
            **eval_common,
        )
        train_result["initial_balance"] = cfg.account.initial_balance
        train_m = calculate_metrics(
            train_result["equity"],
            trades=train_result["trades"],
            n_sessions=train_result.get("n_sessions", 1),
            n_breach_sessions=train_result.get("n_breach_sessions", 0),
            periods_per_year=evaluation_periods_per_year(cfg),
        )
        train_dir_sum = _direction_summary(train_result["trades"])
        train_gate = assess_train_equity(
            train_result["equity"],
            initial=float(cfg.account.initial_balance),
            breached=not bool(train_result.get("survived_full_year", False)),
        )
        (seed_dir / "train_equity_gate.json").write_text(json.dumps(train_gate, indent=2))
        if not train_gate["ok"]:
            log.error("TRAIN_SURVIVAL_FAILED %s", train_gate)
        log.info(
            "[train/in-sample] NetPnL=%.2f %s  Return=%.2f%%  "
            "MaxDD=%.2f%%  Sharpe=%.3f  ProfitFactor=%.3f  WinRate=%.1f%%  Trades=%d  "
            "survived=%s  days=%d/%d  fail_time=%s  max_trailing_dd=%.2f%%  direction=%s  entry_diag=%s",
            train_m.total_pnl,
            str(cfg.account.currency),
            train_m.total_return_pct,
            train_m.max_drawdown * 100,
            train_m.sharpe,
            train_m.profit_factor,
            train_m.win_rate * 100,
            train_m.n_trades,
            train_result.get("survived_full_year"),
            train_result.get("days_traded", 0),
            train_result.get("n_sessions", 0),
            train_result.get("fail_time"),
            train_m.max_drawdown * 100,
            train_dir_sum,
            train_result.get("entry_diag", {}),
        )
        phase_timings_seconds["train_evaluation"] = time.perf_counter() - phase_started
        log.info(
            "PHASE_TIMING train_evaluation_seconds=%.3f",
            phase_timings_seconds["train_evaluation"],
        )
        _write_evaluation_artifacts(
            run_dir=seed_dir,
            seed=int(seed),
            completed_timesteps=int(model.num_timesteps),
            train_result=train_result,
            test_result=test_result,
            train_metrics=train_m,
            test_metrics=test_m,
            train_gate=train_gate,
            account_currency=str(cfg.account.currency),
        )
        log.info(
            "Final evaluation report written: %s",
            seed_dir / "evaluations" / "summary.md",
        )

        # Export both splits for this seed
        phase_started = time.perf_counter()
        if bundle_path is not None:
            from quant_rl.envs.env_bundle_builder import bars_for_export

            train_export_bars = bars_for_export(bundle_path)
            test_export_bars = bars_for_export(test_bundle_path) if test_bundle_path else None
        else:
            train_export_bars = train_bars
            test_export_bars = test_bars
        save_run(
            run_dir=seed_dir,
            train_result=train_result,
            train_metrics=train_m,
            train_bars=train_export_bars,
            train_secondary=train_sec,
            test_result=test_result,
            test_metrics=test_m,
            test_bars=test_export_bars,
            test_secondary=test_sec,
            cfg=cfg,
            save_plots=getattr(cfg.output, "save_plots", True),
            save_html=getattr(cfg.output, "save_html", False),
            save_csv=getattr(cfg.output, "save_csv", True),
            dpi=getattr(cfg.output, "dpi", 150),
            max_order_charts=int(getattr(cfg.output, "max_order_charts", 0)),
            report_profile=str(getattr(cfg.output, "report_profile", "full")),
        )
        if bundle_path is not None:
            del train_export_bars, test_export_bars

        # Save config snapshot for this seed
        (seed_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        per_run_cache_hash = Path(str(feat_cache) + ".hash")
        cache_hash = per_run_cache_hash.read_text(encoding="utf-8").strip()
        checkpoint_path = Path(f"{model_path}.zip")
        if not checkpoint_path.is_file():
            checkpoint_path = model_path
        sequence_feature_schema = _sequence_feature_schema_manifest(train_env)
        manifest = build_training_run_manifest(
            repository=Path(__file__).resolve().parents[2],
            config_path=seed_dir / "config.yaml",
            datasets={},
            dataset_records=dataset_records,
            split={"train_end": str(train_end), "test_start": str(test_start)},
            seed=int(seed),
            completed_steps=int(model.num_timesteps),
            action_schema=_action_schema_manifest(train_env),
            observation_schema=_observation_schema_manifest(train_env),
            sequence_feature_schema=sequence_feature_schema,
            stop_candidate_schema=stop_candidate_schema,
            feature_cache_path=feat_cache,
            feature_cache_hash=cache_hash,
            checkpoint_path=checkpoint_path,
            execution_protocol={
                "data_source": data_source,
                "costs": OmegaConf.to_container(cfg.costs, resolve=True) if "costs" in cfg else {},
                "execution": OmegaConf.to_container(cfg.execution, resolve=True)
                if "execution" in cfg
                else {},
                "fill_delay_ms": _fill_delay_ms(cfg),
                "block_overnight": bool(cfg.env.get("block_overnight", True)),
                "tick_data_configured": bool(primary_ticks is not None),
                "primary_symbol": str(primary_sym),
            },
            training_architecture=_training_architecture_provenance(model, cfg, args.arch),
            training_protocol=_training_protocol_manifest(
                model=model,
                cfg=cfg,
                args=args,
                seed=int(seed),
                requested_steps=int(timesteps),
            ),
            metrics_protocol={
                "annualization": {
                    "periods_per_year": evaluation_periods_per_year(cfg),
                    "convention": str(cfg.evaluation.annualization_convention),
                },
                "initial_balance": float(cfg.account.initial_balance),
                "execution_seed": int(seed),
                "historical_test_role": "previously_inspected_holdout",
                "frequency": "bar_close_equity",
                "exposure_measure": BAR_CLOSE_EXPOSURE_MEASURE,
            },
            artifact_paths={
                "checkpoint": str(checkpoint_path.relative_to(seed_dir)),
                "training_log": str((seed_dir / "training_log.csv").relative_to(seed_dir)),
                "trade_log": str((seed_dir / "trade_log.json").relative_to(seed_dir)),
            },
        )
        if not manifest["source"]["available"] or not manifest["source"]["head"]:
            raise RuntimeError("training provenance requires a readable Git source identity")
        manifest["feature_cache"]["file_sha256"] = (
            file_sha256(feat_cache) if feat_cache.is_file() else None
        )
        manifest_path = seed_dir / "run_manifest.json"
        write_run_manifest(manifest_path, manifest)
        phase_timings_seconds["report_export"] = time.perf_counter() - phase_started
        log.info("PHASE_TIMING report_export_seconds=%.3f", phase_timings_seconds["report_export"])

        # Per-seed artifacts (multi-seed mode only)
        if is_multi_seed:
            _save_seed_artifacts(seed_dir, seed, cfg, train_result, test_result, train_m, test_m)

        # Collect per-seed results for aggregation
        entry_diag = test_result.get("entry_diag", {})
        entry_state_diag = test_result.get("entry_state_diagnostics", {})
        action_space_width = _action_space_width(train_env)
        action_schema = _action_schema_manifest(train_env)
        observation_account_dim = _observation_account_dim(train_env)
        observation_space_shapes = _observation_space_shapes(train_env)
        observation_schema = _observation_schema_manifest(train_env)
        env_flags = {
            key: bool(cfg.env.get(key, False))
            for key in (
                "strategy_actions",
                "allow_agent_sl_mode",
                "allow_agent_tp_mode",
                "allow_multi_tp",
                "allow_simplex",
                "entry_state_machine",
                "entry_state_observation",
                "arm_requires_retest",
            )
        }
        oos_role = (
            "selection_acknowledged"
            if getattr(args, "allow_locked_oos_for_selection", False)
            else "final_report_only"
        )
        seed_results.append(
            {
                "seed": seed,
                "train_m": train_m,
                "test_m": test_m,
                "train_result": train_result,
                "test_result": test_result,
                "train_gate": train_gate,
                "abort_reason": abort_reason,
                "model_num_timesteps": int(model.num_timesteps),
                "entry_diag": entry_diag,
                "entry_state_diagnostics": entry_state_diag,
                "action_space_width": action_space_width,
                "action_schema": action_schema,
                "observation_account_dim": observation_account_dim,
                "observation_space_shapes": observation_space_shapes,
                "observation_schema": observation_schema,
                "sequence_feature_schema": sequence_feature_schema,
                "env_flags": env_flags,
                "oos_role": oos_role,
                "phase_timings_seconds": phase_timings_seconds,
                "run_manifest": str(manifest_path.relative_to(run_dir)),
            }
        )

    # Aggregate results across seeds and write training_log.json
    strategy_actions = bool(cfg.env.get("strategy_actions", False))
    strategy_name = (
        "baseline" if not strategy_actions else str(cfg.strategy.get("name", args.strategy))
    )

    if is_multi_seed:
        training_log = _aggregate_seed_results(
            seed_results=seed_results,
            seed=args.seed,
            mvp=args.mvp,
            algo=args.algo,
            arch=args.arch,
            reward=args.reward,
            timesteps=timesteps,
            train_bars=train_bar_count,
            test_bars=test_bar_count,
            strategy=strategy_name,
            strategy_actions=strategy_actions,
            cfg=cfg,
            args=args,
            run_dir=run_dir,
        )
    else:
        # Single-seed: preserve current behavior exactly
        single = seed_results[0]
        training_log = _build_training_log(
            seed=single["seed"],
            mvp=args.mvp,
            algo=args.algo,
            arch=args.arch,
            reward=args.reward,
            timesteps=timesteps,
            train_bars=train_bar_count,
            test_bars=test_bar_count,
            test_m=single["test_m"],
            test_result=single["test_result"],
            strategy=strategy_name,
            strategy_actions=strategy_actions,
            entry_diag=single.get("entry_diag"),
            entry_state_diagnostics=single.get("entry_state_diagnostics"),
            action_space_width=single.get("action_space_width"),
            observation_account_dim=single.get("observation_account_dim"),
            observation_space_shapes=single.get("observation_space_shapes"),
            action_schema=single.get("action_schema"),
            observation_schema=single.get("observation_schema"),
            env_flags=single.get("env_flags"),
            oos_role=single.get("oos_role", "final_report_only"),
        )
        training_log["train_equity_ok"] = bool(single["train_gate"]["ok"])
        training_log["train_end_equity"] = float(single["train_gate"]["end_equity"])
        training_log["train_equity_slope"] = float(single["train_gate"]["slope"])
        training_log["train_equity_reason"] = str(single["train_gate"]["reason"])
        training_log["early_abort_reason"] = single["abort_reason"] or ""
        training_log["timesteps_completed"] = single["model_num_timesteps"]

    sequence_schemas = [
        result["sequence_feature_schema"]
        for result in seed_results
        if result.get("sequence_feature_schema") is not None
    ]
    if sequence_schemas:
        if any(schema != sequence_schemas[0] for schema in sequence_schemas[1:]):
            raise ValueError("multi-seed run produced inconsistent sequence feature schemas")
        training_log["sequence_feature_schema"] = sequence_schemas[0]

    training_log["phase_timings_seconds"] = {
        str(result["seed"]): result["phase_timings_seconds"] for result in seed_results
    }
    training_log["run_manifests"] = [str(result["run_manifest"]) for result in seed_results]
    cache_hash_path = Path(str(feat_cache) + ".hash")
    if not cache_hash_path.is_file():
        raise RuntimeError(f"feature cache hash sidecar missing: {cache_hash_path}")
    training_log["feature_cache_hash"] = cache_hash_path.read_text(encoding="utf-8").strip()
    training_log["feature_cache_name"] = feat_cache.name
    effective_config_path = run_dir / "config.yaml"
    if not effective_config_path.is_file():
        raise RuntimeError(f"effective config snapshot missing: {effective_config_path}")
    training_log["effective_config_sha256"] = hashlib.sha256(
        effective_config_path.read_bytes()
    ).hexdigest()
    training_log["feature_columns"] = feature_columns
    training_log["feature_schema_sha256"] = hashlib.sha256(
        json.dumps(training_log["feature_columns"], separators=(",", ":")).encode()
    ).hexdigest()
    (run_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))
    if not seed_results[-1]["train_gate"]["ok"]:
        raise SystemExit(
            f"TRAIN_SURVIVAL_FAILED: train-year risk gate failed ({seed_results[-1]['train_gate']['reason']})"
        )

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
                "n_seeds": len(seeds),
            },
        )
        if args.allow_locked_oos_for_selection:
            log_payload = {
                "sharpe": training_log.get(
                    "test_sharpe_mean", training_log.get("test_sharpe", 0.0)
                ),
                "test_sharpe": training_log.get(
                    "test_sharpe_mean", training_log.get("test_sharpe", 0.0)
                ),
                "test_max_dd": training_log.get(
                    "test_max_dd_mean", training_log.get("test_max_dd", 0.0)
                ),
                "test_trades": training_log.get(
                    "test_n_trades_mean", training_log.get("test_trades", 0)
                ),
                "test_return": training_log.get(
                    "test_total_return_pct_mean", training_log.get("test_return", 0.0)
                ),
                "test_breaches": training_log.get("test_breaches", 0),
            }
            wandb.log(log_payload)
        wandb.finish()

    # Walk-forward validation (additive, does not alter the single-split flow)
    if args.walk_forward:
        log.info("Running walk-forward validation with %d splits...", args.wf_splits)
        from quant_rl.evaluation.walkforward import purged_walk_forward

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

            # These fold results inform model selection; use the same fixed
            # action contract as the final train/test reports.
            fold_result = _evaluate_final_split(
                fold_model,
                bars=fold_test_bars,
                features=fold_test_feat,
                obs_window=cfg.env.obs_window,
                initial_balance=cfg.account.initial_balance,
                guardrail_kwargs=_eval_guardrail_kwargs(cfg),
                risk_frac_range=_strategy_risk_ranges(cfg)[0],
                rr_ratio_range=_strategy_risk_ranges(cfg)[1],
                swing_buffer_pts=cfg.risk.swing_buffer_pts,
                sl_mode=str(cfg.risk.get("sl_mode", "fixed")),
                contract_size=cfg.account.contract_size,
                max_loss_per_trade_usd=_max_loss_per_trade(cfg),
                dsr_eta=cfg.env.reward_dsr_eta,
                max_episode_steps=_max_episode_steps(cfg),
                continuous_actions=(args.algo == "sac"),
                use_sweep_reward=(args.reward == "sweep"),
                block_overnight=bool(cfg.env.get("block_overnight", True)),
                eod_risk=dict(cfg.env.get("eod_risk", {})),
                strategy=_strategy_from_cfg(cfg)[0],
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
                direction_override_threshold=float(
                    cfg.env.get("direction_override_threshold", 0.0)
                ),
                allow_agent_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
                sl_mode_defer_threshold=float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
                allow_agent_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
                tp_mode_defer_threshold=float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
                allow_multi_tp=bool(cfg.env.get("allow_multi_tp", False)),
                allow_simplex=bool(cfg.env.get("allow_simplex", True)),
                tp_breakeven_alpha=float(cfg.env.get("tp_breakeven_alpha", 0.5)),
                tickbook=ticks_covering(primary_ticks, fold_test_bars),
                fill_delay_ms=_fill_delay_ms(cfg),
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

        wf_sharpes = [float(r["sharpe"]) for r in wf_results]
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
