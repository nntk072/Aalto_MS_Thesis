"""Training callback construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from learning.rl.callbacks import (
    EarlyAbortCallback,
    LagrangianRiskCallback,
    PeriodicCheckpointCallback,
)
from training.setup.environments import _training_episodic

log = logging.getLogger(__name__)


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
