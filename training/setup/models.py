"""Model initialization, resume validation, and provenance helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


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
