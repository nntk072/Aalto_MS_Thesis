"""Training run manifest and schema helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from omegaconf import OmegaConf

from evaluation.reporting.run_manifest import build_sequence_feature_schema


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
    from training.configuration.variants import compute_tier1_funnel

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
        from training.configuration.quality import quality_variant_metadata

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
