"""Training-facing environment setup compatibility surface."""

from __future__ import annotations

from typing import Any

from decisioning.adapters.sb3 import SB3DecisionProvider
from decisioning.contracts import DecisionProvider
from evaluation.rollout.model import evaluate_model
from simulation.environment.factory import (
    build_eval_common,
    cost_model,
    eval_guardrail_kwargs,
    fill_delay_ms,
    guardrail_kwargs,
    make_env,
    max_episode_steps,
    max_loss_per_trade,
    max_tick_quote_wait_ms,
    mtf_settings,
    randomized_fill_latency_ms,
    strategy_from_cfg,
    strategy_risk_ranges,
    tick_execution_required,
)
from trading.policy.config import resolve_trading_policy

__all__ = [
    "build_eval_common",
    "build_eval_provider",
    "build_sb3_provider",
    "cost_model",
    "eval_guardrail_kwargs",
    "_evaluate_final_split",
    "fill_delay_ms",
    "guardrail_kwargs",
    "max_episode_steps",
    "max_loss_per_trade",
    "max_tick_quote_wait_ms",
    "mtf_settings",
    "randomized_fill_latency_ms",
    "strategy_from_cfg",
    "strategy_risk_ranges",
    "tick_execution_required",
    "_training_episodic",
    "make_env",
]


def _training_episodic(cfg: Any) -> bool:
    """Resolve whether risk breaches terminate training episodes."""
    if resolve_trading_policy(cfg).repeated_payout:
        return True
    training = cfg.get("training", {}) if hasattr(cfg, "get") else getattr(cfg, "training", {})
    mode = str(training.get("episode_mode", "full_split")).strip().lower()
    if mode == "full_split":
        if max_episode_steps(cfg) is not None:
            raise ValueError(
                "training.episode_mode='full_split' requires env.max_episode_steps=null"
            )
        return False
    if mode == "risk_terminated":
        return True
    raise ValueError(
        f"training.episode_mode must be 'full_split' or 'risk_terminated', got {mode!r}"
    )


def _evaluate_final_split(model: Any, **kwargs: Any) -> dict[str, Any]:
    """Evaluate a reported split with deterministic policy actions."""
    kwargs["deterministic"] = True
    return evaluate_model(model, **kwargs)


def build_sb3_provider(model: Any, input_schema_version: str) -> DecisionProvider:
    """Wrap an SB3 model at a training/evaluation composition boundary."""
    return SB3DecisionProvider(model, input_schema_version=input_schema_version)


def build_eval_provider(
    model: Any,
    *,
    cfg: Any,
    args: Any,
    env_vae: Any,
    pre_ny_by_date: Any,
    arch: str | None,
) -> DecisionProvider:
    """Build a provider using the same resolved observation schema as evaluation."""
    eval_common, *_ = build_eval_common(cfg, args, env_vae, pre_ny_by_date, arch=arch)
    return build_sb3_provider(model, str(eval_common["observation_schema_version"]))
