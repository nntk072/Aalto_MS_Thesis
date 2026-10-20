"""Per-seed persistence and multi-seed aggregation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


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
    from training.configuration.variants import compute_tier1_funnel

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
