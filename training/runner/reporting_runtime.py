from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.artifacts.aggregation import _aggregate_seed_results
from training.artifacts.manifests import _build_training_log


def finalize_training_run(
    *,
    cfg: Any,
    args: Any,
    is_multi_seed: bool,
    seed_results: list[dict[str, Any]],
    timesteps: int,
    train_bar_count: int,
    test_bar_count: int,
    run_dir: Path,
    feat_cache: Path,
    feature_columns: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Aggregate seed results, persist the training log, and emit optional W&B metrics."""
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
            wandb.log(
                {
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
            )
        wandb.finish()

    return training_log
