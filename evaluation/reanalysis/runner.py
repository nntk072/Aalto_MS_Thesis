"""Reanalyse a PPO checkpoint into a separate, immutable run destination.

Historical holdout reanalysis is opt-in and explicitly labelled as previously
inspected; it must never overwrite the original training or testing artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configuration.paths import repository_root
from decisioning.actions import build_action_schema, build_discrete_trading_action_schema
from decisioning.adapters.sb3 import SB3DecisionProvider
from decisioning.observations import build_observation_schema
from evaluation import calculate_metrics, evaluation_periods_per_year
from evaluation.reanalysis.checkpoint import resolve_checkpoint
from evaluation.reanalysis.cli import build_reanalysis_parser
from evaluation.reanalysis.datasets import reconstruct_reanalysis_datasets
from evaluation.reanalysis.legacy_sb3 import (
    is_sb3_box_model,
    is_sb3_discrete_model,
    load_sb3_checkpoint,
    seed_sb3,
    validate_sb3_discrete_action_schema,
    validate_sb3_observation_schema,
    validate_sb3_overlay_action_schema,
)
from evaluation.reanalysis.provenance import (
    _evaluation_dataset_records,
    _load_run_config,
    _validate_reanalysis_request,
)
from evaluation.reanalysis.validation import (
    resolve_execution_seed,
    validate_checkpoint_manifest_hash,
    validate_sequence_input_width,
    validate_stop_candidate_schema,
)
from evaluation.reporting.export import save_run
from evaluation.reporting.run_manifest import (
    build_evaluation_provenance,
    build_sequence_feature_schema,
    build_stop_candidate_schema,
    file_sha256,
    validate_sequence_feature_schema,
)
from evaluation.rollout.model import BAR_CLOSE_EXPOSURE_MEASURE, evaluate_model, evaluate_provider
from features import select_legacy_obs_columns, select_obs_columns
from market_data.loading.ticks import ticks_covering
from simulation.environment.factory import (
    build_eval_common,
)
from trading.policy.config import resolve_trading_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

__all__ = [
    "build_reanalysis_parser",
    "evaluate_checkpoint_split",
    "evaluate_train_split",
    "load_checkpoint",
    "main",
    "resolve_checkpoint",
    "resolve_execution_seed",
    "validate_checkpoint_manifest_hash",
    "validate_sequence_input_width",
    "validate_stop_candidate_schema",
    "write_train_only_screen_report",
]


def load_checkpoint(path: Path, *, legacy_layout_key: str | None) -> Any:
    """Compatibility entry point; SB3 checkpoint loading belongs to its adapter."""
    return load_sb3_checkpoint(path, legacy_layout_key=legacy_layout_key)


def _source_final_evaluation_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "evaluations" / "test_report_only.json"
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_reanalysis_action_mode(run_dir: Path, requested_stochastic: bool | None) -> bool:
    """Use an explicit CLI mode or inherit the source run's final test mode."""
    if requested_stochastic is not None:
        return requested_stochastic
    return _source_final_evaluation_metadata(run_dir).get("policy_action_mode") == "stochastic"


def resolve_reanalysis_action_seed(
    run_dir: Path, requested_seed: int | None, *, stochastic: bool
) -> int | None:
    """Use an explicit seed or inherit the source run's sampled-action seed."""
    if requested_seed is not None:
        return requested_seed
    if not stochastic:
        return None
    value = _source_final_evaluation_metadata(run_dir).get("action_rng_seed")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def evaluate_checkpoint_split(provider: Any, *, stochastic: bool, **kwargs: Any) -> dict[str, Any]:
    """Replay one final split using the explicitly selected policy action mode."""
    if hasattr(provider, "metadata") and hasattr(provider, "predict"):
        return evaluate_provider(provider, deterministic=not stochastic, **kwargs)
    return evaluate_model(provider, deterministic=not stochastic, **kwargs)


def evaluate_train_split(
    provider: Any,
    *,
    stochastic: bool,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    tickbook: Any,
    legacy_sequence_columns: bool,
    eval_common: dict[str, Any],
    initial_balance: float,
    periods_per_year: int,
) -> tuple[dict[str, Any], Any]:
    """Evaluate only the supplied training split and calculate its metrics."""
    result = evaluate_checkpoint_split(
        provider,
        stochastic=stochastic,
        bars=bars,
        features=features,
        tickbook=tickbook,
        legacy_sequence_columns=legacy_sequence_columns,
        **eval_common,
    )
    result["initial_balance"] = initial_balance
    metrics = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_sessions=result.get("n_sessions", 1),
        n_breach_sessions=result.get("n_breach_sessions", 0),
        initial_balance=initial_balance,
        periods_per_year=periods_per_year,
    )
    return result, metrics


def write_train_only_screen_report(output_dir: Path, report: dict[str, Any]) -> Path:
    """Write the minimal train-only candidate screen into a new directory."""
    report["evaluation_provenance"] = build_evaluation_provenance(repository_root(__file__))
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "train_only_screen.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report_path


def main() -> None:
    args = build_reanalysis_parser().parse_args()
    if args.train_only_screen and args.stochastic is True:
        raise SystemExit("--train-only-screen requires deterministic policy actions")
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    stochastic = resolve_reanalysis_action_mode(run_dir, args.stochastic)
    action_rng_seed = resolve_reanalysis_action_seed(run_dir, args.seed, stochastic=stochastic)
    if args.train_only_screen and stochastic:
        raise SystemExit("--train-only-screen requires deterministic policy actions")
    output_dir = Path(args.output)
    _validate_reanalysis_request(run_dir, output_dir, allow_historical=args.allow_historical)

    if action_rng_seed is not None:
        log.info("Sampled-action RNG seed: %d", action_rng_seed)
    elif stochastic:
        log.warning(
            "Stochastic policy actions enabled without an action RNG seed; this evaluation "
            "will not reproduce a prior sampled-action replay."
        )
    policy_action_mode = "stochastic" if stochastic else "deterministic"
    log.info("Policy action mode: %s", policy_action_mode)

    def seed_action_rng() -> None:
        if action_rng_seed is not None:
            random.seed(action_rng_seed)
            np.random.seed(action_rng_seed)
            seed_sb3(action_rng_seed)

    cfg = _load_run_config(run_dir)
    # The run's own snapshot is authoritative for data, split, and risk settings,
    # but how many charts to draw is a present-day rendering choice. A snapshot
    # written before a config change must not force the slow default back on.
    save_html = (
        bool(getattr(cfg.output, "save_html", False)) if args.save_html is None else args.save_html
    )
    max_order_charts = args.max_charts
    if max_order_charts is None:
        max_order_charts = int(getattr(cfg.output, "max_order_charts", 0) or 0)
    log.info(
        "Chart rendering: save_html=%s max_order_charts=%d (CLI overrides snapshot)",
        save_html,
        max_order_charts,
    )
    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
    log.info("Loading model: %s", ckpt_path)
    source_manifest_path = run_dir / "run_manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.is_file():
        manifest_value = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_value, dict):
            raise ValueError("source run manifest must be a JSON object")
        source_manifest = manifest_value
    source_feature_cache = source_manifest.get("feature_cache")
    source_datasets = source_manifest.get("datasets")
    if (
        source_feature_cache is None or source_datasets is None
    ) and not args.allow_legacy_data_provenance:
        raise ValueError(
            "source run manifest lacks feature-cache or dataset identities; pass "
            "--allow-legacy-data-provenance only for an explicitly unverified legacy replay"
        )
    checkpoint_hash_disposition = validate_checkpoint_manifest_hash(ckpt_path, source_manifest)
    model: Any = load_checkpoint(ckpt_path, legacy_layout_key=args.legacy_action_layout)
    expected_stop_schema = build_stop_candidate_schema(cfg)
    checkpoint_stop_schema = getattr(model, "stop_candidate_schema", None)
    manifest_stop_schema = source_manifest.get("stop_candidate_schema")
    stop_schema_disposition = validate_stop_candidate_schema(
        checkpoint_schema=checkpoint_stop_schema,
        manifest_schema=manifest_stop_schema,
        expected_schema=expected_stop_schema,
        allow_legacy=bool(args.allow_legacy_stop_schema),
    )
    policy = resolve_trading_policy(cfg)
    expected_observation_schema = build_observation_schema(
        (
            "account-state-v3-repeated-payout"
            if policy.repeated_payout
            else str(cfg.env.get("observation_schema_version", "account-state-v1"))
        ),
        entry_state_visibility=bool(
            cfg.env.get("entry_state_machine", False)
            and cfg.env.get("entry_state_observation", True)
        ),
    )
    validate_sb3_observation_schema(
        model,
        expected_observation_schema,
        allow_legacy_v1=bool(args.allow_legacy_observation_schema),
    )
    is_overlay = bool(cfg.env.get("strategy_actions", False))
    if is_overlay:
        if args.legacy_action_layout == "legacy4_no_exit_v1":
            if not args.allow_legacy_action_schema:
                raise ValueError("legacy4_no_exit_v1 requires --allow-legacy-action-schema")
            # Historical 4D policies had no exit coordinate. Pin every optional
            # coordinate off before constructing the current 5D target layout.
            cfg.env.agent_direction_control = False
            cfg.env.allow_agent_sl_mode = False
            cfg.env.allow_agent_tp_mode = False
            cfg.env.allow_multi_tp = False
            cfg.env.allow_simplex = False
        multi_tp = bool(cfg.env.get("allow_multi_tp", False))
        schema = build_action_schema(
            has_direction=bool(cfg.env.get("agent_direction_control", False)),
            has_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
            has_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
            has_multi_tp=multi_tp,
            has_simplex=multi_tp and bool(cfg.env.get("allow_simplex", True)),
        )
        model = validate_sb3_overlay_action_schema(
            model,
            schema,
            allow_legacy=args.allow_legacy_action_schema,
            legacy_layout_key=args.legacy_action_layout,
        )
    elif is_sb3_box_model(model):
        raise ValueError("Box checkpoint with non-overlay run config: action semantics unclear")
    elif is_sb3_discrete_model(model):
        discrete_schema = build_discrete_trading_action_schema(int(model.action_space.n))
        source_action_schema = source_manifest.get("action_schema")
        if source_action_schema is not None and not isinstance(source_action_schema, dict):
            raise ValueError("source run manifest action_schema must be an object or null")
        validate_sb3_discrete_action_schema(
            model,
            discrete_schema,
            source_action_schema,
            allow_legacy=bool(args.allow_legacy_action_schema),
            legacy_layout_key=args.legacy_action_layout,
        )

    datasets = reconstruct_reanalysis_datasets(
        args=args,
        run_dir=run_dir,
        cfg=cfg,
        source_manifest=source_manifest,
    )
    primary_ticks = datasets.primary_ticks
    features = datasets.features
    train_bars = datasets.train_bars
    test_bars = datasets.test_bars
    train_feat = datasets.train_features
    test_feat = datasets.test_features
    train_sec = datasets.train_secondary
    test_sec = datasets.test_secondary
    train_end = datasets.train_end
    test_start = datasets.test_start
    feat_cache = datasets.feature_cache
    feature_cache_disposition = datasets.feature_cache_disposition
    evaluated_feature_cache_hash = datasets.evaluated_feature_cache_hash
    evaluated_feature_cache_file_sha256 = datasets.evaluated_feature_cache_file_sha256
    source_cache_replay_path = datasets.source_cache_replay_path
    evaluated_datasets = datasets.dataset_records
    dataset_parity_disposition = datasets.dataset_parity_disposition
    training_data_source = datasets.training_data_source
    backend_parity_disposition = datasets.backend_parity_disposition
    train_tickbook = ticks_covering(primary_ticks, train_bars)

    # Eval CLI does not expose training --algo / --reward flags; PPO replay
    # uses the configuration's saved environment defaults.
    if args.tick_fill_slippage_points < 0:
        raise ValueError("--tick-fill-slippage-points cannot be negative")
    eval_common, eval_strategy, _, _, _ = build_eval_common(
        cfg,
        args=None,
        env_vae=None,
        pre_ny_by_date=None,
        tick_fill_slippage_points=args.tick_fill_slippage_points,
    )
    if args.tick_fill_slippage_points and not eval_common["tick_execution_required"]:
        raise ValueError("tick-fill slippage override requires tick execution in the saved config")
    execution_seed = resolve_execution_seed(args.seed, source_manifest)
    if eval_common["randomized_fill_latency_ms"] > 0 and execution_seed is None:
        raise ValueError(
            "randomized tick-fill reanalysis requires --seed or a seed in the source run manifest"
        )
    eval_common["execution_seed"] = execution_seed
    sequence_schema_is_legacy = source_manifest.get("sequence_feature_schema") is None
    use_legacy_sequence_columns = bool(
        sequence_schema_is_legacy and args.allow_legacy_sequence_schema
    )
    observation_column_selector = (
        select_legacy_obs_columns if use_legacy_sequence_columns else select_obs_columns
    )
    sequence_columns = [
        str(column)
        for column in observation_column_selector(features, eval_strategy.raw_columns).columns
    ]
    if "reachable_r" not in sequence_columns:
        sequence_columns.append("reachable_r")
    expected_sequence_schema = build_sequence_feature_schema(sequence_columns)
    sequence_schema_disposition = validate_sequence_feature_schema(
        source_manifest.get("sequence_feature_schema"),
        expected_sequence_schema,
        allow_legacy=bool(args.allow_legacy_sequence_schema),
    )
    validate_sequence_input_width(model, expected_sequence_schema["width"])
    provider = SB3DecisionProvider(
        model,
        input_schema_version=expected_observation_schema.version,
    )

    train_tickbook = ticks_covering(primary_ticks, train_bars)
    if args.train_only_screen:
        seed_action_rng()
        train_result, train_m = evaluate_train_split(
            provider,
            stochastic=stochastic,
            bars=train_bars,
            features=train_feat,
            tickbook=train_tickbook,
            legacy_sequence_columns=use_legacy_sequence_columns,
            eval_common=eval_common,
            initial_balance=float(cfg.account.initial_balance),
            periods_per_year=evaluation_periods_per_year(cfg),
        )
        log.info(
            "[train screen] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  fail_time=%s",
            train_m.sharpe,
            train_m.max_drawdown * 100,
            train_m.n_trades,
            train_m.total_return_pct,
            train_result.get("fail_time"),
        )
        train_dataset_records = _evaluation_dataset_records(
            train_bars=train_bars,
            train_features=train_feat,
            test_bars=train_bars.iloc[:0],
            test_features=train_feat.iloc[:0],
            train_secondary=train_sec,
            primary_ticks=train_tickbook,
        )
        train_dataset_records = {
            key: value
            for key, value in train_dataset_records.items()
            if key in {"train_bars", "train_features", "train_secondary", "primary_ticks"}
        }
        report_path = write_train_only_screen_report(
            output_dir,
            {
                "artifact_role": "candidate_screen_train_only",
                "test_split_evaluated": False,
                "source_run": str(run_dir.resolve()),
                "source_run_manifest_sha256": (
                    file_sha256(source_manifest_path) if source_manifest_path.is_file() else None
                ),
                "source_config_sha256": (
                    file_sha256(run_dir / "config.yaml")
                    if (run_dir / "config.yaml").is_file()
                    else None
                ),
                "checkpoint": str(ckpt_path.resolve()),
                "checkpoint_sha256": file_sha256(ckpt_path),
                "checkpoint_manifest_hash_disposition": checkpoint_hash_disposition,
                "training_split": {
                    "start": str(train_bars.index.min()),
                    "end": str(train_bars.index.max()),
                    "bar_count": len(train_bars),
                    "metrics": asdict(train_m),
                    "trade_count": train_m.n_trades,
                    "survived_full_period": bool(train_result.get("survived_full_year", False)),
                    "completed_data_split": bool(train_result.get("completed_data_split", False)),
                    "episode_end_reason": train_result.get("episode_end_reason"),
                    "fail_time": (
                        str(train_result["fail_time"])
                        if train_result.get("fail_time") is not None
                        else None
                    ),
                },
                "training_input_identity": train_dataset_records,
                "dataset_parity_disposition": dataset_parity_disposition,
                "tick_identity_scope": {
                    "source_manifest_full_tick_record": (
                        source_datasets.get("primary_ticks")
                        if isinstance(source_datasets, dict)
                        else None
                    ),
                    "replayed_training_tick_record": train_dataset_records.get("primary_ticks"),
                    "full_tick_identity_reloaded": False,
                    "disposition": "training_partition_only",
                },
                "feature_cache": {
                    "identity": feat_cache.name,
                    "content_hash": evaluated_feature_cache_hash or None,
                    "file_sha256": evaluated_feature_cache_file_sha256,
                    "disposition": feature_cache_disposition,
                },
                "evaluation_protocol": {
                    "policy_action_mode": policy_action_mode,
                    "seed": action_rng_seed,
                    "execution_seed": execution_seed,
                    "training_data_source": training_data_source,
                    "evaluation_backend": "frame",
                    "backend_parity_disposition": backend_parity_disposition,
                    "sequence_feature_schema": expected_sequence_schema,
                    "sequence_feature_schema_disposition": sequence_schema_disposition,
                    "observation_schema": expected_observation_schema.to_dict(),
                },
            },
        )
        log.info("Train-only screen written to %s", report_path)
        return

    seed_action_rng()
    test_result = evaluate_checkpoint_split(
        provider,
        stochastic=stochastic,
        bars=test_bars,
        features=test_feat,
        tickbook=ticks_covering(primary_ticks, test_bars),
        legacy_sequence_columns=use_legacy_sequence_columns,
        **eval_common,
    )
    test_result["initial_balance"] = cfg.account.initial_balance
    test_m = calculate_metrics(
        test_result["equity"],
        trades=test_result["trades"],
        n_sessions=test_result.get("n_sessions", 1),
        n_breach_sessions=test_result.get("n_breach_sessions", 0),
        initial_balance=float(cfg.account.initial_balance),
        periods_per_year=evaluation_periods_per_year(cfg),
    )
    log.info(
        "[test] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  fail_time=%s",
        test_m.sharpe,
        test_m.max_drawdown * 100,
        test_m.n_trades,
        test_m.total_return_pct,
        test_result.get("fail_time"),
    )
    seed_action_rng()
    train_result, train_m = evaluate_train_split(
        provider,
        stochastic=stochastic,
        bars=train_bars,
        features=train_feat,
        tickbook=train_tickbook,
        legacy_sequence_columns=use_legacy_sequence_columns,
        eval_common=eval_common,
        initial_balance=float(cfg.account.initial_balance),
        periods_per_year=evaluation_periods_per_year(cfg),
    )
    log.info(
        "[train] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  fail_time=%s",
        train_m.sharpe,
        train_m.max_drawdown * 100,
        train_m.n_trades,
        train_m.total_return_pct,
        train_result.get("fail_time"),
    )
    if test_m.n_trades == 0:
        log.error(
            "INACTIVE: evaluation produced zero completed trades; "
            "metrics are not meaningful until action behavior is diagnosed."
        )

    # Create the new destination exclusively. Even a race or interrupted
    # reanalysis cannot erase source artifacts or reuse an existing output.
    output_dir.mkdir(parents=True, exist_ok=False)
    save_run(
        run_dir=output_dir,
        train_result=train_result,
        train_metrics=train_m,
        train_bars=train_bars,
        train_secondary=train_sec,
        train_features=train_feat,
        test_result=test_result,
        test_metrics=test_m,
        test_bars=test_bars,
        test_secondary=test_sec,
        test_features=test_feat,
        cfg=cfg,
        save_plots=getattr(cfg.output, "save_plots", True),
        save_html=save_html,
        save_csv=getattr(cfg.output, "save_csv", True),
        dpi=getattr(cfg.output, "dpi", 150),
        max_order_charts=max_order_charts,
    )

    source_log_path = run_dir / "training_log.json"
    training_log_path = output_dir / "training_log.json"
    training_log = json.loads(source_log_path.read_text()) if source_log_path.exists() else {}
    training_log.update(
        {
            "test_sharpe": float(test_m.sharpe),
            "test_max_dd": float(test_m.max_drawdown),
            "test_trades": test_m.n_trades,
            "test_return": float(test_m.total_return_pct),
            "test_breaches": test_result.get("n_breach_sessions", 0),
            "test_survived_full_year": bool(test_result.get("survived_full_year", False)),
            "test_completed_data_split": bool(test_result.get("completed_data_split", False)),
            "test_episode_end_reason": test_result.get("episode_end_reason"),
            "test_fail_time": (
                str(test_result["fail_time"]) if test_result.get("fail_time") is not None else None
            ),
            "eval_checkpoint": ckpt_path.name,
            "eval_timestamp": datetime.now().isoformat(),
            "eval_ftmo": True,
            "eval_policy_action_mode": policy_action_mode,
            "eval_seed": action_rng_seed,
            "eval_execution_seed": execution_seed,
            "eval_tick_fill_slippage_points": args.tick_fill_slippage_points,
            "sequence_feature_schema": expected_sequence_schema,
            "sequence_feature_schema_disposition": sequence_schema_disposition,
        }
    )
    training_log_path.write_text(json.dumps(training_log, indent=2))

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "artifact_role": "historical_reanalysis",
        "historical_holdout_inspected": True,
        "evaluation_provenance": build_evaluation_provenance(repository_root(__file__)),
        "source_run": str(run_dir.resolve()),
        "source_config_sha256": (
            _sha256(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else None
        ),
        "reanalysis_config_sha256": _sha256(output_dir / "config.yaml"),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": _sha256(ckpt_path),
        "checkpoint_manifest_hash_disposition": checkpoint_hash_disposition,
        "action_schema_version": getattr(model, "_action_schema_version", None),
        "action_layout_key": getattr(model, "_action_layout_key", None),
        "action_schema": getattr(model, "_action_schema", None),
        "observation_schema": expected_observation_schema.to_dict(),
        "trading_policy": policy.to_dict(),
        "sequence_feature_schema": expected_sequence_schema,
        "sequence_feature_schema_disposition": sequence_schema_disposition,
        "legacy_sequence_columns_used": use_legacy_sequence_columns,
        "legacy_action_schema_opt_in": bool(args.allow_legacy_action_schema),
        "legacy_observation_schema_opt_in": bool(args.allow_legacy_observation_schema),
        "legacy_sequence_schema_opt_in": bool(args.allow_legacy_sequence_schema),
        "legacy_stop_schema_opt_in": bool(args.allow_legacy_stop_schema),
        "legacy_data_provenance_opt_in": bool(args.allow_legacy_data_provenance),
        "stop_schema_disposition": stop_schema_disposition,
        "stop_candidate_schema": checkpoint_stop_schema or manifest_stop_schema,
        "legacy_action_layout_selected": args.legacy_action_layout,
        "feature_cache_identity": feat_cache.name,
        "feature_cache_hash": evaluated_feature_cache_hash or None,
        "feature_cache_file_sha256": evaluated_feature_cache_file_sha256,
        "evaluated_feature_cache_path": str(feat_cache.resolve()),
        "source_feature_cache_replay_path": (
            str(feat_cache) if source_cache_replay_path is not None else None
        ),
        "source_feature_cache": source_feature_cache,
        "feature_cache_disposition": feature_cache_disposition,
        "source_datasets": source_datasets,
        "evaluated_datasets": evaluated_datasets,
        "dataset_parity_disposition": dataset_parity_disposition,
        "training_data_source": training_data_source,
        "evaluation_backend": "frame",
        "backend_parity_disposition": backend_parity_disposition,
        "train_end": train_end,
        "test_start": test_start,
        "seed": action_rng_seed,
        "metrics_protocol": {
            "annualization": {
                "periods_per_year": evaluation_periods_per_year(cfg),
                "convention": str(
                    (cfg.get("evaluation") or {}).get(
                        "annualization_convention",
                        "legacy M1 convention: 252 sessions/year x 390 bars/session",
                    )
                ),
            },
            "initial_balance": float(cfg.account.initial_balance),
            "frequency": "bar_close_equity",
            "exposure_measure": BAR_CLOSE_EXPOSURE_MEASURE,
        },
        "policy_action_mode": policy_action_mode,
        "execution_seed": execution_seed,
        "execution_cost_override": {
            "configured_cost_model_slippage_points": float(
                eval_common["cost_model"].slippage_points
            ),
            "effective_tick_fill_slippage_points": float(args.tick_fill_slippage_points),
            "applies_to": "executed_tick_side_only",
            "position_sizing_cost_model_unchanged": True,
            "trigger_quote_unchanged": True,
            "opposite_quote_side_unchanged": True,
        },
        "reanalysis_timestamp": datetime.now().isoformat(),
        "validity_note": "Historical inspected holdout; not pristine OOS evidence.",
    }
    (output_dir / "reanalysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done. Reanalysis artifacts written to: %s", output_dir)


if __name__ == "__main__":
    main()
