#!/usr/bin/env python3
"""Compare a saved PPO checkpoint through frame and shared-bundle evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from gymnasium import spaces

from quant_rl.action_schema import build_action_schema
from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.split import get_split_config, split_bars, split_train_test
from quant_rl.data.ticks import ticks_covering
from quant_rl.envs.env_bundle_builder import build_env_bundle
from quant_rl.envs.env_spec import EnvSpec
from quant_rl.envs.worker import make_bundle_env
from quant_rl.eval import eval_run
from quant_rl.eval.reanalysis_provenance import (
    _evaluation_dataset_records,
    _feature_build_scope,
    _load_exact_source_feature_cache,
    _validate_reanalysis_datasets,
)
from quant_rl.eval.rollout import BAR_CLOSE_EXPOSURE_MEASURE, evaluate_model
from quant_rl.eval.run_manifest import (
    build_sequence_feature_schema,
    build_stop_candidate_schema,
    validate_sequence_feature_schema,
)
from quant_rl.evaluation import calculate_metrics, evaluation_periods_per_year
from quant_rl.features.build import (
    build_features,
    select_legacy_obs_columns,
    select_obs_columns,
)
from quant_rl.models.agent import (
    validate_model_action_schema,
    validate_model_observation_schema,
)
from quant_rl.observation_schema import build_observation_schema
from quant_rl.train.train_rl import _build_eval_common, _strategy_from_cfg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(run_dir: Path, checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    for parent in run_dir.parents:
        path = parent / "attempt_provenance.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return path, value
    raise FileNotFoundError(f"attempt_provenance.json not found above {run_dir}")


def _metrics(result: dict[str, Any], cfg: Any) -> dict[str, Any]:
    value = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_sessions=int(result.get("n_sessions", 1)),
        n_breach_sessions=int(result.get("n_breach_sessions", 0)),
        initial_balance=float(cfg.account.initial_balance),
        periods_per_year=evaluation_periods_per_year(cfg),
    )
    raw = asdict(value) if is_dataclass(value) else vars(value)
    return {
        str(key): (float(item) if isinstance(item, (int, float, np.number)) else item)
        for key, item in raw.items()
    }


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        try:
            pd.testing.assert_series_equal(left, right, check_exact=True)
        except AssertionError:
            return False
        return True
    if is_dataclass(left):
        left = asdict(cast(Any, left))
    if is_dataclass(right):
        right = asdict(cast(Any, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return bool((np.isnan(left) and np.isnan(right)) or left == right)
    return bool(left == right)


def compare_results(frame_result: dict[str, Any], bundle_result: dict[str, Any]) -> dict[str, bool]:
    """Compare the outputs that determine policy behavior and reported results."""
    fields = (
        "equity",
        "trades",
        "account",
        "breaches",
        "breach_events",
        "n_sessions",
        "n_breach_sessions",
        "n_sessions_with_trades",
        "n_sessions_skipped",
        "days_traded",
        "survived_full_year",
        "completed_data_split",
        "episode_end_reason",
        "reward_sum",
        "reward_attribution",
        "n_steps",
        "exposure_bars",
        "exposure_fraction",
        "exposure_measure",
        "fail_time",
        "max_trailing_dd",
        "action_counts",
        "entry_diag",
        "entry_state_diagnostics",
    )
    return {field: _equal(frame_result[field], bundle_result[field]) for field in fields}


def _evaluate_bundle_result(
    model: Any,
    bundle_env: Any,
    *,
    strategy_actions: bool,
    continuous_actions: bool,
) -> dict[str, Any]:
    """Replay a bundle env with the same policy action interpretation as frames."""
    return evaluate_model(
        model,
        pd.DataFrame(),
        pd.DataFrame(),
        deterministic=True,
        continuous_actions=continuous_actions,
        strategy_actions=strategy_actions,
        env=bundle_env,
    )


def select_split(
    split: str,
    train_bars: pd.DataFrame,
    test_bars: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Choose the exact saved training or report-only test split for parity."""
    if split == "train":
        return train_bars, train_features, "in_sample_training"
    if split == "test":
        return test_bars, test_features, "previously_inspected_selection_acknowledged_test"
    raise ValueError(f"unsupported evaluation split: {split}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved source run directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument(
        "--bundle-dir", type=Path, required=True, help="New directory for the selected split bundle"
    )
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Acknowledge that the saved config's historical test split was inspected",
    )
    parser.add_argument(
        "--legacy-action-layout",
        default=None,
        help="Explicit layout for an unversioned legacy checkpoint, if needed",
    )
    parser.add_argument("--allow-legacy-action-schema", action="store_true")
    parser.add_argument("--allow-legacy-observation-schema", action="store_true")
    parser.add_argument("--allow-legacy-stop-schema", action="store_true")
    parser.add_argument(
        "--allow-legacy-sequence-schema",
        action="store_true",
        help="Explicitly replay a manifest without ordered sequence coordinates as unverified.",
    )
    parser.add_argument(
        "--source-feature-cache",
        type=Path,
        default=None,
        help="Read-only exact training feature cache to use for historical checkpoint parity.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Compare the complete saved training split or report-only test split (default: test)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_dir = args.run.resolve()
    output = args.output.resolve()
    bundle_dir = args.bundle_dir.resolve()
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint.is_absolute()
        else (run_dir / args.checkpoint).resolve()
    )
    getattr(eval_run, "_validate_reanalysis_request")(
        run_dir, output, allow_historical=args.allow_historical
    )
    if output.exists() or bundle_dir.exists():
        raise FileExistsError("refusing to overwrite an existing parity report or bundle")
    output.relative_to(Path.cwd().resolve())
    bundle_dir.relative_to(Path.cwd().resolve())
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    provenance_path, provenance = _checkpoint_provenance(run_dir, checkpoint)
    cfg = getattr(eval_run, "_load_run_config")(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("source run manifest must be a JSON object")
    sequence_schema_is_legacy = manifest.get("sequence_feature_schema") is None
    if sequence_schema_is_legacy and not args.allow_legacy_sequence_schema:
        raise ValueError(
            "source run lacks ordered sequence coordinates; pass "
            "--allow-legacy-sequence-schema only for explicitly unverified replay"
        )
    use_legacy_sequence_columns = bool(
        sequence_schema_is_legacy and args.allow_legacy_sequence_schema
    )
    model = eval_run._load_checkpoint(checkpoint, legacy_layout_key=args.legacy_action_layout)
    checkpoint_hash_disposition = eval_run._validate_checkpoint_manifest_hash(checkpoint, manifest)
    stop_schema_disposition = eval_run._validate_stop_candidate_schema(
        checkpoint_schema=getattr(model, "stop_candidate_schema", None),
        manifest_schema=manifest.get("stop_candidate_schema"),
        expected_schema=build_stop_candidate_schema(cfg),
        allow_legacy=bool(args.allow_legacy_stop_schema),
    )
    expected_schema = build_observation_schema(
        str(cfg.env.get("observation_schema_version", "account-state-v1")),
        entry_state_visibility=bool(
            cfg.env.get("entry_state_machine", False)
            and cfg.env.get("entry_state_observation", True)
        ),
    )
    validate_model_observation_schema(
        model, expected_schema, allow_legacy_v1=args.allow_legacy_observation_schema
    )
    strategy_actions = bool(cfg.env.get("strategy_actions", False))
    if strategy_actions:
        multi_tp = bool(cfg.env.get("allow_multi_tp", False))
        schema = build_action_schema(
            has_direction=bool(cfg.env.get("agent_direction_control", False)),
            has_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
            has_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
            has_multi_tp=multi_tp,
            has_simplex=multi_tp and bool(cfg.env.get("allow_simplex", True)),
        )
        validate_model_action_schema(
            model,
            schema,
            allow_legacy=args.allow_legacy_action_schema,
            legacy_layout_key=args.legacy_action_layout,
        )
    elif isinstance(model.action_space, spaces.Box):
        raise ValueError("Box checkpoint with non-overlay config has ambiguous action semantics")

    data = run_pipeline(cfg)
    primary_symbol = str(cfg.data.primary)
    secondary_symbol = str(cfg.data.secondary)
    primary_m1 = data[primary_symbol]["M1"]
    secondary_m1 = data.get(secondary_symbol, {}).get("M1")
    train_end, test_start = get_split_config(cfg)
    if args.source_feature_cache is not None:
        config_record = manifest.get("config")
        expected_config_hash = (
            config_record.get("sha256") if isinstance(config_record, dict) else None
        )
        if not isinstance(expected_config_hash, str) or _sha256(run_dir / "config.yaml") != (
            expected_config_hash
        ):
            raise ValueError("saved run config does not match the source manifest")
        cache_path = args.source_feature_cache.expanduser()
        if not cache_path.is_absolute():
            cache_path = (Path.cwd() / cache_path).resolve()
        features, feature_cache, _, _ = _load_exact_source_feature_cache(
            manifest.get("feature_cache"), cache_path
        )
    else:
        train_mask, feature_cache = _feature_build_scope(
            primary_m1, cfg, Path(cfg.data.cache_dir), primary_symbol, train_end, secondary_m1
        )
        features = build_features(
            primary_m1,
            secondary=secondary_m1,
            cfg=cfg,
            train_mask=train_mask,
            cache_path=feature_cache,
        )
    train_bars, test_bars, train_features, test_features = split_train_test(
        primary_m1, features, train_end, test_start
    )
    train_secondary, test_secondary = (None, None)
    if secondary_m1 is not None and not secondary_m1.empty:
        train_secondary, test_secondary = split_bars(secondary_m1, train_end, test_start)
    primary_ticks = build_tick_books(cfg).get(primary_symbol)
    dataset_disposition = _validate_reanalysis_datasets(
        manifest.get("datasets"),
        _evaluation_dataset_records(
            train_bars=train_bars,
            train_features=train_features,
            test_bars=test_bars,
            test_features=test_features,
            train_secondary=train_secondary,
            test_secondary=test_secondary,
            primary_ticks=primary_ticks,
        ),
        allow_legacy=False,
    )
    eval_bars, eval_features, split_role = select_split(
        args.split, train_bars, test_bars, train_features, test_features
    )
    eval_tickbook = ticks_covering(primary_ticks, eval_bars)
    eval_common, _, _, _, _ = _build_eval_common(cfg, args=None, env_vae=None, pre_ny_by_date=None)
    strategy = _strategy_from_cfg(cfg)[0]
    sequence_selector = (
        select_legacy_obs_columns if use_legacy_sequence_columns else select_obs_columns
    )
    sequence_columns = [
        str(column) for column in sequence_selector(features, strategy.raw_columns).columns
    ]
    if "reachable_r" not in sequence_columns:
        sequence_columns.append("reachable_r")
    sequence_schema = build_sequence_feature_schema(sequence_columns)
    sequence_schema_disposition = validate_sequence_feature_schema(
        manifest.get("sequence_feature_schema"),
        sequence_schema,
        allow_legacy=use_legacy_sequence_columns,
    )
    eval_run._validate_sequence_input_width(model, sequence_schema["width"])
    frame_result = evaluate_model(
        model,
        bars=eval_bars,
        features=eval_features,
        tickbook=eval_tickbook,
        deterministic=True,
        legacy_sequence_columns=use_legacy_sequence_columns,
        **eval_common,
    )

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = build_env_bundle(
        bundle_dir,
        eval_bars,
        eval_features,
        cfg,
        raw_columns=list(strategy.raw_columns),
        feature_config_hash=feature_cache.stem,
        tickbook=eval_tickbook,
        pre_ny_by_date=None,
        legacy_sequence_columns=use_legacy_sequence_columns,
    )
    arch = "mtf" if bool(eval_common.get("mtf", False)) else "tcn"
    spec = EnvSpec.from_cfg(
        cfg,
        algo="ppo",
        reward="sweep" if bool(eval_common.get("use_sweep_reward", False)) else "dsr",
        arch=arch,
        bundle_dir=str(bundle_path),
        episodic=False,
        evaluation=True,
    )
    bundle_env = make_bundle_env(spec)
    bundle_result = _evaluate_bundle_result(
        model,
        bundle_env,
        continuous_actions=bool(eval_common["continuous_actions"]),
        strategy_actions=bool(eval_common["strategy_actions"]),
    )
    comparisons = compare_results(frame_result, bundle_result)
    frame_metrics = _metrics(frame_result, cfg)
    bundle_metrics = _metrics(bundle_result, cfg)
    metric_equal = _equal(frame_metrics, bundle_metrics)
    matches = all(comparisons.values()) and metric_equal
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    working_tree_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed" if matches else "failed",
        "artifact_role": "historical_reanalysis_backend_parity",
        "historical_holdout_inspected": True,
        "campaign_id": provenance.get("campaign_id"),
        "campaign_fingerprint": provenance.get("campaign_fingerprint"),
        "source_attempt": provenance.get("attempt"),
        "repository": {
            "head": current_head,
            "working_tree_dirty": working_tree_dirty,
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
        },
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "source_run": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_manifest_hash_disposition": checkpoint_hash_disposition,
        "stop_schema_disposition": stop_schema_disposition,
        "checkpoint_timesteps": int(model.num_timesteps),
        "attempt_provenance": str(provenance_path),
        "attempt_provenance_sha256": _sha256(provenance_path),
        "config_sha256": _sha256(run_dir / "config.yaml"),
        "sequence_feature_schema": sequence_schema,
        "sequence_feature_schema_disposition": sequence_schema_disposition,
        "legacy_sequence_columns_used": use_legacy_sequence_columns,
        "dataset_parity_disposition": dataset_disposition,
        "source_feature_cache_replay": args.source_feature_cache is not None,
        "feature_cache": str(feature_cache),
        "feature_cache_sha256": _sha256(feature_cache),
        "split": {
            "name": args.split,
            "role": split_role,
            "train_end": train_end,
            "test_start": test_start,
            "bars": int(len(eval_bars)),
        },
        "policy_action_mode": "deterministic_saved_checkpoint",
        "metrics_protocol": {
            "annualization": {
                "periods_per_year": evaluation_periods_per_year(cfg),
                "convention": str(cfg.evaluation.annualization_convention),
            },
            "initial_balance": float(cfg.account.initial_balance),
            "frequency": "bar_close_equity",
            "exposure_measure": BAR_CLOSE_EXPOSURE_MEASURE,
        },
        "backends": ["frames", "shared_bundle"],
        "bundle_path": str(bundle_path),
        "comparisons": {**comparisons, "aggregate_metrics": metric_equal},
        "frame_metrics": frame_metrics,
        "bundle_metrics": bundle_metrics,
        "frame_trade_rows": int(len(frame_result["trades"])),
        "bundle_trade_rows": int(len(bundle_result["trades"])),
        "frame_steps": int(frame_result["n_steps"]),
        "bundle_steps": int(bundle_result["n_steps"]),
        "limitations": [
            "Backend parity does not establish training completion, strategy validity, or confirmatory evidence.",
            "The test split was previously inspected and selection-acknowledged; the training split is in-sample. Neither is pristine confirmatory OOS evidence.",
            "Repository HEAD, working-tree state, and audit-script hash identify the parity implementation used.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "comparisons": report["comparisons"],
                "frame_trades": report["frame_metrics"].get("n_trades"),
                "bundle_trades": report["bundle_metrics"].get("n_trades"),
            },
            sort_keys=True,
        )
    )
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
