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
from quant_rl.data.split import get_split_config, split_train_test
from quant_rl.data.ticks import ticks_covering
from quant_rl.envs.env_bundle_builder import build_env_bundle
from quant_rl.envs.env_spec import EnvSpec
from quant_rl.envs.worker import make_bundle_env
from quant_rl.eval import eval_run
from quant_rl.eval.reanalysis_provenance import _feature_build_scope
from quant_rl.eval.rollout import evaluate_model
from quant_rl.evaluation import calculate_metrics
from quant_rl.features.build import build_features
from quant_rl.models.agent import validate_model_action_schema, validate_model_observation_schema
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


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    value = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_sessions=int(result.get("n_sessions", 1)),
        n_breach_sessions=int(result.get("n_breach_sessions", 0)),
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
        "reward_sum",
        "reward_attribution",
        "n_steps",
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
    model = eval_run._load_checkpoint(checkpoint, legacy_layout_key=args.legacy_action_layout)
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
    eval_bars, eval_features, split_role = select_split(
        args.split, train_bars, test_bars, train_features, test_features
    )
    primary_ticks = build_tick_books(cfg).get(primary_symbol)
    eval_tickbook = ticks_covering(primary_ticks, eval_bars)
    eval_common, _, _, _, _ = _build_eval_common(cfg, args=None, env_vae=None, pre_ny_by_date=None)
    frame_result = evaluate_model(
        model,
        bars=eval_bars,
        features=eval_features,
        tickbook=eval_tickbook,
        deterministic=True,
        **eval_common,
    )

    strategy = _strategy_from_cfg(cfg)[0]
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
    frame_metrics = _metrics(frame_result)
    bundle_metrics = _metrics(bundle_result)
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
        "checkpoint_timesteps": int(model.num_timesteps),
        "attempt_provenance": str(provenance_path),
        "attempt_provenance_sha256": _sha256(provenance_path),
        "config_sha256": _sha256(run_dir / "config.yaml"),
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
