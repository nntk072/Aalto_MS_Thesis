"""Reanalyse a PPO checkpoint into a separate, immutable run destination.

Historical holdout reanalysis is opt-in and explicitly labelled as previously
inspected; it must never overwrite the original training or testing artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import json_to_data

from quant_rl.action_schema import build_action_schema
from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.split import (
    get_split_config,
    make_train_mask,
    split_bars,
    split_train_test,
)
from quant_rl.data.ticks import ticks_covering
from quant_rl.eval.export import save_run
from quant_rl.eval.reanalysis_provenance import (
    _evaluation_dataset_records,
    _feature_build_scope,
    _load_exact_source_feature_cache,
    _load_run_config,
    _reanalysis_backend_disposition,
    _validate_reanalysis_datasets,
    _validate_reanalysis_feature_cache,
    _validate_reanalysis_request,
)
from quant_rl.eval.rollout import BAR_CLOSE_EXPOSURE_MEASURE, evaluate_model
from quant_rl.eval.run_manifest import (
    build_evaluation_provenance,
    build_sequence_feature_schema,
    build_stop_candidate_schema,
    file_sha256,
    validate_sequence_feature_schema,
)
from quant_rl.evaluation import calculate_metrics, evaluation_periods_per_year
from quant_rl.features.build import (
    build_features,
    select_legacy_obs_columns,
    select_obs_columns,
)
from quant_rl.models.agent import (
    LegacyFourToFiveActionAdapter,
    validate_model_action_schema,
)
from quant_rl.models.encoder import LegacyTCNEncoderV1, TCNEncoder
from quant_rl.models.ppo_policy import (
    ClampedStdMultiInputPolicy,
    LegacyClampedStdMultiInputPolicy,
)
from quant_rl.train.train_rl import (
    _build_eval_common,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    """Return the model checkpoint to evaluate.

    Prefers an explicit ``--checkpoint``, then ``model/ppo_final.zip``
    (written on successful training completion), then the highest-step
    ``ppo_ckpt_*_steps.zip`` (for a run whose training hasn't finished, or
    was interrupted, but has intermediate checkpoints).
    """
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path

    model_dir = run_dir / "model"
    final = model_dir / "ppo_final.zip"
    if final.exists():
        return final

    ckpts = list(model_dir.glob("ppo_ckpt_*_steps.zip"))
    if not ckpts:
        raise SystemExit(f"No model checkpoints found under {model_dir}")

    def _steps(p: Path) -> int:
        m = _CKPT_STEPS_RE.search(p.name)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_steps)
    log.warning("No ppo_final.zip in %s; using latest checkpoint: %s", model_dir, latest.name)
    return latest


def _load_checkpoint(path: Path, *, legacy_layout_key: str | None) -> Any:
    """Load a checkpoint, selecting the archived extractor only by explicit layout."""
    if legacy_layout_key != "legacy4_no_exit_v1":
        return PPO.load(path)
    with ZipFile(path) as archive:
        if "data" not in archive.namelist():
            raise ValueError(f"legacy checkpoint has no SB3 data record: {path}")
        data = json_to_data(archive.read("data").decode("utf-8"))
    action_space = data.get("action_space")
    if getattr(action_space, "shape", None) != (4,):
        raise ValueError("legacy4_no_exit_v1 requires a saved 4D action space")
    policy_kwargs = data.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict):
        raise ValueError("legacy 4D checkpoint has no readable policy kwargs")
    extractor = policy_kwargs.get("features_extractor_class")
    if extractor not in (TCNEncoder, LegacyTCNEncoderV1):
        raise ValueError("legacy4_no_exit_v1 migration is defined only for the archived TCN family")
    migrated_kwargs = dict(policy_kwargs)
    if extractor is TCNEncoder:
        migrated_kwargs["features_extractor_class"] = LegacyTCNEncoderV1
    return PPO.load(
        path,
        device="cpu",
        custom_objects={
            "policy_class": LegacyClampedStdMultiInputPolicy,
            "policy_kwargs": migrated_kwargs,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint-reanalysis CLI with an explicit action mode."""
    parser = argparse.ArgumentParser(
        description="Reanalyse a PPO checkpoint into a separate, new output directory."
    )
    parser.add_argument("--run", required=True, help="Existing source RL run (read-only)")
    parser.add_argument("--output", required=True, help="New, non-existing reanalysis directory")
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Acknowledge that this run's historical test split is already inspected",
    )
    parser.add_argument(
        "--allow-legacy-action-schema",
        action="store_true",
        help="Explicitly opt in to an unversioned checkpoint using the selected legacy layout",
    )
    parser.add_argument(
        "--allow-legacy-observation-schema",
        action="store_true",
        help="Explicitly opt in to an unversioned account-state-v1 checkpoint",
    )
    parser.add_argument(
        "--allow-legacy-sequence-schema",
        action="store_true",
        help=(
            "Explicitly replay a manifest without ordered model-feature coordinates using "
            "the pre-filter sequence columns; the schema remains legacy_unverified"
        ),
    )
    parser.add_argument(
        "--allow-legacy-stop-schema",
        action="store_true",
        help=(
            "Explicitly opt in when an older checkpoint or run manifest has no stop-candidate "
            "schema; this does not assert that it used the expanded multi-HTF menu"
        ),
    )
    parser.add_argument(
        "--allow-legacy-data-provenance",
        action="store_true",
        help=(
            "Explicitly allow reanalysis when the source run lacks feature-cache or dataset "
            "identities; the resulting report will label input parity unverified"
        ),
    )
    parser.add_argument(
        "--source-feature-cache",
        default=None,
        help=(
            "Read-only replay of a preserved training cache; its filename, sidecar content hash, "
            "file SHA-256, saved config, and train/test feature fingerprints must match the run"
        ),
    )
    parser.add_argument(
        "--legacy-action-layout",
        default=None,
        help=(
            "Explicit layout for an unversioned checkpoint; must match the run config "
            "(for example 5, 6_dir, 6_sl, 7, 8, 11). Use legacy4_no_exit_v1 only "
            "for the documented 4D no-exit policy migration."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (relative to --run or absolute); defaults to "
        "model/ppo_final.zip, else the highest-step model/ppo_ckpt_*_steps.zip",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=None,
        help="Cap on per-trade charts per split (default: config output.max_order_charts, else 0).",
    )
    parser.add_argument(
        "--save-html",
        dest="save_html",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Render interactive HTML (default: config output.save_html).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Opt in to sampled policy actions; deterministic actions are the default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed NumPy and PyTorch. This makes --stochastic evaluation repeatable; "
            "without --stochastic, policy actions remain deterministic."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Force the data pipeline to rebuild caches"
    )
    return parser


def _evaluate_checkpoint_split(model: Any, *, stochastic: bool, **kwargs: Any) -> dict[str, Any]:
    """Replay one final split using the explicitly selected policy action mode."""
    return evaluate_model(model, deterministic=not stochastic, **kwargs)


def _resolve_execution_seed(
    requested_seed: int | None, source_manifest: dict[str, Any]
) -> int | None:
    """Use an explicit replay seed or the recorded training seed for fill jitter."""
    if requested_seed is not None:
        if requested_seed < 0:
            raise ValueError("evaluation seed must be non-negative")
        return requested_seed
    training_protocol = source_manifest.get("training_protocol")
    metrics_protocol = source_manifest.get("metrics_protocol")
    candidates = (
        source_manifest.get("seed"),
        training_protocol.get("seed") if isinstance(training_protocol, dict) else None,
        metrics_protocol.get("execution_seed") if isinstance(metrics_protocol, dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return None


def _validate_sequence_input_width(model: Any, expected_width: int) -> int:
    """Require the checkpoint's sequence tensor width to match its feature schema."""
    observation_spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    get_space = getattr(observation_spaces, "get", None)
    sequence_space = get_space("seq") if callable(get_space) else None
    shape = getattr(sequence_space, "shape", None)
    if not shape or int(shape[-1]) != int(expected_width):
        raise ValueError(
            "checkpoint sequence input width disagrees with evaluated feature coordinates: "
            f"checkpoint={shape}, expected={expected_width}"
        )
    return int(shape[-1])


def _validate_stop_candidate_schema(
    *,
    checkpoint_schema: dict[str, Any] | None,
    manifest_schema: dict[str, Any] | None,
    expected_schema: dict[str, Any],
    allow_legacy: bool,
) -> str:
    """Reject checkpoint/config drift and label explicit latest-swing legacy use."""
    if checkpoint_schema is None or manifest_schema is None:
        if not allow_legacy:
            raise ValueError(
                "checkpoint or run manifest has no stop-candidate schema; pass "
                "--allow-legacy-stop-schema only when intentionally reanalysing a legacy run"
            )
        available_schema = checkpoint_schema if checkpoint_schema is not None else manifest_schema
        if available_schema is None:
            if expected_schema.get("menu") != "legacy_last_accepted_swing_v1":
                raise ValueError(
                    "legacy checkpoint has no stop schema and the run config requests the "
                    "expanded multi-timeframe menu; legacy opt-in cannot assert that menu "
                    "was trained"
                )
        elif available_schema != expected_schema:
            raise ValueError(
                "available stop-candidate schema differs from the run config; legacy opt-in "
                "does not authorize evaluating under a changed stop menu"
            )
        return "legacy_opt_in"
    if checkpoint_schema != manifest_schema:
        raise ValueError("checkpoint and run manifest stop-candidate schemas disagree")
    if checkpoint_schema != expected_schema:
        raise ValueError(
            "checkpoint stop-candidate schema differs from the run config; refusing to evaluate "
            "under a changed stop menu"
        )
    return "versioned"


def _validate_checkpoint_manifest_hash(path: Path, manifest: dict[str, Any]) -> str:
    """Verify the checkpoint is the exact artifact recorded by its run manifest."""
    checkpoint_record = manifest.get("checkpoint")
    expected_hash = checkpoint_record.get("sha256") if isinstance(checkpoint_record, dict) else None
    if expected_hash is None:
        return "unavailable_legacy_manifest"
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            "checkpoint SHA-256 does not match the source run manifest; refusing reanalysis"
        )
    return "verified"


def main() -> None:
    args = _build_parser().parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    output_dir = Path(args.output)
    _validate_reanalysis_request(run_dir, output_dir, allow_historical=args.allow_historical)

    if args.seed is not None:
        # Seed all stochastic components in this fresh reanalysis process.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        log.info("Seeded evaluation RNG: %d", args.seed)
    elif args.stochastic:
        log.warning(
            "Stochastic policy actions enabled without --seed; this evaluation "
            "will not reproduce a prior sampled-action replay."
        )
    policy_action_mode = "stochastic" if args.stochastic else "deterministic"
    log.info("Policy action mode: %s", policy_action_mode)

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
    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
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
    source_execution_protocol = source_manifest.get("execution_protocol")
    if (
        source_feature_cache is None or source_datasets is None
    ) and not args.allow_legacy_data_provenance:
        raise ValueError(
            "source run manifest lacks feature-cache or dataset identities; pass "
            "--allow-legacy-data-provenance only for an explicitly unverified legacy replay"
        )
    checkpoint_hash_disposition = _validate_checkpoint_manifest_hash(ckpt_path, source_manifest)
    _ = ClampedStdMultiInputPolicy
    model: Any = _load_checkpoint(ckpt_path, legacy_layout_key=args.legacy_action_layout)
    expected_stop_schema = build_stop_candidate_schema(cfg)
    checkpoint_stop_schema = getattr(model, "stop_candidate_schema", None)
    manifest_stop_schema = source_manifest.get("stop_candidate_schema")
    stop_schema_disposition = _validate_stop_candidate_schema(
        checkpoint_schema=checkpoint_stop_schema,
        manifest_schema=manifest_stop_schema,
        expected_schema=expected_stop_schema,
        allow_legacy=bool(args.allow_legacy_stop_schema),
    )
    from quant_rl.models.agent import validate_model_observation_schema
    from quant_rl.observation_schema import build_observation_schema

    expected_observation_schema = build_observation_schema(
        str(cfg.env.get("observation_schema_version", "account-state-v1")),
        entry_state_visibility=bool(
            cfg.env.get("entry_state_machine", False)
            and cfg.env.get("entry_state_observation", True)
        ),
    )
    validate_model_observation_schema(
        model,
        expected_observation_schema,
        allow_legacy_v1=bool(args.allow_legacy_observation_schema),
    )
    is_overlay = bool(cfg.env.get("strategy_actions", False))
    if is_overlay:
        if not isinstance(model.action_space, spaces.Box):
            raise ValueError("checkpoint action space must be Box for strategy overlay")
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
        migration = validate_model_action_schema(
            model,
            schema,
            allow_legacy=args.allow_legacy_action_schema,
            legacy_layout_key=args.legacy_action_layout,
        )
        if migration == "legacy4_no_exit_v1_to_overlay-action-v2-5":
            model = LegacyFourToFiveActionAdapter(model)
    elif isinstance(model.action_space, spaces.Box):
        raise ValueError("Box checkpoint with non-overlay run config: action semantics unclear")

    data = run_pipeline(cfg, force=args.force)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    train_end, test_start = get_split_config(cfg)
    source_cache_replay_path = None
    if args.source_feature_cache:
        config_record = source_manifest.get("config")
        source_config_path = run_dir / "config.yaml"
        expected_config_hash = (
            config_record.get("sha256") if isinstance(config_record, dict) else None
        )
        if not isinstance(expected_config_hash, str) or not expected_config_hash:
            raise ValueError("exact source-cache replay requires a saved config SHA-256")
        if (
            not source_config_path.is_file()
            or file_sha256(source_config_path) != expected_config_hash
        ):
            raise ValueError("saved run config does not match its manifest SHA-256")
        source_cache_replay_path = Path(args.source_feature_cache).expanduser()
        if not source_cache_replay_path.is_absolute():
            source_cache_replay_path = (
                Path(__file__).resolve().parents[2] / source_cache_replay_path
            )
        train_mask = make_train_mask(pd.DatetimeIndex(primary_m1.index), train_end)
        (
            features,
            feat_cache,
            evaluated_feature_cache_hash,
            evaluated_feature_cache_file_sha256,
        ) = _load_exact_source_feature_cache(source_feature_cache, source_cache_replay_path)
        feature_cache_disposition = "source_cache_replay_verified"
    else:
        train_mask, feat_cache = _feature_build_scope(
            primary_m1, cfg, cache_dir, str(cfg.data.primary), train_end, secondary_m1
        )
        features = build_features(
            primary_m1,
            secondary=secondary_m1,
            cfg=cfg,
            train_mask=train_mask,
            cache_path=feat_cache,
        )
        (
            feature_cache_disposition,
            evaluated_feature_cache_hash,
            evaluated_feature_cache_file_sha256,
        ) = _validate_reanalysis_feature_cache(
            source_feature_cache,
            feat_cache,
            allow_legacy=bool(args.allow_legacy_data_provenance),
        )

    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start)
    primary_ticks = build_tick_books(cfg).get(str(cfg.data.primary))
    evaluated_datasets = _evaluation_dataset_records(
        train_bars=train_bars,
        train_features=train_feat,
        test_bars=test_bars,
        test_features=test_feat,
        train_secondary=train_sec,
        test_secondary=test_sec,
        primary_ticks=primary_ticks,
    )
    dataset_parity_disposition = _validate_reanalysis_datasets(
        source_datasets,
        evaluated_datasets,
        allow_legacy=bool(args.allow_legacy_data_provenance),
    )
    training_data_source, backend_parity_disposition = _reanalysis_backend_disposition(
        source_execution_protocol,
        evaluation_backend="frame",
    )
    log.info(
        "Evaluating train=%d bars (≤%s)  test=%d bars (≥%s) under FTMO eval guardrails",
        len(train_bars),
        train_end,
        len(test_bars),
        test_start,
    )

    # Eval CLI does not expose training --algo / --reward flags; PPO replay
    # uses the configuration's saved environment defaults.
    eval_common, eval_strategy, _, _, _ = _build_eval_common(
        cfg, args=None, env_vae=None, pre_ny_by_date=None
    )
    execution_seed = _resolve_execution_seed(args.seed, source_manifest)
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
    _validate_sequence_input_width(model, expected_sequence_schema["width"])

    test_result = _evaluate_checkpoint_split(
        model,
        stochastic=args.stochastic,
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

    train_result = _evaluate_checkpoint_split(
        model,
        stochastic=args.stochastic,
        bars=train_bars,
        features=train_feat,
        tickbook=ticks_covering(primary_ticks, train_bars),
        legacy_sequence_columns=use_legacy_sequence_columns,
        **eval_common,
    )
    train_result["initial_balance"] = cfg.account.initial_balance
    train_m = calculate_metrics(
        train_result["equity"],
        trades=train_result["trades"],
        n_sessions=train_result.get("n_sessions", 1),
        n_breach_sessions=train_result.get("n_breach_sessions", 0),
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
            "eval_seed": args.seed,
            "eval_execution_seed": execution_seed,
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
        "evaluation_provenance": build_evaluation_provenance(Path(__file__).resolve().parents[2]),
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
        "seed": args.seed,
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
        "reanalysis_timestamp": datetime.now().isoformat(),
        "validity_note": "Historical inspected holdout; not pristine OOS evidence.",
    }
    (output_dir / "reanalysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done. Reanalysis artifacts written to: %s", output_dir)


if __name__ == "__main__":
    main()
