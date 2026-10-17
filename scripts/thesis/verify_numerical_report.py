#!/usr/bin/env python3
"""Verify a deterministic checkpoint report against its exported raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from quant_rl.eval.export import _fill_events, _position_summaries
from quant_rl.evaluation import calculate_metrics, evaluation_periods_per_year

_SPLITS = ("training", "testing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _same_resolved_config(left: DictConfig, right: DictConfig) -> bool:
    """Compare saved source and reanalysis configs after resolving OmegaConf values."""
    try:
        return OmegaConf.to_container(left, resolve=True) == OmegaConf.to_container(
            right, resolve=True
        )
    except (TypeError, ValueError):
        return False


def _same_number(actual: Any, expected: Any) -> bool:
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not isinstance(expected, (int, float))
        or isinstance(expected, bool)
    ):
        return False
    try:
        left, right = float(actual), float(expected)
    except (OverflowError, TypeError, ValueError):
        return False
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _constraint_cost_errors(split_dir: Path, diagnostics: dict[str, Any]) -> list[str]:
    """Reconcile optional Q5 cost summaries against their per-step source rows."""
    errors: list[str] = []
    artifact_path = split_dir / "constraint_costs.csv"
    summary = diagnostics.get("constraint_cost_summary")
    if summary is None:
        if artifact_path.exists():
            errors.append(f"{split_dir.name}: constraint_costs.csv lacks a summary record")
        return errors
    if not isinstance(summary, dict):
        return [f"{split_dir.name}: constraint_cost_summary must be an object"]
    if summary.get("schema") != "q5_discounted_constraint_cost_v1":
        errors.append(f"{split_dir.name}: constraint-cost schema is unsupported")
    if summary.get("definition") != (
        "per_step_max_of_normalized_daily_max_loss_trailing_costs_and_breach_indicator"
    ):
        errors.append(f"{split_dir.name}: constraint-cost definition is unsupported")
    if not artifact_path.is_file():
        return [f"{split_dir.name}: constraint_costs.csv is missing"]

    frame = pd.read_csv(artifact_path)
    columns = [
        "step",
        "time",
        "discount_factor",
        "daily_cost",
        "max_loss_cost",
        "trailing_cost",
        "breach_indicator",
        "step_cost",
        "discounted_step_cost",
    ]
    if list(frame.columns) != columns:
        return [f"{split_dir.name}: constraint_costs.csv has an unexpected schema"]
    expected_steps = diagnostics.get("n_steps")
    if len(frame) != expected_steps or summary.get("n_steps") != expected_steps:
        errors.append(f"{split_dir.name}: constraint cost row count differs from n_steps")
    if frame["time"].isna().any() or frame["time"].astype(str).str.len().eq(0).any():
        errors.append(f"{split_dir.name}: constraint-cost timestamps are missing")

    numeric_columns = [key for key in columns if key != "time"]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return [*errors, f"{split_dir.name}: constraint-cost rows contain non-finite values"]
    if (numeric[numeric_columns[2:]] < 0.0).any().any():
        errors.append(f"{split_dir.name}: constraint-cost values must be nonnegative")
    if not numeric["step"].eq(np.arange(1, len(frame) + 1)).all():
        errors.append(f"{split_dir.name}: constraint-cost step sequence is invalid")

    gamma = summary.get("discount_gamma")
    gamma_number = (
        float(gamma) if isinstance(gamma, (int, float)) and not isinstance(gamma, bool) else None
    )
    if gamma_number is None or not math.isfinite(gamma_number) or not 0.0 < gamma_number <= 1.0:
        return [*errors, f"{split_dir.name}: constraint-cost discount gamma is invalid"]
    expected_discount = np.power(gamma_number, np.arange(len(frame), dtype=float))
    if not np.allclose(numeric["discount_factor"], expected_discount, rtol=1e-10, atol=1e-12):
        errors.append(f"{split_dir.name}: constraint-cost discount factors do not reconcile")
    expected_step_cost = numeric[
        ["daily_cost", "max_loss_cost", "trailing_cost", "breach_indicator"]
    ].max(axis=1)
    if not np.allclose(numeric["step_cost"], expected_step_cost, rtol=1e-10, atol=1e-12):
        errors.append(f"{split_dir.name}: per-step constraint cost does not reconcile")
    expected_discounted_cost = numeric["step_cost"] * numeric["discount_factor"]
    if not np.allclose(
        numeric["discounted_step_cost"], expected_discounted_cost, rtol=1e-10, atol=1e-12
    ):
        errors.append(f"{split_dir.name}: discounted per-step cost does not reconcile")
    for key, column in (
        ("undiscounted_total", "step_cost"),
        ("discounted_total", "discounted_step_cost"),
    ):
        if not _same_number(summary.get(key), float(numeric[column].sum())):
            errors.append(f"{split_dir.name}: constraint-cost {key} does not reconcile")
    components = summary.get("components")
    if not isinstance(components, dict):
        errors.append(f"{split_dir.name}: constraint-cost component summary is missing")
    else:
        for key in ("daily_cost", "max_loss_cost", "trailing_cost"):
            component = components.get(key)
            if not isinstance(component, dict):
                errors.append(f"{split_dir.name}: constraint-cost component {key} is missing")
                continue
            for total_key, actual in (
                ("undiscounted_total", float(numeric[key].sum())),
                (
                    "discounted_total",
                    float((numeric[key] * numeric["discount_factor"]).sum()),
                ),
            ):
                if not _same_number(component.get(total_key), actual):
                    errors.append(
                        f"{split_dir.name}: constraint-cost {key} {total_key} does not reconcile"
                    )
    return errors


def verify_split(
    split_dir: Path,
    *,
    initial_balance: float,
    periods_per_year: int,
) -> list[str]:
    """Recompute a split's metrics and check its raw fill/position diagnostics."""
    errors: list[str] = []
    required = (
        "equity.csv",
        "trades.csv",
        "fills.csv",
        "positions.csv",
        "breach_events.csv",
        "metrics.json",
        "evaluation_diagnostics.json",
        "session_activity.json",
    )
    missing = [name for name in required if not (split_dir / name).is_file()]
    if missing:
        return [f"{split_dir.name}: missing required artifacts {missing}"]

    equity_frame = pd.read_csv(split_dir / "equity.csv", index_col=0)
    if "equity" not in equity_frame.columns:
        return [f"{split_dir.name}: equity.csv must contain an equity column"]
    equity = pd.to_numeric(equity_frame["equity"], errors="coerce")
    trades = pd.read_csv(split_dir / "trades.csv")
    fills = pd.read_csv(split_dir / "fills.csv")
    positions = pd.read_csv(split_dir / "positions.csv")
    breach_events = pd.read_csv(split_dir / "breach_events.csv")
    metrics_report = _read_json(split_dir / "metrics.json")
    diagnostics = _read_json(split_dir / "evaluation_diagnostics.json")
    session_activity = _read_json(split_dir / "session_activity.json")

    required_diagnostics = {
        "n_steps",
        "exposure_bars",
        "exposure_fraction",
        "exposure_measure",
        "n_sessions",
        "n_breach_sessions",
        "survived_full_year",
        "completed_data_split",
        "episode_end_reason",
        "fail_time",
        "action_counts",
        "execution_event_count",
        "fill_quote_source_event_count",
        "fill_quote_source_counts",
        "position_record_count",
        "identified_position_count",
        "unidentified_position_event_count",
    }
    missing_diagnostics = sorted(required_diagnostics.difference(diagnostics))
    if missing_diagnostics:
        return [f"{split_dir.name}: diagnostics missing {missing_diagnostics}"]

    for key in (
        "n_steps",
        "exposure_bars",
        "n_sessions",
        "n_breach_sessions",
        "execution_event_count",
        "fill_quote_source_event_count",
        "position_record_count",
        "identified_position_count",
        "unidentified_position_event_count",
    ):
        if not _is_nonnegative_int(diagnostics[key]):
            errors.append(f"{split_dir.name}: diagnostics {key} must be a nonnegative integer")
    for key in ("n_steps", "n_sessions"):
        if _is_nonnegative_int(diagnostics[key]) and diagnostics[key] == 0:
            errors.append(f"{split_dir.name}: diagnostics {key} must be positive")
    if (
        _is_nonnegative_int(diagnostics["n_breach_sessions"])
        and _is_nonnegative_int(diagnostics["n_sessions"])
        and diagnostics["n_breach_sessions"] > diagnostics["n_sessions"]
    ):
        errors.append(f"{split_dir.name}: breach-session count exceeds session count")
    if not isinstance(diagnostics["survived_full_year"], bool):
        errors.append(f"{split_dir.name}: survived_full_year must be a boolean")
    if not isinstance(diagnostics["completed_data_split"], bool):
        errors.append(f"{split_dir.name}: completed_data_split must be a boolean")
    if (
        not isinstance(diagnostics["episode_end_reason"], str)
        or not diagnostics["episode_end_reason"]
    ):
        errors.append(f"{split_dir.name}: episode_end_reason must be a non-empty string")
    elif diagnostics["episode_end_reason"] not in {
        "data_end",
        "max_episode_steps",
        "risk_breach",
        "terminated",
        "truncated",
    }:
        errors.append(f"{split_dir.name}: episode_end_reason is not recognized")
    if isinstance(diagnostics["completed_data_split"], bool) and isinstance(
        diagnostics["episode_end_reason"], str
    ):
        if diagnostics["completed_data_split"] != (diagnostics["episode_end_reason"] == "data_end"):
            errors.append(f"{split_dir.name}: split-completion flag disagrees with end reason")
    if diagnostics["fail_time"] is not None and not isinstance(diagnostics["fail_time"], str):
        errors.append(f"{split_dir.name}: fail_time must be null or a string")
    if list(breach_events.columns) != [
        "time",
        "session_id",
        "reason",
        "equity",
        "fill_time",
    ]:
        errors.append(f"{split_dir.name}: breach_events.csv has an unexpected schema")
    elif not breach_events.empty:
        if breach_events["session_id"].isna().any():
            errors.append(f"{split_dir.name}: breach event is missing session_id")
        if (
            breach_events["reason"].isna().any()
            or (breach_events["reason"].astype(str).str.len() == 0).any()
        ):
            errors.append(f"{split_dir.name}: breach event is missing reason")
        event_equity = pd.to_numeric(breach_events["equity"], errors="coerce")
        if event_equity.isna().any() or not event_equity.map(math.isfinite).all():
            errors.append(f"{split_dir.name}: breach event equity is not finite")
        expected_fail_time = str(breach_events.iloc[0]["time"])
        if diagnostics["fail_time"] != expected_fail_time:
            errors.append(f"{split_dir.name}: fail_time differs from first breach event")
    expected_survival = breach_events.empty and diagnostics["completed_data_split"]
    if diagnostics["survived_full_year"] != expected_survival:
        errors.append(
            f"{split_dir.name}: survival flag disagrees with breach events and split completion"
        )
    if (diagnostics["fail_time"] is None) != breach_events.empty:
        errors.append(f"{split_dir.name}: fail_time presence disagrees with breach events")
    action_counts = diagnostics["action_counts"]
    if not isinstance(action_counts, dict) or any(
        not isinstance(key, str) or not _is_nonnegative_int(count)
        for key, count in action_counts.items()
    ):
        errors.append(f"{split_dir.name}: action_counts must map strings to nonnegative integers")
    elif sum(action_counts.values()) != diagnostics["n_steps"]:
        errors.append(f"{split_dir.name}: action counts do not sum to n_steps")
    errors.extend(_constraint_cost_errors(split_dir, diagnostics))
    if errors:
        return errors

    if int(breach_events["session_id"].nunique()) != diagnostics["n_breach_sessions"]:
        errors.append(f"{split_dir.name}: breach-session count does not match breach_events.csv")
    for key in (
        "n_sessions",
        "n_breach_sessions",
        "n_sessions_with_trades",
        "n_sessions_skipped",
    ):
        if key not in session_activity or not _is_nonnegative_int(session_activity[key]):
            errors.append(f"{split_dir.name}: session_activity {key} is invalid")
        elif session_activity[key] != diagnostics.get(key):
            errors.append(f"{split_dir.name}: session_activity {key} differs from diagnostics")
    if (
        _is_nonnegative_int(session_activity.get("n_sessions_with_trades"))
        and _is_nonnegative_int(session_activity.get("n_sessions_skipped"))
        and _is_nonnegative_int(session_activity.get("n_sessions"))
        and session_activity["n_sessions_with_trades"] + session_activity["n_sessions_skipped"]
        != session_activity["n_sessions"]
    ):
        errors.append(f"{split_dir.name}: session activity counts do not sum to n_sessions")
    if errors:
        return errors

    expected_fills = _fill_events(trades)
    expected_positions = _position_summaries(trades)
    if len(fills) != len(expected_fills):
        errors.append(
            f"{split_dir.name}: fills.csv has {len(fills)} rows; "
            f"trades.csv reconstructs {len(expected_fills)}"
        )
    if len(positions) != len(expected_positions):
        errors.append(
            f"{split_dir.name}: positions.csv has {len(positions)} rows; "
            f"trades.csv reconstructs {len(expected_positions)}"
        )
    for name, exported, reconstructed in (
        ("fills.csv", fills, expected_fills),
        ("positions.csv", positions, expected_positions),
    ):
        try:
            pd.testing.assert_frame_equal(
                exported.reset_index(drop=True),
                reconstructed.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-8,
            )
        except AssertionError as exc:
            errors.append(f"{split_dir.name}: {name} does not reconcile to trades.csv: {exc}")
    identified_count = (
        int(positions["reconstruction_status"].eq("identified_by_position_id").sum())
        if "reconstruction_status" in positions.columns
        else -1
    )
    unidentified_count = (
        int(
            positions["reconstruction_status"]
            .astype(str)
            .str.startswith("unidentified_event_")
            .sum()
        )
        if "reconstruction_status" in positions.columns
        else -1
    )
    expected_counts = {
        "execution_event_count": len(expected_fills),
        "position_record_count": len(expected_positions),
        "identified_position_count": int(
            expected_positions["reconstruction_status"].eq("identified_by_position_id").sum()
        )
        if not expected_positions.empty
        else 0,
        "unidentified_position_event_count": int(
            expected_positions["reconstruction_status"]
            .astype(str)
            .str.startswith("unidentified_event_")
            .sum()
        )
        if not expected_positions.empty
        else 0,
    }
    actual_counts = {
        "execution_event_count": len(fills),
        "position_record_count": len(positions),
        "identified_position_count": identified_count,
        "unidentified_position_event_count": unidentified_count,
    }
    for key, expected in expected_counts.items():
        if actual_counts[key] != expected or diagnostics.get(key) != expected:
            errors.append(
                f"{split_dir.name}: {key} mismatch "
                f"(diagnostics={diagnostics.get(key)}, artifacts={actual_counts[key]}, "
                f"reconstructed={expected})"
            )

    source_counts = diagnostics.get("fill_quote_source_counts")
    allowed_fill_sources = {
        "tick",
        "bar_open",
        "bar_close",
        "bar_range_level",
        "permitted_session_bar_close",
        "terminal_bar_close",
    }
    if not isinstance(source_counts, dict) or any(
        not isinstance(source, str)
        or source not in allowed_fill_sources
        or not _is_nonnegative_int(count)
        for source, count in (source_counts.items() if isinstance(source_counts, dict) else [])
    ):
        errors.append(
            f"{split_dir.name}: fill_quote_source_counts must map recognized sources to counts"
        )
    elif (
        not _is_nonnegative_int(diagnostics.get("fill_quote_source_event_count"))
        or sum(source_counts.values()) != diagnostics["fill_quote_source_event_count"]
    ):
        errors.append(f"{split_dir.name}: fill quote source counts do not sum to their event count")
    if "fill_quote_source" not in fills.columns:
        errors.append(f"{split_dir.name}: fills.csv is missing fill_quote_source")
    else:
        actual_source_counts = {
            str(source): int(count)
            for source, count in fills["fill_quote_source"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .items()
        }
        if actual_source_counts != source_counts:
            errors.append(f"{split_dir.name}: fill quote source counts differ from fills.csv")
    if diagnostics.get("fill_quote_source_event_count") != diagnostics.get("execution_event_count"):
        errors.append(
            f"{split_dir.name}: fill quote source event count differs from execution event count"
        )

    n_steps_value = diagnostics.get("n_steps")
    n_steps = cast(int, n_steps_value) if _is_nonnegative_int(n_steps_value) else 0
    exposure_bars_value = diagnostics.get("exposure_bars")
    exposure_bars = (
        cast(int, exposure_bars_value) if _is_nonnegative_int(exposure_bars_value) else -1
    )
    exposure_fraction = diagnostics.get("exposure_fraction")
    exposure_fraction_number = (
        float(exposure_fraction)
        if isinstance(exposure_fraction, (int, float)) and not isinstance(exposure_fraction, bool)
        else None
    )
    if not 0 <= exposure_bars <= n_steps:
        errors.append(f"{split_dir.name}: exposure_bars is outside [0, n_steps]")
    if (
        exposure_fraction_number is None
        or not math.isfinite(exposure_fraction_number)
        or not (0.0 <= exposure_fraction_number <= 1.0)
    ):
        errors.append(f"{split_dir.name}: exposure_fraction must be finite and within [0, 1]")
    elif _is_nonnegative_int(exposure_bars) and not _same_number(
        exposure_fraction_number, exposure_bars / n_steps if n_steps else 0.0
    ):
        errors.append(f"{split_dir.name}: exposure fraction does not match exposure bars")
    if diagnostics.get("exposure_measure") != (
        "fraction_of_evaluation_steps_with_position_at_bar_close"
    ):
        errors.append(f"{split_dir.name}: unknown exposure measurement definition")

    if equity.empty or equity.isna().any() or not equity.map(math.isfinite).all():
        errors.append(f"{split_dir.name}: equity.csv is empty or contains non-finite values")
        return errors
    if len(equity) != n_steps + 1:
        errors.append(
            f"{split_dir.name}: equity.csv has {len(equity)} points for {n_steps} "
            "evaluation steps; expected the initial balance plus one point per step"
        )
    recomputed = calculate_metrics(
        equity,
        trades=trades,
        n_sessions=int(diagnostics.get("n_sessions", 0)),
        n_breach_sessions=int(diagnostics.get("n_breach_sessions", 0)),
        initial_balance=initial_balance,
        periods_per_year=periods_per_year,
    )
    expected_metrics = asdict(recomputed)
    for key, expected in expected_metrics.items():
        if key not in metrics_report:
            errors.append(f"{split_dir.name}: metrics.json missing {key}")
        elif key == "extras":
            actual_extras = metrics_report[key]
            if not isinstance(actual_extras, dict):
                errors.append(f"{split_dir.name}: metrics extras is not an object")
            else:
                for extra_key, extra_value in expected.items():
                    if extra_key not in actual_extras or not _same_number(
                        actual_extras[extra_key], extra_value
                    ):
                        errors.append(
                            f"{split_dir.name}: metric extra {extra_key} does not reconcile"
                        )
        elif not _same_number(metrics_report[key], expected):
            errors.append(f"{split_dir.name}: metric {key} does not reconcile to raw artifacts")
    expected_activity = "ACTIVE" if recomputed.n_trades > 0 else "INACTIVE"
    if metrics_report.get("activity_status") != expected_activity:
        errors.append(f"{split_dir.name}: activity status does not match completed trades")

    final_equity = float(equity.iloc[-1])
    if not _same_number(metrics_report.get("initial_balance"), initial_balance):
        errors.append(f"{split_dir.name}: reported initial balance differs from config")
    if not _same_number(metrics_report.get("final_equity"), final_equity):
        errors.append(f"{split_dir.name}: metric final_equity differs from equity.csv")
    if not _same_number(final_equity - initial_balance, metrics_report.get("total_pnl")):
        errors.append(f"{split_dir.name}: PnL does not equal final equity minus initial balance")
    return errors


def _source_manifest_errors(
    source_manifest: dict[str, Any],
    reanalysis_manifest: dict[str, Any],
    *,
    source_config_sha256: Any,
    checkpoint_sha256: Any,
) -> list[str]:
    """Ensure copied provenance records still match the original run manifest."""
    errors: list[str] = []
    source_config = source_manifest.get("config")
    if not isinstance(source_config, dict) or source_config.get("sha256") != source_config_sha256:
        errors.append("source config SHA-256 differs from original run manifest")

    original_cache = source_manifest.get("feature_cache")
    replay_cache = reanalysis_manifest.get("source_feature_cache")
    if not isinstance(original_cache, dict) or not isinstance(replay_cache, dict):
        errors.append("source feature-cache record is missing from an original manifest")
    else:
        for key in ("identity", "sha256", "file_sha256"):
            if original_cache.get(key) != replay_cache.get(key):
                errors.append(f"source feature-cache {key} differs from original run manifest")

    original_datasets = source_manifest.get("datasets")
    replay_datasets = reanalysis_manifest.get("source_datasets")
    if not isinstance(original_datasets, dict) or not isinstance(replay_datasets, dict):
        errors.append("source dataset records are missing from an original manifest")
    elif set(original_datasets) != set(replay_datasets):
        errors.append("reanalysis source dataset set differs from original run manifest")
    else:
        for dataset_name in sorted(original_datasets):
            original_record = original_datasets[dataset_name]
            replay_record = replay_datasets[dataset_name]
            if (
                not isinstance(original_record, dict)
                or not isinstance(replay_record, dict)
                or original_record.get("sha256") != replay_record.get("sha256")
            ):
                errors.append(
                    f"source dataset SHA-256 differs from original run manifest: {dataset_name}"
                )

    original_checkpoint = source_manifest.get("checkpoint")
    if (
        reanalysis_manifest.get("checkpoint_manifest_hash_disposition") != "verified"
        or not isinstance(original_checkpoint, dict)
        or original_checkpoint.get("sha256") != checkpoint_sha256
    ):
        errors.append("checkpoint SHA-256 differs from original run manifest")

    schema_keys = ("action_schema", "observation_schema", "stop_candidate_schema")
    for schema_key in schema_keys:
        original_schema = source_manifest.get(schema_key)
        evaluated_schema = reanalysis_manifest.get(schema_key)
        original_contract: dict[str, Any] | None = None
        evaluated_contract: dict[str, Any] | None = None
        if not isinstance(original_schema, dict):
            errors.append(f"original run manifest {schema_key} is missing")
        if schema_key == "action_schema":
            if not isinstance(evaluated_schema, dict):
                errors.append("reanalysis action_schema is missing")
                continue
            if not isinstance(original_schema, dict):
                continue
            original_contract = original_schema
            evaluated_contract = {
                key: value
                for key, value in evaluated_schema.items()
                if key != "checkpoint_metadata"
            }
            if evaluated_schema.get("checkpoint_metadata") != "versioned":
                errors.append("reanalysis action schema metadata is not versioned")
            if reanalysis_manifest.get("action_schema_version") != evaluated_schema.get(
                "version"
            ) or reanalysis_manifest.get("action_layout_key") != evaluated_schema.get("key"):
                errors.append("reanalysis action schema summary fields disagree")
        else:
            if isinstance(original_schema, dict):
                original_contract = original_schema
            if isinstance(evaluated_schema, dict):
                evaluated_contract = evaluated_schema
        if original_contract is None or evaluated_contract is None:
            continue
        if original_contract != evaluated_contract:
            errors.append(f"reanalysis {schema_key} differs from original run manifest")
    if reanalysis_manifest.get("stop_schema_disposition") != "versioned":
        errors.append("stop-candidate schema was evaluated through a legacy opt-in")
    for opt_in_key in (
        "legacy_action_schema_opt_in",
        "legacy_observation_schema_opt_in",
        "legacy_stop_schema_opt_in",
    ):
        if reanalysis_manifest.get(opt_in_key) is not False:
            errors.append(f"reanalysis used an unsupported legacy schema opt-in: {opt_in_key}")

    original_protocol = source_manifest.get("execution_protocol")
    if not isinstance(original_protocol, dict) or original_protocol.get(
        "data_source"
    ) != reanalysis_manifest.get("training_data_source"):
        errors.append("training backend differs from original run manifest")
    return errors


def _evaluation_provenance_errors(value: Any) -> list[str]:
    """Require source and runtime identities for the replay that made a report."""
    if not isinstance(value, dict):
        return ["reanalysis manifest lacks evaluation_provenance"]
    errors: list[str] = []
    source_record = value.get("source")
    runtime_record = value.get("runtime")
    if (
        not isinstance(source_record, dict)
        or source_record.get("available") is not True
        or not isinstance(source_record.get("head"), str)
        or len(source_record["head"]) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_record["head"].lower())
        or not isinstance(source_record.get("dirty_files"), list)
    ):
        errors.append("reanalysis source provenance is incomplete")
    elif any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not item.get("path")
        or not isinstance(item.get("sha256"), str)
        or len(item["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in item["sha256"].lower())
        for item in source_record["dirty_files"]
    ):
        errors.append("reanalysis dirty-source records are malformed")
    required_versions = (
        "python",
        "platform",
        "numpy",
        "pandas",
        "torch",
        "stable-baselines3",
        "gymnasium",
    )
    if (
        not isinstance(runtime_record, dict)
        or any(
            not isinstance(runtime_record.get(name), str) or not runtime_record[name]
            for name in required_versions
        )
        or not isinstance(value.get("python_executable"), str)
        or not value.get("python_executable")
    ):
        errors.append("reanalysis runtime provenance is incomplete")
    return errors


def verify_report_directory(run_dir: Path) -> dict[str, Any]:
    """Validate provenance gates and numerical artifacts for both evaluation splits."""
    errors: list[str] = []
    manifest_path = run_dir / "reanalysis_manifest.json"
    timing_path = run_dir / "report_export_timing.json"
    config_path = run_dir / "config.yaml"
    if not manifest_path.is_file() or not timing_path.is_file() or not config_path.is_file():
        missing = [
            path.name for path in (manifest_path, timing_path, config_path) if not path.is_file()
        ]
        return {"status": "failed", "errors": [f"missing root artifacts {missing}"]}

    manifest = _read_json(manifest_path)
    export_record = _read_json(timing_path)
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        errors.append("config.yaml is not a valid mapping")
    report_config_hash = manifest.get("reanalysis_config_sha256")
    if not isinstance(report_config_hash, str) or not report_config_hash:
        errors.append("reanalysis manifest is missing reanalysis_config_sha256")
    elif _sha256(config_path) != report_config_hash:
        errors.append("reanalysis config SHA-256 does not match its manifest")
    if manifest.get("artifact_role") != "historical_reanalysis":
        errors.append("reanalysis manifest has the wrong artifact role")
    if manifest.get("historical_holdout_inspected") is not True:
        errors.append("historical holdout exposure is not explicitly recorded")
    if manifest.get("policy_action_mode") != "deterministic":
        errors.append("policy action mode is not deterministic")
    errors.extend(_evaluation_provenance_errors(manifest.get("evaluation_provenance")))
    if manifest.get("feature_cache_disposition") not in {
        "verified",
        "source_cache_replay_verified",
    }:
        errors.append("feature-cache provenance is not verified")
    source_cache = manifest.get("source_feature_cache")
    if not isinstance(source_cache, dict):
        errors.append("reanalysis manifest lacks the source feature-cache record")
    else:
        cache_pairs = (
            ("identity", manifest.get("feature_cache_identity")),
            ("sha256", manifest.get("feature_cache_hash")),
            ("file_sha256", manifest.get("feature_cache_file_sha256")),
        )
        for key, evaluated_value in cache_pairs:
            if (
                not isinstance(source_cache.get(key), str)
                or source_cache.get(key) != evaluated_value
            ):
                errors.append(f"source/evaluated feature-cache {key} values do not match")

    source_datasets = manifest.get("source_datasets")
    evaluated_datasets = manifest.get("evaluated_datasets")
    if (
        manifest.get("dataset_parity_disposition") != "verified"
        or not isinstance(source_datasets, dict)
        or not isinstance(evaluated_datasets, dict)
    ):
        errors.append("train/test dataset provenance is not verified")
    elif set(source_datasets) != set(evaluated_datasets):
        errors.append("source/evaluated train/test dataset names differ")
    else:
        for dataset_name in sorted(source_datasets):
            source_record = source_datasets[dataset_name]
            evaluated_record = evaluated_datasets[dataset_name]
            if (
                not isinstance(source_record, dict)
                or not isinstance(evaluated_record, dict)
                or not isinstance(source_record.get("sha256"), str)
                or source_record.get("sha256") != evaluated_record.get("sha256")
            ):
                errors.append(f"source/evaluated dataset SHA-256 does not match: {dataset_name}")
    if manifest.get("backend_parity_disposition") != "same_backend":
        errors.append("training/evaluation backend parity requires a separate audit")
    if manifest.get("training_data_source") not in {"frames", "frame"}:
        errors.append("training backend is not a verified frame backend")
    if manifest.get("evaluation_backend") != "frame":
        errors.append("evaluation backend is not the frame backend")
    if export_record.get("save_csv") is not True:
        errors.append("report export did not retain mandatory CSV artifacts")

    checkpoint_value = manifest.get("checkpoint")
    checkpoint_hash = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        errors.append("reanalysis manifest is missing the checkpoint path")
    else:
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            errors.append(f"checkpoint is missing: {checkpoint}")
        elif _sha256(checkpoint) != checkpoint_hash:
            errors.append("checkpoint SHA-256 does not match reanalysis manifest")
    source_config_hash = manifest.get("source_config_sha256")
    source_run_value = manifest.get("source_run")
    if not isinstance(source_run_value, str) or not source_run_value:
        errors.append("reanalysis manifest is missing source_run")
    else:
        source_run = Path(source_run_value)
        source_config = source_run / "config.yaml"
        if not source_config.is_file():
            errors.append(f"source config is missing: {source_config}")
        elif _sha256(source_config) != source_config_hash:
            errors.append("source config SHA-256 does not match reanalysis manifest")
        else:
            try:
                original_config = OmegaConf.load(source_config)
                if (
                    not isinstance(cfg, DictConfig)
                    or not isinstance(original_config, DictConfig)
                    or not _same_resolved_config(original_config, cfg)
                ):
                    errors.append("reanalysis config does not match the resolved source-run config")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"cannot compare source and reanalysis configs: {exc}")
        source_manifest_path = source_run / "run_manifest.json"
        if not source_manifest_path.is_file():
            errors.append(f"source run manifest is missing: {source_manifest_path}")
        else:
            source_manifest = _read_json(source_manifest_path)
            errors.extend(
                _source_manifest_errors(
                    source_manifest,
                    manifest,
                    source_config_sha256=source_config_hash,
                    checkpoint_sha256=checkpoint_hash,
                )
            )

    try:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved_cfg, dict):
            raise TypeError("resolved config must be an object")
        initial_balance = float(cfg.account.initial_balance)
        periods_per_year = evaluation_periods_per_year(cast(dict[str, Any], resolved_cfg))
        evaluation_config = resolved_cfg.get("evaluation", {})
        if not isinstance(evaluation_config, dict):
            evaluation_config = {}
        annualization_convention = str(
            evaluation_config.get(
                "annualization_convention",
                "legacy M1 convention: 252 sessions/year x 390 bars/session",
            )
        )
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"cannot resolve metric protocol from config: {exc}")
        initial_balance, periods_per_year = 0.0, 0
        annualization_convention = ""
    metric_protocol = manifest.get("metrics_protocol")
    if not isinstance(metric_protocol, dict):
        errors.append("reanalysis manifest lacks metrics_protocol")
    else:
        annualization = metric_protocol.get("annualization")
        if not isinstance(annualization, dict):
            errors.append("metrics_protocol lacks annualization")
        else:
            if annualization.get("periods_per_year") != periods_per_year:
                errors.append("metrics protocol periods/year differs from config")
            if annualization.get("convention") != annualization_convention:
                errors.append("metrics protocol annualization convention differs from config")
        if not _same_number(metric_protocol.get("initial_balance"), initial_balance):
            errors.append("metrics protocol initial balance differs from config")
        if metric_protocol.get("exposure_measure") != (
            "fraction_of_evaluation_steps_with_position_at_bar_close"
        ):
            errors.append("metrics protocol has an unknown exposure definition")
        if metric_protocol.get("frequency") != "bar_close_equity":
            errors.append("metrics protocol has an unknown equity frequency")

    for split in _SPLITS:
        errors.extend(
            verify_split(
                run_dir / split,
                initial_balance=initial_balance,
                periods_per_year=periods_per_year,
            )
        )
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "evidence_class": "historical_exploratory_not_confirmatory",
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Historical deterministic reanalysis directory")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        report = verify_report_directory(run_dir)
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "evidence_class": "historical_exploratory_not_confirmatory",
            "run_dir": str(run_dir),
            "errors": [f"verification could not complete: {type(exc).__name__}: {exc}"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
