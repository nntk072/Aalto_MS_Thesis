from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts.thesis.verify_numerical_report import (
    _PARITY_COMPARISONS,
    _backend_parity_audit_errors,
    _deleted_dirty_source_record,
    _deleted_source_records,
    _evaluation_provenance_errors,
    _same_resolved_config,
    _source_manifest_errors,
    verify_split,
)

from quant_rl.eval.export import save_run
from quant_rl.evaluation import calculate_metrics


def test_deleted_dirty_source_provenance_uses_the_recorded_git_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.thesis.verify_numerical_report.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "b" * 40 + "\n", ""),
    )

    record = _deleted_dirty_source_record(
        {"path": "data/deleted.csv"}, "a" * 40, repo_root=tmp_path
    )

    assert record == {
        "path": "data/deleted.csv",
        "base_commit": "a" * 40,
        "base_blob_oid": "b" * 40,
        "disposition": "deleted_from_recorded_head",
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "data/deleted.csv").write_text("present", encoding="utf-8")
    assert (
        _deleted_dirty_source_record({"path": "data/deleted.csv"}, "a" * 40, repo_root=tmp_path)
        is None
    )
    assert _deleted_dirty_source_record({"path": "../outside.csv"}, "a" * 40) is None
    monkeypatch.setattr(
        "scripts.thesis.verify_numerical_report.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "c" * 40 + "\n", ""),
    )
    records = _deleted_source_records(
        {
            "source": {
                "head": "a" * 40,
                "dirty_files": [{"path": "data/not-present-in-head.csv"}],
            }
        }
    )
    assert records == [
        {
            "path": "data/not-present-in-head.csv",
            "base_commit": "a" * 40,
            "base_blob_oid": "c" * 40,
            "disposition": "deleted_from_recorded_head",
        }
    ]


def _write_backend_parity_report(tmp_path: Path, split: str, **overrides: Any) -> Path:
    source_run = tmp_path / "campaign/runs/attempts/variant/attempt1/run"
    source_run.mkdir(parents=True, exist_ok=True)
    provenance_path = source_run.parent / "attempt_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "attempt": 1,
                "campaign_id": "campaign",
                "campaign_fingerprint": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    datasets = {"train_bars": {"sha256": "f" * 64}}
    metrics_protocol = {
        "annualization": {"periods_per_year": 10, "convention": "test"},
        "initial_balance": 100.0,
        "frequency": "bar_close_equity",
        "exposure_measure": "test_exposure",
        "execution_seed": 50,
    }
    (source_run / "run_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": {"path": "/campaign/model/checkpoint.zip", "sha256": "c" * 64},
                "config": {"sha256": "d" * 64},
                "feature_cache": {"file_sha256": "e" * 64},
                "training_protocol": {"completed_steps": 2048, "data_source": "bundle"},
                "metrics_protocol": metrics_protocol,
                "split": {"train_end": "2025-12-31", "test_start": "2026-01-01"},
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "status": "passed",
        "artifact_role": "historical_reanalysis_backend_parity",
        "checkpoint_sha256": "c" * 64,
        "checkpoint": "/campaign/model/checkpoint.zip",
        "source_run": str(source_run),
        "campaign_id": "campaign",
        "campaign_fingerprint": "a" * 64,
        "source_attempt": 1,
        "attempt_provenance": str(provenance_path.resolve()),
        "attempt_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "config_sha256": "d" * 64,
        "feature_cache_sha256": "e" * 64,
        "checkpoint_timesteps": 2048,
        "training_data_source": "bundle",
        "execution_seed": 50,
        "metrics_protocol": metrics_protocol,
        "source_datasets": datasets,
        "evaluated_datasets": datasets,
        "dataset_parity_disposition": "verified",
        "checkpoint_manifest_hash_disposition": "verified",
        "backends": ["frames", "shared_bundle"],
        "comparisons": dict.fromkeys(_PARITY_COMPARISONS, True),
        "split": {
            "name": split,
            "role": (
                "in_sample_training"
                if split == "train"
                else "previously_inspected_selection_acknowledged_test"
            ),
            "train_end": "2025-12-31",
            "test_start": "2026-01-01",
        },
    }
    report.update(overrides)
    if "split_role" in overrides:
        report["split"]["role"] = overrides["split_role"]
    path = tmp_path / f"{split}_parity.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_cross_backend_numerical_report_requires_matching_train_and_test_parity(
    tmp_path: Path,
) -> None:
    manifest = {
        "backend_parity_disposition": "cross_backend_requires_checkpoint_parity",
        "training_data_source": "bundle",
        "evaluation_backend": "frame",
        "source_run": str(tmp_path / "campaign/runs/attempts/variant/attempt1/run"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint": "/campaign/model/checkpoint.zip",
        "source_config_sha256": "d" * 64,
        "source_feature_cache": {"file_sha256": "e" * 64},
        "source_datasets": {"train_bars": {"sha256": "f" * 64}},
        "evaluated_datasets": {"train_bars": {"sha256": "f" * 64}},
        "execution_seed": 50,
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "metrics_protocol": {
            "annualization": {"periods_per_year": 10, "convention": "test"},
            "initial_balance": 100.0,
            "frequency": "bar_close_equity",
            "exposure_measure": "test_exposure",
        },
    }
    train = _write_backend_parity_report(tmp_path, "train")
    test = _write_backend_parity_report(tmp_path, "test")

    errors, records = _backend_parity_audit_errors(manifest, [train, test])

    assert errors == []
    assert len(records) == 2
    assert all(len(record["sha256"]) == 64 for record in records)


def test_cross_backend_numerical_report_rejects_unbound_parity(tmp_path: Path) -> None:
    manifest = {
        "backend_parity_disposition": "cross_backend_requires_checkpoint_parity",
        "training_data_source": "bundle",
        "evaluation_backend": "frame",
        "source_run": str(tmp_path / "campaign/runs/attempts/variant/attempt1/run"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint": "/campaign/model/checkpoint.zip",
        "source_config_sha256": "d" * 64,
        "source_feature_cache": {"file_sha256": "e" * 64},
    }
    train = _write_backend_parity_report(tmp_path, "train")
    test = _write_backend_parity_report(tmp_path, "test", checkpoint_sha256="f" * 64)

    errors, _ = _backend_parity_audit_errors(manifest, [train, test])

    assert "test backend parity checkpoint differs from the numerical report" in errors


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"checkpoint_timesteps": 4096}, "step count differs from its source run"),
        ({"execution_seed": 51}, "execution seed differs from numerical report"),
        (
            {"source_datasets": {"train_bars": {"sha256": "0" * 64}}},
            "source datasets differ from numerical report",
        ),
        ({"comparisons": {"equity": True}}, "comparisons are not all equal"),
        (
            {"split_role": "previously_inspected_selection_acknowledged_test"},
            "split role is incorrect",
        ),
    ],
)
def test_cross_backend_numerical_report_rejects_unmatched_parity_provenance(
    tmp_path: Path, override: dict[str, Any], expected_error: str
) -> None:
    manifest = {
        "backend_parity_disposition": "cross_backend_requires_checkpoint_parity",
        "training_data_source": "bundle",
        "evaluation_backend": "frame",
        "source_run": str(tmp_path / "campaign/runs/attempts/variant/attempt1/run"),
        "checkpoint_sha256": "c" * 64,
        "checkpoint": "/campaign/model/checkpoint.zip",
        "source_config_sha256": "d" * 64,
        "source_feature_cache": {"file_sha256": "e" * 64},
        "source_datasets": {"train_bars": {"sha256": "f" * 64}},
        "evaluated_datasets": {"train_bars": {"sha256": "f" * 64}},
        "execution_seed": 50,
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "metrics_protocol": {
            "annualization": {"periods_per_year": 10, "convention": "test"},
            "initial_balance": 100.0,
            "frequency": "bar_close_equity",
            "exposure_measure": "test_exposure",
        },
    }
    train = _write_backend_parity_report(tmp_path, "train", **override)
    test = _write_backend_parity_report(tmp_path, "test")

    errors, _ = _backend_parity_audit_errors(manifest, [train, test])

    assert any(expected_error in error for error in errors)


def test_same_backend_numerical_report_does_not_need_external_parity(tmp_path: Path) -> None:
    manifest = {
        "backend_parity_disposition": "same_backend",
        "training_data_source": "frames",
        "evaluation_backend": "frame",
    }

    errors, records = _backend_parity_audit_errors(manifest, None)

    assert errors == []
    assert records == []


def test_evaluation_provenance_requires_source_hashes_and_runtime_versions() -> None:
    provenance: dict[str, Any] = {
        "source": {
            "available": True,
            "head": "a" * 40,
            "dirty_files": [{"path": "quant_rl/eval/eval_run.py", "sha256": "b" * 64}],
        },
        "runtime": {
            "python": "3.12.0",
            "platform": "test-platform",
            "numpy": "2.0.0",
            "pandas": "2.2.0",
            "torch": "2.7.1",
            "stable-baselines3": "2.8.0",
            "gymnasium": "1.1.0",
            "python_hash_seed_env": None,
        },
        "python_executable": "/venv/bin/python",
    }

    assert _evaluation_provenance_errors(provenance) == []

    provenance["runtime"].pop("python_hash_seed_env")
    assert any(
        "PYTHONHASHSEED record" in error for error in _evaluation_provenance_errors(provenance)
    )
    provenance["runtime"]["python_hash_seed_env"] = None
    provenance["runtime"]["python_hash_seed_env"] = 12
    assert any(
        "PYTHONHASHSEED record" in error for error in _evaluation_provenance_errors(provenance)
    )
    provenance["runtime"]["python_hash_seed_env"] = None

    provenance["source"]["dirty_files"][0].pop("sha256")
    errors = _evaluation_provenance_errors(provenance)
    assert any("dirty-source records are malformed" in error for error in errors)


def _write_split(tmp_path: Path) -> Path:
    equity = pd.Series([100.0, 110.0, 115.0], index=pd.RangeIndex(3))
    trades = pd.DataFrame(
        [
            {
                "type": "open",
                "fill_quote_source": "bar_open",
                "position_id": "p1",
                "direction": 1,
                "time": "2025-01-01T00:00:00",
                "price": 10.0,
            },
            {
                "type": "close",
                "fill_quote_source": "bar_close",
                "position_id": "p1",
                "direction": 1,
                "time": "2025-01-01T00:02:00",
                "price": 11.5,
                "pnl": 15.0,
                "position_complete": True,
            },
        ]
    )
    result = {
        "equity": equity,
        "trades": trades,
        "initial_balance": 100.0,
        "n_steps": 2,
        "exposure_bars": 2,
        "exposure_fraction": 1.0,
        "exposure_measure": "fraction_of_evaluation_steps_with_position_at_bar_close",
        "n_sessions": 1,
        "n_breach_sessions": 0,
        "survived_full_year": True,
        "completed_data_split": True,
        "episode_end_reason": "data_end",
        "fail_time": None,
        "action_counts": {"hold": 2},
    }
    metrics = calculate_metrics(
        equity,
        trades=trades,
        n_sessions=1,
        initial_balance=100.0,
        periods_per_year=10,
    )
    run_dir = save_run(
        run_dir=tmp_path / "report",
        train_result=result,
        train_metrics=metrics,
        save_plots=False,
        save_html=False,
        save_csv=True,
    )
    return run_dir / "training"


def _add_q5_cost_evidence(split_dir: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "step": 1,
                "time": "2025-01-01T00:01:00",
                "discount_factor": 1.0,
                "daily_cost": 0.1,
                "max_loss_cost": 0.0,
                "trailing_cost": 0.0,
                "breach_indicator": 0.0,
                "step_cost": 0.1,
                "discounted_step_cost": 0.1,
            },
            {
                "step": 2,
                "time": "2025-01-01T00:02:00",
                "discount_factor": 0.9,
                "daily_cost": 0.0,
                "max_loss_cost": 0.0,
                "trailing_cost": 0.2,
                "breach_indicator": 0.0,
                "step_cost": 0.2,
                "discounted_step_cost": 0.18,
            },
        ]
    )
    rows.to_csv(split_dir / "constraint_costs.csv", index=False)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["constraint_cost_summary"] = {
        "schema": "q5_discounted_constraint_cost_v1",
        "definition": "per_step_max_of_normalized_daily_max_loss_trailing_costs_and_breach_indicator",
        "discount_gamma": 0.9,
        "n_steps": 2,
        "undiscounted_total": 0.3,
        "discounted_total": 0.28,
        "components": {
            "daily_cost": {"undiscounted_total": 0.1, "discounted_total": 0.1},
            "max_loss_cost": {"undiscounted_total": 0.0, "discounted_total": 0.0},
            "trailing_cost": {"undiscounted_total": 0.2, "discounted_total": 0.18},
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")


def test_verify_split_reconciles_exported_raw_artifacts(tmp_path):
    split_dir = _write_split(tmp_path)
    breach_events = pd.read_csv(split_dir / "breach_events.csv")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert errors == []
    assert list(breach_events.columns) == [
        "time",
        "session_id",
        "reason",
        "equity",
        "fill_time",
    ]
    assert breach_events.empty


def test_verify_split_reconciles_q5_discounted_constraint_costs(tmp_path: Path) -> None:
    split_dir = _write_split(tmp_path)
    _add_q5_cost_evidence(split_dir)

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert errors == []
    costs = pd.read_csv(split_dir / "constraint_costs.csv")
    costs.loc[1, "discount_factor"] = 0.8
    costs.to_csv(split_dir / "constraint_costs.csv", index=False)
    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)
    assert any("discount factors do not reconcile" in error for error in errors)


def test_verify_split_reconciles_fill_quote_source_counts(tmp_path: Path) -> None:
    split_dir = _write_split(tmp_path)
    assert verify_split(split_dir, initial_balance=100.0, periods_per_year=10) == []

    fills = pd.read_csv(split_dir / "fills.csv")
    fills.loc[0, "fill_quote_source"] = "tick"
    fills.to_csv(split_dir / "fills.csv", index=False)
    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)
    assert any("fill quote source counts differ from fills.csv" in error for error in errors)


def test_reanalysis_config_comparison_checks_resolved_values():
    from omegaconf import OmegaConf

    source = OmegaConf.create({"data": {"primary": "US100.cash"}, "ppo": {"gamma": 0.99}})
    equivalent = OmegaConf.create({"data": {"primary": "US100.cash"}, "ppo": {"gamma": 0.99}})
    changed = OmegaConf.create({"data": {"primary": "US500.cash"}, "ppo": {"gamma": 0.99}})

    assert _same_resolved_config(source, equivalent)
    assert not _same_resolved_config(source, changed)


def test_verify_split_detects_modified_position_export(tmp_path):
    split_dir = _write_split(tmp_path)
    positions = pd.read_csv(split_dir / "positions.csv")
    positions.loc[0, "entry_price"] = 999.0
    positions.to_csv(split_dir / "positions.csv", index=False)

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("positions.csv does not reconcile" in error for error in errors)


def test_verify_split_detects_modified_metric_endpoint(tmp_path):
    split_dir = _write_split(tmp_path)
    metrics_path = split_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final_equity"] = 999.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("final_equity" in error for error in errors)


def test_verify_split_checks_equity_length_against_step_count(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["n_steps"] = 1
    diagnostics["exposure_bars"] = 1
    diagnostics["exposure_fraction"] = 1.0
    diagnostics["action_counts"] = {"hold": 1}
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("expected the initial balance plus one point per step" in error for error in errors)


def test_verify_split_rejects_empty_evaluation_split(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["n_steps"] = 0
    diagnostics["n_sessions"] = 0
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("diagnostics n_steps must be positive" in error for error in errors)
    assert any("diagnostics n_sessions must be positive" in error for error in errors)


def test_verify_split_rejects_infinite_equity_values(tmp_path):
    split_dir = _write_split(tmp_path)
    equity_path = split_dir / "equity.csv"
    equity = pd.read_csv(equity_path, index_col=0)
    equity.loc[1, "equity"] = float("inf")
    equity.to_csv(equity_path)

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("equity.csv is empty or contains non-finite values" in error for error in errors)


def test_verify_split_rejects_survival_flag_without_matching_breach_records(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["survived_full_year"] = False
    diagnostics["fail_time"] = "2025-01-01 00:00:00"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("survival flag disagrees" in error for error in errors)
    assert any("fail_time presence disagrees" in error for error in errors)


def test_verify_split_rejects_survival_flag_for_incomplete_data_split(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["completed_data_split"] = False
    diagnostics["episode_end_reason"] = "max_episode_steps"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any(
        "survival flag disagrees with breach events and split completion" in error
        for error in errors
    )


def test_verify_split_rejects_completion_reason_mismatch(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["episode_end_reason"] = "max_episode_steps"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("split-completion flag disagrees with end reason" in error for error in errors)


def test_verify_split_rejects_boolean_as_numeric_metric(tmp_path):
    split_dir = _write_split(tmp_path)
    metrics_path = split_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final_equity"] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("final_equity" in error for error in errors)


def test_verify_split_rejects_numeric_string_metric(tmp_path):
    split_dir = _write_split(tmp_path)
    metrics_path = split_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final_equity"] = "115.0"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("final_equity" in error for error in errors)


def test_verify_split_reports_missing_diagnostics_without_crashing(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics.pop("n_sessions")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert errors == ["training: diagnostics missing ['n_sessions']"]


def test_verify_split_checks_action_counts_against_evaluation_steps(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["action_counts"] = {"hold": 1}
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("action counts do not sum to n_steps" in error for error in errors)


def test_verify_split_reports_malformed_exposure_bars_without_crashing(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["exposure_bars"] = "two"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("exposure_bars must be a nonnegative integer" in error for error in errors)


def test_verify_split_rejects_exposure_fraction_outside_unit_interval(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["exposure_fraction"] = 1.5
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("exposure_fraction must be finite and within [0, 1]" in error for error in errors)


def test_verify_split_reports_malformed_breach_count_without_crashing(tmp_path):
    split_dir = _write_split(tmp_path)
    diagnostics_path = split_dir / "evaluation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["n_breach_sessions"] = None
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("n_breach_sessions must be a nonnegative integer" in error for error in errors)


def test_verify_split_checks_activity_label_against_completed_trades(tmp_path):
    split_dir = _write_split(tmp_path)
    metrics_path = split_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["activity_status"] = "INACTIVE"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    errors = verify_split(split_dir, initial_balance=100.0, periods_per_year=10)

    assert any("activity status" in error for error in errors)


def test_source_manifest_records_must_match_reanalysis_manifest():
    dataset_record = {"sha256": "d" * 64}
    cache_record = {
        "identity": "features.parquet",
        "sha256": "a" * 16,
        "file_sha256": "b" * 64,
    }
    source_manifest = {
        "config": {"sha256": "c" * 64},
        "feature_cache": cache_record,
        "datasets": {"train_bars": dataset_record},
        "checkpoint": {"sha256": "e" * 64},
        "execution_protocol": {"data_source": "frames"},
        "metrics_protocol": {"execution_seed": 50},
        "action_schema": {
            "version": "overlay-action-v2",
            "key": "5",
            "width": 5,
            "coordinates": ["intensity", "stop", "risk", "target", "exit"],
        },
        "observation_schema": {"version": "account-state-v1"},
        "stop_candidate_schema": {"schema_version": "rq1-stop-menu-v1"},
    }
    reanalysis_manifest: dict[str, Any] = {
        "source_feature_cache": cache_record.copy(),
        "source_datasets": {"train_bars": dataset_record.copy()},
        "checkpoint_manifest_hash_disposition": "verified",
        "training_data_source": "frames",
        "execution_seed": 50,
        "action_schema": {
            "version": "overlay-action-v2",
            "key": "5",
            "width": 5,
            "coordinates": ["intensity", "stop", "risk", "target", "exit"],
            "checkpoint_metadata": "versioned",
        },
        "action_schema_version": "overlay-action-v2",
        "action_layout_key": "5",
        "observation_schema": {"version": "account-state-v1"},
        "stop_candidate_schema": {"schema_version": "rq1-stop-menu-v1"},
        "stop_schema_disposition": "versioned",
        "legacy_action_schema_opt_in": False,
        "legacy_observation_schema_opt_in": False,
        "legacy_stop_schema_opt_in": False,
    }

    assert (
        _source_manifest_errors(
            source_manifest,
            reanalysis_manifest,
            source_config_sha256="c" * 64,
            checkpoint_sha256="e" * 64,
        )
        == []
    )

    wrong_seed_manifest = reanalysis_manifest.copy()
    wrong_seed_manifest["execution_seed"] = 51
    errors = _source_manifest_errors(
        source_manifest,
        wrong_seed_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("execution seed differs" in error for error in errors)

    missing_action_schema = source_manifest.copy()
    missing_action_schema.pop("action_schema")
    missing_replay_schema = reanalysis_manifest.copy()
    missing_replay_schema.pop("action_schema")
    errors = _source_manifest_errors(
        missing_action_schema,
        missing_replay_schema,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("original run manifest action_schema is missing" in error for error in errors)
    assert any("reanalysis action_schema is missing" in error for error in errors)

    reanalysis_manifest["source_datasets"]["train_bars"]["sha256"] = "f" * 64
    errors = _source_manifest_errors(
        source_manifest,
        reanalysis_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("source dataset SHA-256" in error for error in errors)

    reanalysis_manifest["source_datasets"]["train_bars"]["sha256"] = "d" * 64
    reanalysis_manifest["observation_schema"] = {"version": "account-state-v2"}
    errors = _source_manifest_errors(
        source_manifest,
        reanalysis_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("observation_schema differs" in error for error in errors)

    reanalysis_manifest["observation_schema"] = source_manifest["observation_schema"]
    reanalysis_manifest["action_schema"]["key"] = "6_dir"
    reanalysis_manifest["action_layout_key"] = "6_dir"
    errors = _source_manifest_errors(
        source_manifest,
        reanalysis_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("action_schema differs" in error for error in errors)

    reanalysis_manifest["action_schema"]["key"] = "5"
    reanalysis_manifest["action_layout_key"] = "6_dir"
    errors = _source_manifest_errors(
        source_manifest,
        reanalysis_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("action schema summary fields disagree" in error for error in errors)

    reanalysis_manifest["action_layout_key"] = "5"
    reanalysis_manifest["stop_schema_disposition"] = "legacy_opt_in"
    reanalysis_manifest["legacy_stop_schema_opt_in"] = True
    errors = _source_manifest_errors(
        source_manifest,
        reanalysis_manifest,
        source_config_sha256="c" * 64,
        checkpoint_sha256="e" * 64,
    )
    assert any("legacy opt-in" in error for error in errors)
