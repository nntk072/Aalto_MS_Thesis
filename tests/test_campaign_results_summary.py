from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.matrix.summarize_campaign_results import (
    _pnl_reconciliation,
    build_summary,
    render_markdown,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_campaign_summary_reads_metrics_logs_and_legacy_gpu_samples(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    run_key = "variant__seed42"
    attempt_dir = campaign_dir / "runs" / "attempts" / run_key / "attempt1"
    run_dir = attempt_dir / run_key / "20261014_000000_run"
    log_path = campaign_dir / "logs" / "workers" / f"{run_key}.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "[test/report-only] Sharpe=-0.5 MaxDD=1.00% Trades=2 Return=-0.10% "
        "survived=True days=2/130\n"
        "[train/in-sample] Sharpe=-0.2 MaxDD=2.00% Trades=3 Return=-0.20% "
        "survived=True days=3/259\n",
        encoding="utf-8",
    )
    _write_json(
        campaign_dir / "campaign.json",
        {
            "created": "2026-10-14T00:00:00+00:00",
            "job_id": "1234",
            "git_sha": "abc",
            "fingerprint": "f" * 64,
            "timesteps": 100_000,
            "data_source": "bundle",
            "n_envs": 64,
            "worker_cpus": 6,
            "worker_gpus_per_node": 2,
            "worker_nodes": ["gpuarm2"],
            "evaluation": {
                "locked_oos_role": "selection_acknowledged",
                "locked_oos_exposure": "previously_inspected_holdout",
            },
            "jobs": [
                {
                    "key": run_key,
                    "variant": "variant",
                    "seed": 42,
                    "status": "failed",
                    "exit_code": 1,
                    "failure_reason": "training equity gate failed",
                    "attempts": 1,
                    "log": str(log_path),
                }
            ],
        },
    )
    _write_json(
        campaign_dir / "reports" / "campaign_status.json",
        {"timestamp": "2026-10-14T01:00:00+00:00", "statuses": {"failed": 1}},
    )
    _write_json(attempt_dir / "attempt_provenance.json", {"run_key": run_key})
    _write_json(
        run_dir / "training" / "metrics.json",
        {"total_return_pct": -0.2, "total_pnl": -200.0, "n_trades": 3},
    )
    _write_json(
        run_dir / "testing" / "metrics.json",
        {
            "total_return_pct": -0.1,
            "total_pnl": -100.0,
            "max_drawdown": 0.01,
            "sharpe": -0.5,
            "profit_factor": 0.8,
            "n_trades": 2,
        },
    )
    _write_json(
        run_dir / "train_equity_gate.json", {"ok": False, "reason": "max_loss_stopped_trading"}
    )
    episode_path = run_dir / "model" / "episode_equity.csv"
    episode_path.parent.mkdir(parents=True)
    episode_path.write_text(
        "environment_index,episode_steps,completed_data_split,episode_end_reason\n"
        "0,99504,true,data_split_end\n"
        "0,21000,false,risk_limit\n"
        "1,99504,true,data_split_end\n",
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "account:\n  currency: USD\n  initial_balance: 100000.0\n", encoding="utf-8"
    )
    for split, equity in (("training", 99_800.0), ("testing", 99_900.0)):
        equity_path = run_dir / split / "equity.csv"
        equity_path.parent.mkdir(parents=True, exist_ok=True)
        equity_path.write_text(",equity\n2026-01-01,100000.0\n2026-01-02," + str(equity) + "\n")

    resource_path = campaign_dir / "logs" / "resources_gpuarm2.csv"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    with resource_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_utc",
                "process_tree_pss_bytes",
                "campaign_active",
                "gpu_utilization_pct",
                "gpu_memory_used_mib",
            ]
        )
        writer.writerow(["2026-10-14T00:30:00Z", 1024**3, 1, 70, "30000\n80, 31000"])

    summary = build_summary(campaign_dir)
    report = render_markdown(summary)

    job = summary["jobs"][0]
    assert summary["gate_failure_count"] == 1
    assert summary["train_gate_failure_reason_counts"] == {"max_loss_stopped_trading": 1}
    assert summary["test_survived_low_activity_count"] == 1
    assert summary["pnl_reconciliation_counts"] == {
        "verified": 2,
        "mismatch": 0,
        "unavailable": 0,
    }
    assert job["account_currency"] == "USD"
    assert job["train_pnl_reconciliation"]["status"] == "verified"
    assert job["test_pnl_reconciliation"]["equity_delta"] == -100.0
    assert job["evidence_sha256"]["training/equity.csv"] is not None
    assert job["evidence_sha256"]["testing/equity.csv"] is not None
    assert job["evidence_sha256"]["episode_equity.csv"] is not None
    assert job["training_episode_coverage"]["full_split_episode_count"] == 2
    assert job["training_episode_coverage"]["environments_with_full_split_episode"] == 2
    assert summary["training_episode_coverage_summary"]["telemetry_available_runs"] == 1
    assert summary["training_episode_coverage_summary"]["full_split_episode_count"] == 2
    assert job["test_replay_report_only"]["days_traded"] == 2
    assert summary["resource_profiles"][0]["gpu_devices"][1]["mean_utilization_pct"] == 80
    assert "Train survival | Full-year episodes | Envs reached year end" in report
    assert "Completed full-training-split episodes: **2** across **2 environments**" in report
    assert (
        "[episode coverage](../runs/attempts/variant__seed42/attempt1/variant__seed42/20261014_000000_run/model/episode_equity.csv)"
        in report
    )
    assert "PnL (USD)" in report
    assert "[worker log](../logs/workers/variant__seed42.log)" in report
    assert (
        "[run files](../runs/attempts/variant__seed42/attempt1/variant__seed42/20261014_000000_run)"
        in report
    )
    assert "2 verified**, **0 mismatched**, **0 unavailable" in report
    assert "Gate failure reasons (overlapping): `max_loss_stopped_trading` 1/1" in report
    assert "[the source-fidelity audit](source_fidelity_audit.md)" in report
    assert "reconstructed from the sampler's raw two-row" in report


def test_pnl_reconciliation_exposes_saved_equity_mismatch(tmp_path: Path) -> None:
    split_dir = tmp_path / "testing"
    split_dir.mkdir()
    (split_dir / "equity.csv").write_text(",equity\n2026-01-01,100000.0\n2026-01-02,99900.0\n")

    result = _pnl_reconciliation(
        tmp_path,
        "testing",
        {"total_pnl": -101.0},
        100_000.0,
    )

    assert result["status"] == "mismatch"
    assert result["difference"] == 1.0


def test_episode_coverage_reports_missing_and_legacy_telemetry(tmp_path: Path) -> None:
    from scripts.matrix.summarize_campaign_results import _episode_coverage

    assert _episode_coverage(None)["status"] == "missing"
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("environment_index,episode_steps\n0,123\n", encoding="utf-8")
    result = _episode_coverage(legacy)
    assert result["status"] == "legacy_schema"
    assert result["full_split_episode_count"] == 0
    assert "completed_data_split" in result["missing_fields"]


def test_episode_coverage_exposes_unknown_and_invalid_rows(tmp_path: Path) -> None:
    from scripts.matrix.summarize_campaign_results import _episode_coverage

    path = tmp_path / "episodes.csv"
    path.write_text(
        "environment_index,episode_steps,completed_data_split,episode_end_reason\n"
        "0,100,true,data_split_end\n"
        "1,not-an-int,unknown,unknown\n"
        "bad,20,false,risk_limit\n",
        encoding="utf-8",
    )
    result = _episode_coverage(path)
    assert result["status"] == "available"
    assert result["episode_count"] == 3
    assert result["full_split_episode_count"] == 1
    assert result["completion_unknown_count"] == 1
    assert result["invalid_episode_rows"] == 2
