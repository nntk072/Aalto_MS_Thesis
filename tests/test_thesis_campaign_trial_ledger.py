from __future__ import annotations

import json
from pathlib import Path

from scripts.thesis.build_campaign_trial_ledger import build_trial_ledger


def _campaign(tmp_path: Path, *, attempts: int = 2) -> Path:
    campaign = tmp_path / "campaign-a"
    (campaign / "runs" / "attempts").mkdir(parents=True)
    (campaign / "reports").mkdir()
    manifest = {
        "fingerprint": "f" * 64,
        "git_sha": "a" * 40,
        "job_id": 1234,
        "timesteps": 20_000_000,
        "n_envs": 64,
        "data_inputs": {"bars.csv": "b" * 64},
        "jobs": [
            {
                "key": "variant__seed50",
                "variant": "variant",
                "seed": 50,
                "attempts": attempts,
                "status": "running",
                "started": "2026-10-16T00:00:00+00:00",
            }
        ],
    }
    (campaign / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")
    attempt = campaign / "runs/attempts/variant__seed50/attempt1"
    run_dir = attempt / "variant__seed50/run1"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    (attempt / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "started": "2026-10-16T00:00:00+00:00",
                "slurm_job_id": "7777",
                "identity": {"git_sha": "c" * 40},
            }
        ),
        encoding="utf-8",
    )
    (attempt / "variant__seed50/resolved_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "ppo_ckpt_10000000_steps.zip").write_bytes(b"checkpoint")
    second_attempt = campaign / "runs/attempts/variant__seed50/attempt2"
    second_attempt.mkdir(parents=True)
    (second_attempt / "attempt_provenance.json").write_text(
        json.dumps({"slurm_job_id": "5678", "identity": {"git_sha": "c" * 40}}),
        encoding="utf-8",
    )
    return campaign


def test_trial_ledger_reconciles_stale_running_state_and_keeps_missing_identity_null(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    ledger = build_trial_ledger(
        [campaign],
        scheduler_states={1234: "COMPLETED", 5678: "FAILED", 7777: "TIMEOUT"},
    )

    assert ledger["campaign_ids"] == ["campaign-a"]
    records = {record["record_id"]: record for record in ledger["records"]}
    run = records["campaign-a::variant__seed50::run"]
    first = records["campaign-a::variant__seed50::attempt1"]
    second = records["campaign-a::variant__seed50::attempt2"]
    assert run["status"] == "interrupted"
    assert run["scheduler_state"] == "FAILED"
    assert run["slurm_job_ids"] == [1234, 5678, 7777]
    assert first["status"] == "interrupted"
    assert first["slurm_job_id"] == 7777
    assert first["scheduler_state"] == "TIMEOUT"
    assert first["completed_timesteps"] == 10_000_000
    assert second["status"] == "interrupted"
    assert second["config_sha256"] is None
    assert second["identity_completeness"] == "partial"


def test_trial_ledger_uses_failure_reconciliation_report_for_attempt_status(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path, attempts=1)
    (campaign / "reports/attempt_infrastructure_reconciliation.json").write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "key": "variant__seed50",
                        "attempts": 1,
                        "classification": "infrastructure_failed_before_training",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ledger = build_trial_ledger([campaign], scheduler_states={1234: "FAILED"})
    attempt = next(record for record in ledger["records"] if record["record_type"] == "attempt")

    assert attempt["status"] == "failed"
    assert attempt["disposition"] == "infrastructure_failed_before_training"
