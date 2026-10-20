from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.research.thesis.evidence.reconcile_campaign_attempts import reconcile_campaign


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reconciliation_keeps_attempt_history_and_episode_coverage(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    run_key = "variant__seed50"
    _write_json(
        campaign_dir / "campaign.json",
        {
            "created": "2026-10-16T00:00:00+00:00",
            "git_sha": "a" * 40,
            "fingerprint": "b" * 64,
            "timesteps": 1_000,
            "n_envs": 2,
            "data_source": "bundle",
            "evaluation": {"locked_oos_role": "selection_acknowledged"},
            "jobs": [
                {
                    "key": run_key,
                    "variant": "variant",
                    "seed": 50,
                    "status": "interrupted",
                    "attempts": 2,
                }
            ],
        },
    )
    _write_json(
        campaign_dir / "reports/campaign_status.json",
        {"timestamp": "2026-10-16T01:00:00+00:00", "statuses": {"failed": 1}},
    )

    attempt1 = campaign_dir / "runs/attempts" / run_key / "attempt1"
    _write_json(
        attempt1 / "attempt_provenance.json",
        {"started": "2026-10-16T00:00:00+00:00", "slurm_job_id": 123, "resume_origin": "parent"},
    )
    run_dir = attempt1 / run_key / f"{run_key}_run/model"
    run_dir.mkdir(parents=True)
    (run_dir / "ppo_ckpt_1000_steps.zip").write_bytes(b"checkpoint")
    (run_dir / "episode_equity.csv").write_text(
        "environment_index,episode_steps,completed_data_split,episode_end_reason\n"
        "0,99504,true,data_end\n"
        "1,99504,true,data_end\n",
        encoding="utf-8",
    )
    log1 = campaign_dir / "logs/workers" / f"{run_key}__attempt1.log"
    log1.parent.mkdir(parents=True)
    log1.write_text(f"END key={run_key} exit=0\n", encoding="utf-8")

    attempt2 = campaign_dir / "runs/attempts" / run_key / "attempt2"
    _write_json(
        attempt2 / "attempt_provenance.json",
        {
            "started": "2026-10-16T00:30:00+00:00",
            "slurm_job_id": 456,
            "resume_origin": "interrupted_campaign_attempt",
            "resume_from_attempt": 1,
            "resume_checkpoint": str(run_dir / "ppo_ckpt_1000_steps.zip"),
            "resume_checkpoint_sha256": "c" * 64,
        },
    )
    log2 = campaign_dir / "logs/workers" / f"{run_key}__attempt2.log"
    log2.write_text(
        "Traceback (most recent call last):\n"
        "FileExistsError: [Errno 17] File exists: 'cache'\n"
        f"END key={run_key} exit=1\n",
        encoding="utf-8",
    )

    report = reconcile_campaign(campaign_dir)

    assert report["summary"] == {
        "run_identities": 1,
        "runs_with_budget_checkpoint": 1,
        "runs_with_full_split_episode_telemetry": 1,
        "attempt4_cache_failures_before_training": 0,
        "all_run_attempts_retained": True,
    }
    run = report["runs"][0]
    assert run["manifest_status"] == "interrupted"
    assert len(run["attempts"]) == 2
    assert (
        run["attempts"][0]["training_episode_coverage"]["environments_with_full_split_episode"] == 2
    )
    assert run["attempts"][0]["latest_checkpoint"]["steps"] == 1_000
    assert run["attempts"][1]["provenance"]["resume_from_attempt"] == 1
    assert run["attempts"][1]["worker_log"]["classification"] == (
        "infrastructure_failure_before_training"
    )
    assert run["attempts"][1]["worker_log"]["exit_code"] == 1


def test_reconciliation_does_not_call_partial_checkpoints_complete(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    run_key = "variant__seed7"
    _write_json(
        campaign_dir / "campaign.json",
        {
            "git_sha": "a" * 40,
            "fingerprint": "b" * 64,
            "timesteps": 20_000_000,
            "jobs": [
                {
                    "key": run_key,
                    "status": "interrupted",
                    "attempts": 1,
                }
            ],
        },
    )
    attempt = campaign_dir / "runs/attempts" / run_key / "attempt1"
    _write_json(attempt / "attempt_provenance.json", {"slurm_job_id": 123})
    model_dir = attempt / run_key / f"{run_key}_run/model"
    model_dir.mkdir(parents=True)
    (model_dir / "ppo_ckpt_14000000_steps.zip").write_bytes(b"partial")

    report = reconcile_campaign(campaign_dir)

    assert report["summary"]["runs_with_budget_checkpoint"] == 0
    assert report["runs"][0]["maximum_saved_checkpoint_steps"] == 14_000_000
    assert report["campaign"]["manifest_status_counts"] == {"interrupted": 1}
