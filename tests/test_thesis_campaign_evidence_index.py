"""Tests for the provenance-first thesis campaign evidence index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.research.thesis.evidence.build_campaign_evidence_index import (
    build_campaign_evidence_index,
    write_index,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_index_hashes_campaign_artifacts_and_verifies_resume_checkpoint(tmp_path: Path) -> None:
    root = tmp_path
    campaign = root / "outputs" / "gpu_campaigns" / "campaign-test"
    attempt = campaign / "runs" / "attempts" / "variant__seed50" / "attempt4"
    checkpoint = (
        campaign
        / "runs"
        / "attempts"
        / "variant__seed50"
        / "attempt3"
        / "model"
        / "ppo_ckpt_12000000_steps.zip"
    )
    attempt.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (attempt / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "attempt": 4,
                "campaign_fingerprint": "a" * 64,
                "run_key": "variant__seed50",
                "resume_checkpoint": str(checkpoint),
                "resume_checkpoint_sha256": _sha256(checkpoint),
                "resume_from_attempt": 3,
                "resume_origin": "previous_attempt",
            }
        )
    )
    worker_log = campaign / "logs" / "workers" / "variant__seed50__attempt4.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("worker exited during infrastructure recovery\n")
    manifest_path = campaign / "campaign.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": "a" * 64,
                "git_sha": "b" * 40,
                "timesteps": 20_000_000,
                "source_hashes": {},
                "jobs": [
                    {
                        "key": "variant__seed50",
                        "variant": "variant",
                        "seed": 50,
                        "attempts": 4,
                        "status": "failed",
                    }
                ],
            }
        )
    )

    index = build_campaign_evidence_index(campaign)
    run = index["runs"][0]

    assert run["classification"] == "failed_attempt_with_hash_verified_resume_candidate"
    assert run["attempts"][3]["resume_checkpoint"]["file_hash_verified"] is True
    assert run["full_split_exposure"] == "not_verified_by_this_index"
    assert any(
        artifact["path"] == "outputs/gpu_campaigns/campaign-test/logs/workers/"
        "variant__seed50__attempt4.log"
        for artifact in index["campaign_artifacts"]
    )

    output = campaign / "reports" / "thesis_evidence_index.json"
    write_index(index, output)
    assert json.loads(output.read_text())["schema"] == "aalto.campaign_evidence_index.v1"


def test_index_does_not_call_changed_resume_checkpoint_verified(tmp_path: Path) -> None:
    campaign = tmp_path / "outputs" / "gpu_campaigns" / "campaign-test"
    attempt = campaign / "runs" / "attempts" / "variant__seed50" / "attempt2"
    checkpoint = campaign / "runs" / "attempts" / "variant__seed50" / "attempt1" / "model.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"changed")
    attempt.mkdir(parents=True)
    (attempt / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "resume_checkpoint": str(checkpoint),
                "resume_checkpoint_sha256": "0" * 64,
            }
        )
    )
    manifest = campaign / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "source_hashes": {},
                "timesteps": 1,
                "jobs": [
                    {
                        "key": "variant__seed50",
                        "attempts": 2,
                        "status": "failed",
                    }
                ],
            }
        )
    )

    run = build_campaign_evidence_index(campaign)["runs"][0]

    assert run["classification"] == "failed_attempt_with_unverified_resume_candidate"
    assert run["attempts"][1]["resume_checkpoint"]["file_hash_verified"] is False


def test_success_classification_requires_budget_and_final_checkpoint(tmp_path: Path) -> None:
    campaign = tmp_path / "outputs" / "gpu_campaigns" / "campaign-test"
    attempt = campaign / "runs" / "attempts" / "variant__seed50" / "attempt1"
    model_dir = attempt / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "ppo_final.zip").write_bytes(b"final-checkpoint")
    (attempt / "training_log.json").write_text(
        json.dumps({"training_protocol": {"completed_steps": 1000}})
    )
    manifest = campaign / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "source_hashes": {},
                "timesteps": 1000,
                "jobs": [
                    {
                        "key": "variant__seed50",
                        "attempts": 1,
                        "status": "succeeded",
                    }
                ],
            }
        )
    )

    run = build_campaign_evidence_index(campaign)["runs"][0]

    assert run["classification"] == "completed_requested_budget_with_final_checkpoint"
    assert run["latest_checkpoint_by_step"]["path"].endswith("model/ppo_final.zip")
