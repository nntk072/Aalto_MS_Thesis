"""Tests for preserving campaign identity while repairing cache failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.triton.prepare_campaign_resume import prepare_campaign_resume


def _campaign_fixture(root: Path) -> tuple[Path, bytes]:
    campaign = root / "campaign"
    keys = [f"variant_{index:02d}" for index in range(18)]
    fingerprint = "a" * 64
    lineage = {key: {"checkpoint_sha256": f"{index:064x}"} for index, key in enumerate(keys)}
    jobs = []
    for index, key in enumerate(keys):
        attempt = 4
        attempt_dir = campaign / "runs" / "attempts" / key / f"attempt{attempt}"
        attempt_dir.mkdir(parents=True)
        checkpoint = (
            campaign
            / "runs"
            / "attempts"
            / key
            / "attempt3"
            / "model"
            / "ppo_ckpt_11000000_steps.zip"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        provenance = {
            "attempt": attempt,
            "campaign_fingerprint": fingerprint,
            "resume_checkpoint": str(checkpoint),
            "resume_checkpoint_sha256": checkpoint_hash,
        }
        (attempt_dir / "attempt_provenance.json").write_text(json.dumps(provenance))
        log = campaign / "logs" / "workers" / f"{key}__attempt{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "FileExistsError: [Errno 17] File exists: 'cache'\n"
            "quant_rl/data/pipeline.py run_pipeline\n"
        )
        jobs.append(
            {
                "key": key,
                "attempts": attempt,
                "status": "failed",
                "failure_reason": "infrastructure failure",
            }
        )
    manifest = {
        "fingerprint": fingerprint,
        "initial_checkpoint_lineage": lineage,
        "jobs": jobs,
    }
    manifest_path = campaign / "campaign.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return campaign, manifest_path.read_bytes()


def test_resume_requeues_only_proven_infrastructure_failures(tmp_path: Path) -> None:
    campaign, original = _campaign_fixture(tmp_path)

    result = prepare_campaign_resume(campaign)

    recovered = json.loads((campaign / "campaign.json").read_text())
    assert result["requeued"] == 18
    assert {job["status"] for job in recovered["jobs"]} == {"interrupted"}
    assert {job["resume_from_attempt"] for job in recovered["jobs"]} == {4}
    assert recovered["fingerprint"] == "a" * 64
    assert (
        recovered["initial_checkpoint_lineage"]
        == json.loads(original)["initial_checkpoint_lineage"]
    )
    backup = campaign / "reports" / "campaign_manifest_before_cache_recovery.json"
    assert backup.read_bytes() == original
    report = json.loads((campaign / "reports" / "cache_recovery.json").read_text())
    assert report["identity_changed"] is False
    assert len(report["requeued_identities"]) == 18
    after_recovery = (campaign / "campaign.json").read_bytes()
    second_result = prepare_campaign_resume(campaign)
    assert second_result["requeued"] == 0
    assert second_result["already_resumable"] is True
    assert (campaign / "campaign.json").read_bytes() == after_recovery


def test_resume_refuses_unknown_failure_without_mutating_manifest(tmp_path: Path) -> None:
    campaign, original = _campaign_fixture(tmp_path)
    log = campaign / "logs" / "workers" / "variant_17__attempt4.log"
    log.write_text("ValueError: strategy failure\n")

    with pytest.raises(ValueError, match="known cache infrastructure failure"):
        prepare_campaign_resume(campaign)

    assert (campaign / "campaign.json").read_bytes() == original
    assert not (campaign / "reports" / "campaign_manifest_before_cache_recovery.json").exists()
