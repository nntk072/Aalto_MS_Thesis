"""Tests for preserving campaign identity while repairing cache failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compute.slurm.resume import prepare_campaign_resume


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
            "run_key": key,
            "variant": key.removesuffix("__seed50"),
            "seed": 50,
            "resume_checkpoint": str(checkpoint),
            "resume_checkpoint_sha256": checkpoint_hash,
        }
        (attempt_dir / "attempt_provenance.json").write_text(json.dumps(provenance))
        log = campaign / "logs" / "workers" / f"{key}__attempt{attempt}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "FileExistsError: [Errno 17] File exists: 'cache'\n"
            "market_data/loading/pipeline.py run_pipeline\n"
        )
        jobs.append(
            {
                "key": key,
                "variant": key.removesuffix("__seed50"),
                "seed": 50,
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
    assert {job["resume_from_attempt"] for job in recovered["jobs"]} == {3}
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


def test_resume_repairs_incorrect_attempt_pointer_from_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    campaign, _original = _campaign_fixture(tmp_path)
    prepare_campaign_resume(campaign)
    manifest_path = campaign / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    for job in manifest["jobs"]:
        job["resume_from_attempt"] = 4
    incorrect_manifest = json.dumps(manifest, indent=2) + "\n"
    manifest_path.write_text(incorrect_manifest)

    result = prepare_campaign_resume(campaign)

    repaired = json.loads(manifest_path.read_text())
    assert result["resume_attempt_corrections"] == 18
    assert {job["resume_from_attempt"] for job in repaired["jobs"]} == {3}
    manifest_backup = campaign / "reports" / "campaign_manifest_before_resume_attempt_repair.json"
    assert manifest_backup.read_text() == incorrect_manifest
    report = json.loads((campaign / "reports" / "cache_recovery.json").read_text())
    assert len(report["resume_attempt_corrections"]) == 18
    assert all(
        correction["previous_resume_from_attempt"] == 4 and correction["resume_from_attempt"] == 3
        for correction in report["resume_attempt_corrections"]
    )
    assert prepare_campaign_resume(campaign)["already_resumable"] is True


def test_resume_refuses_unknown_failure_without_mutating_manifest(tmp_path: Path) -> None:
    campaign, original = _campaign_fixture(tmp_path)
    log = campaign / "logs" / "workers" / "variant_17__attempt4.log"
    log.write_text("ValueError: strategy failure\n")

    with pytest.raises(ValueError, match="known cache infrastructure failure"):
        prepare_campaign_resume(campaign)

    assert (campaign / "campaign.json").read_bytes() == original
    assert not (campaign / "reports" / "campaign_manifest_before_cache_recovery.json").exists()
