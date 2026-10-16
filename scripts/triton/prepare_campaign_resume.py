#!/usr/bin/env python3
"""Requeue only the known cache-path infrastructure failures for resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FAILURE_SIGNATURE = "FileExistsError: [Errno 17] File exists:"
CACHE_FAILURE_SIGNATURE = "FileExistsError: [Errno 17] File exists: 'cache'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _checkpoint_attempt(checkpoint: Path, campaign_dir: Path) -> int:
    relative = checkpoint.resolve().relative_to(campaign_dir.resolve())
    attempts = [
        int(match.group(1))
        for part in relative.parts
        if (match := re.fullmatch(r"attempt(\d+)", part)) is not None
    ]
    if len(attempts) != 1:
        raise ValueError(f"resume checkpoint has ambiguous attempt lineage: {checkpoint}")
    return attempts[0]


def _write_immutable_backup(path: Path, contents: bytes) -> None:
    if path.exists():
        if path.read_bytes() != contents:
            raise FileExistsError(f"recovery backup differs from current evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_campaign_resume(campaign_dir: Path) -> dict[str, Any]:
    manifest_path = campaign_dir / "campaign.json"
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes)
    fingerprint = manifest.get("fingerprint")
    jobs = manifest.get("jobs")
    if not isinstance(fingerprint, str) or not isinstance(jobs, list) or len(jobs) != 18:
        raise ValueError("expected the registered 18-run campaign manifest")

    failed: list[dict[str, Any]] = []
    checkpoint_steps: list[int] = []
    resume_attempt_by_key: dict[str, int] = {}
    for job in jobs:
        if job.get("status") != "failed":
            continue
        key = str(job.get("key", ""))
        attempt_number = int(job.get("attempts", 0))
        if not key or attempt_number < 1:
            raise ValueError("failed campaign identity has invalid key/attempt")
        attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{attempt_number}"
        log_path = campaign_dir / "logs" / "workers" / f"{key}__attempt{attempt_number}.log"
        provenance_path = attempt_dir / "attempt_provenance.json"
        if not log_path.is_file() or not provenance_path.is_file():
            raise ValueError(f"missing failed-attempt evidence for {key}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if (
            FAILURE_SIGNATURE not in log_text
            or CACHE_FAILURE_SIGNATURE not in log_text
            or "quant_rl/data/pipeline.py" not in log_text
        ):
            raise ValueError(f"{key} does not match the known cache infrastructure failure")
        attempt_checkpoints = list(attempt_dir.glob("**/model/ppo_ckpt_*_steps.zip"))
        if attempt_checkpoints:
            raise ValueError(f"failed attempt unexpectedly contains checkpoints for {key}")

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance.get("attempt") != attempt_number
            or provenance.get("run_key") != key
            or provenance.get("variant") != job.get("variant")
            or provenance.get("seed") != job.get("seed")
        ):
            raise ValueError(f"attempt provenance number mismatch for {key}")
        if provenance.get("campaign_fingerprint") != fingerprint:
            raise ValueError(f"attempt fingerprint mismatch for {key}")
        checkpoint_raw = provenance.get("resume_checkpoint")
        checkpoint_hash = provenance.get("resume_checkpoint_sha256")
        if not checkpoint_raw or not checkpoint_hash:
            raise ValueError(f"no verified resume checkpoint recorded for {key}")
        checkpoint = Path(checkpoint_raw)
        if not checkpoint.resolve().is_relative_to(campaign_dir.resolve()):
            raise ValueError(f"resume checkpoint escapes this campaign for {key}")
        if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_hash:
            raise ValueError(f"resume checkpoint is missing or changed for {key}")
        checkpoint_attempt = _checkpoint_attempt(checkpoint, campaign_dir)
        if checkpoint_attempt >= attempt_number:
            raise ValueError(f"resume checkpoint is not earlier than failed attempt for {key}")
        match = re.search(r"ppo_ckpt_(\d+)_steps\.zip$", checkpoint.name)
        if not match:
            raise ValueError(f"resume checkpoint step is not identifiable for {key}")
        checkpoint_steps.append(int(match.group(1)))
        resume_attempt_by_key[key] = checkpoint_attempt
        failed.append(
            {
                "key": key,
                "attempt": attempt_number,
                "log": str(log_path.relative_to(campaign_dir)),
                "log_sha256": _sha256(log_path),
                "resume_checkpoint": checkpoint_raw,
                "resume_checkpoint_sha256": checkpoint_hash,
                "resume_checkpoint_steps": int(match.group(1)),
                "resume_checkpoint_attempt": checkpoint_attempt,
            }
        )

    if not failed:
        recovery_path = campaign_dir / "reports" / "cache_recovery.json"
        if not recovery_path.is_file():
            return {
                "campaign_fingerprint": fingerprint,
                "requeued": 0,
                "already_resumable": True,
            }
        recovery_bytes = recovery_path.read_bytes()
        recovery_report = json.loads(recovery_bytes)
        if recovery_report.get("campaign_fingerprint") != fingerprint:
            raise ValueError("cache recovery report fingerprint does not match campaign")
        jobs_by_key = {str(job.get("key", "")): job for job in jobs}
        corrections: list[dict[str, Any]] = []
        for item in recovery_report.get("requeued_identities", []):
            key = str(item.get("key", ""))
            job = jobs_by_key.get(key)
            if job is None or job.get("status") != "interrupted":
                continue
            attempt_number = int(item.get("attempt", 0))
            log_path = campaign_dir / str(item.get("log", ""))
            if not log_path.is_file() or _sha256(log_path) != item.get("log_sha256"):
                raise ValueError(f"recovery report log is missing or changed for {key}")
            provenance_path = (
                campaign_dir
                / "runs"
                / "attempts"
                / key
                / f"attempt{attempt_number}"
                / "attempt_provenance.json"
            )
            if not provenance_path.is_file():
                raise ValueError(f"recovery report provenance is missing for {key}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            checkpoint = Path(str(item.get("resume_checkpoint", "")))
            if (
                provenance.get("campaign_fingerprint") != fingerprint
                or provenance.get("run_key") != key
                or provenance.get("attempt") != attempt_number
                or provenance.get("resume_checkpoint") != str(checkpoint)
                or provenance.get("resume_checkpoint_sha256")
                != item.get("resume_checkpoint_sha256")
                or not checkpoint.is_file()
                or _sha256(checkpoint) != item.get("resume_checkpoint_sha256")
            ):
                raise ValueError(f"recovery report checkpoint is missing or changed for {key}")
            checkpoint_attempt = _checkpoint_attempt(checkpoint, campaign_dir)
            previous_attempt = job.get("resume_from_attempt")
            if previous_attempt != checkpoint_attempt:
                corrections.append(
                    {
                        "key": key,
                        "previous_resume_from_attempt": previous_attempt,
                        "resume_from_attempt": checkpoint_attempt,
                        "resume_checkpoint": str(checkpoint),
                    }
                )
                job["resume_from_attempt"] = checkpoint_attempt

        if corrections:
            current_manifest_bytes = manifest_path.read_bytes()
            backup_path = (
                campaign_dir / "reports" / "campaign_manifest_before_resume_attempt_repair.json"
            )
            _write_immutable_backup(backup_path, current_manifest_bytes)
            report_backup = (
                campaign_dir / "reports" / "cache_recovery_before_resume_attempt_repair.json"
            )
            _write_immutable_backup(report_backup, recovery_bytes)
            recovery_report["resume_attempt_corrections"] = corrections
            recovery_report["resume_attempt_repair_manifest_backup"] = str(
                backup_path.relative_to(campaign_dir)
            )
            _atomic_json(manifest_path, manifest)
            _atomic_json(recovery_path, recovery_report)
        return {
            "campaign_fingerprint": fingerprint,
            "requeued": 0,
            "resume_attempt_corrections": len(corrections),
            "already_resumable": not corrections,
        }

    backup_path = campaign_dir / "reports" / "campaign_manifest_before_cache_recovery.json"
    _write_immutable_backup(backup_path, original_bytes)

    lineage_before = json.dumps(manifest.get("initial_checkpoint_lineage"), sort_keys=True)
    for job in jobs:
        if job.get("status") == "failed":
            job["status"] = "interrupted"
            job["resume_from_attempt"] = resume_attempt_by_key[str(job["key"])]
    if (
        manifest.get("fingerprint") != fingerprint
        or json.dumps(manifest.get("initial_checkpoint_lineage"), sort_keys=True) != lineage_before
    ):
        raise RuntimeError("recovery unexpectedly changed campaign identity")

    recovery_report = {
        "schema": "aalto.campaign_cache_recovery.v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "campaign_id": campaign_dir.name,
        "campaign_fingerprint": fingerprint,
        "pre_recovery_manifest": str(backup_path.relative_to(campaign_dir)),
        "pre_recovery_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "classification": (
            "known infrastructure failure before training; all run logs/provenance retained"
        ),
        "requeued_identities": failed,
        "resume_checkpoint_steps_min": min(checkpoint_steps),
        "resume_checkpoint_steps_max": max(checkpoint_steps),
        "identity_changed": False,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(campaign_dir / "reports" / "cache_recovery.json", recovery_report)
    return {
        "campaign_fingerprint": fingerprint,
        "requeued": len(failed),
        "already_resumable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_campaign_resume(args.campaign_dir), sort_keys=True))


if __name__ == "__main__":
    main()
