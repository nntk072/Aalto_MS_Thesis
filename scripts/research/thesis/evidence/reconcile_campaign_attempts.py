#!/usr/bin/env python3
"""Reconcile all saved attempts for a campaign identity from primary artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHECKPOINT_RE = re.compile(r"ppo_ckpt_(\d+)_steps\.zip$")
_EXIT_RE = re.compile(r"(?:END key=.*? exit=|=== .*? exit=)(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _log_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "exit_code": None,
        "failure_excerpt": None,
    }
    if not path.is_file():
        return result
    result["path"] = path.as_posix()
    result["sha256"] = _sha256(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in _EXIT_RE.finditer(text):
        result["exit_code"] = int(match.group(1))
    lines = text.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if (
            "Traceback (most recent call last)" in stripped
            or "Error:" in stripped
            or "Exception:" in stripped
            or "error: " in stripped.lower()
        ):
            result["failure_excerpt"] = stripped[-500:]
            break
    if "FileExistsError" in text and "cache" in text:
        result["classification"] = "infrastructure_failure_before_training"
    elif result["exit_code"] == 0:
        result["classification"] = "process_exit_zero"
    elif result["exit_code"] is not None:
        result["classification"] = "process_failed_or_interrupted"
    return result


def _checkpoint(attempt_dir: Path) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in attempt_dir.rglob("ppo_ckpt_*_steps.zip"):
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    steps, path = max(candidates, key=lambda item: item[0])
    return {
        "steps": steps,
        "path": path.as_posix(),
        "sha256": _sha256(path),
    }


def _episode_coverage(attempt_dir: Path) -> dict[str, Any]:
    paths = sorted(attempt_dir.rglob("episode_equity.csv"))
    result: dict[str, Any] = {
        "status": "missing",
        "artifacts": [],
        "episode_count": 0,
        "full_split_episode_count": 0,
        "environments_with_full_split_episode": 0,
        "completed_environment_indices": [],
        "episode_steps": [],
        "end_reasons": [],
    }
    if not paths:
        return result
    episodes: list[dict[str, str]] = []
    for path in paths:
        record = {"path": path.as_posix(), "sha256": _sha256(path)}
        result["artifacts"].append(record)
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                required = {
                    "environment_index",
                    "episode_steps",
                    "completed_data_split",
                    "episode_end_reason",
                }
                if not required.issubset(set(reader.fieldnames or ())):
                    result["status"] = "legacy_or_incomplete_schema"
                    continue
                episodes.extend(reader)
        except (OSError, csv.Error):
            result["status"] = "unreadable"
    if not episodes:
        return result

    completed: set[int] = set()
    steps: set[int] = set()
    reasons: set[str] = set()
    full_count = 0
    for episode in episodes:
        if episode.get("completed_data_split", "").strip().lower() not in {
            "true",
            "1",
            "yes",
        }:
            continue
        try:
            env_index = int(episode["environment_index"])
            step_count = int(episode["episode_steps"])
        except (KeyError, ValueError):
            continue
        full_count += 1
        completed.add(env_index)
        steps.add(step_count)
        reasons.add(episode.get("episode_end_reason", ""))
    result.update(
        status="available",
        episode_count=len(episodes),
        full_split_episode_count=full_count,
        environments_with_full_split_episode=len(completed),
        completed_environment_indices=sorted(completed),
        episode_steps=sorted(steps),
        end_reasons=sorted(reasons),
    )
    return result


def reconcile_campaign(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    campaign = _read_json(campaign_dir / "campaign.json")
    if campaign is None:
        raise ValueError(f"missing or invalid campaign manifest: {campaign_dir / 'campaign.json'}")

    run_reports: list[dict[str, Any]] = []
    for job in campaign.get("jobs", []):
        if not isinstance(job, dict) or not job.get("key"):
            continue
        key = str(job["key"])
        attempt_count = int(job.get("attempts", 0))
        attempts: list[dict[str, Any]] = []
        for attempt_number in range(1, attempt_count + 1):
            attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{attempt_number}"
            provenance_path = attempt_dir / "attempt_provenance.json"
            provenance = _read_json(provenance_path)
            log_path = campaign_dir / "logs" / "workers" / f"{key}__attempt{attempt_number}.log"
            if not log_path.is_file() and attempt_number == 1:
                log_path = campaign_dir / "logs" / "workers" / f"{key}.log"
            attempts.append(
                {
                    "attempt": attempt_number,
                    "provenance": {
                        "path": provenance_path.as_posix() if provenance else None,
                        "sha256": _sha256(provenance_path) if provenance else None,
                        "started": provenance.get("started") if provenance else None,
                        "slurm_job_id": provenance.get("slurm_job_id") if provenance else None,
                        "resume_origin": provenance.get("resume_origin") if provenance else None,
                        "resume_from_attempt": provenance.get("resume_from_attempt")
                        if provenance
                        else None,
                        "resume_checkpoint": provenance.get("resume_checkpoint")
                        if provenance
                        else None,
                        "resume_checkpoint_sha256": provenance.get("resume_checkpoint_sha256")
                        if provenance
                        else None,
                    },
                    "worker_log": _log_summary(log_path),
                    "latest_checkpoint": _checkpoint(attempt_dir),
                    "training_episode_coverage": _episode_coverage(attempt_dir),
                    "run_artifact_directory_exists": any(attempt_dir.glob(f"{key}/{key}_*")),
                }
            )

        checkpoints = [
            attempt["latest_checkpoint"]
            for attempt in attempts
            if attempt["latest_checkpoint"] is not None
        ]
        complete_episode_attempts = [
            attempt
            for attempt in attempts
            if attempt["training_episode_coverage"]["environments_with_full_split_episode"]
        ]
        run_reports.append(
            {
                "run_key": key,
                "variant": job.get("variant"),
                "seed": job.get("seed"),
                "manifest_status": job.get("status"),
                "attempt_count": attempt_count,
                "requested_timesteps": campaign.get("timesteps"),
                "maximum_saved_checkpoint_steps": max(
                    (checkpoint["steps"] for checkpoint in checkpoints), default=None
                ),
                "latest_checkpoint": max(checkpoints, key=lambda item: item["steps"])
                if checkpoints
                else None,
                "attempts_with_full_split_episode_coverage": [
                    {
                        "attempt": attempt["attempt"],
                        "episode_coverage": attempt["training_episode_coverage"],
                    }
                    for attempt in complete_episode_attempts
                ],
                "attempts": attempts,
            }
        )

    status_snapshot = _read_json(campaign_dir / "reports" / "campaign_status.json")
    manifest_statuses = Counter(
        str(job.get("status", "unknown")) for job in campaign.get("jobs", [])
    )
    attempt4_cache_failures = sum(
        any(
            attempt["attempt"] == 4
            and attempt["worker_log"].get("classification")
            == "infrastructure_failure_before_training"
            and attempt["latest_checkpoint"] is None
            for attempt in run["attempts"]
        )
        for run in run_reports
    )
    complete_run_count = sum(
        run["maximum_saved_checkpoint_steps"] is not None
        and run["maximum_saved_checkpoint_steps"] >= int(campaign.get("timesteps", 0))
        for run in run_reports
    )
    return {
        "schema_version": 1,
        "artifact_role": "campaign_attempt_reconciliation",
        "reconciled_at_utc": datetime.now(UTC).isoformat(),
        "campaign": {
            "directory": str(campaign_dir),
            "git_sha": campaign.get("git_sha"),
            "fingerprint": campaign.get("fingerprint"),
            "manifest_sha256": _sha256(campaign_dir / "campaign.json"),
            "requested_timesteps_per_variant": campaign.get("timesteps"),
            "n_envs": campaign.get("n_envs"),
            "data_source": campaign.get("data_source"),
            "evaluation_protocol": campaign.get("evaluation"),
            "manifest_status_counts": dict(sorted(manifest_statuses.items())),
            "status_snapshot": status_snapshot,
        },
        "summary": {
            "run_identities": len(run_reports),
            "runs_with_budget_checkpoint": complete_run_count,
            "runs_with_full_split_episode_telemetry": sum(
                bool(run["attempts_with_full_split_episode_coverage"]) for run in run_reports
            ),
            "attempt4_cache_failures_before_training": attempt4_cache_failures,
            "all_run_attempts_retained": all(
                len(run["attempts"]) == run["attempt_count"] for run in run_reports
            ),
        },
        "limitations": [
            "A checkpoint below the requested budget is recoverable progress, not a completed run.",
            "Episode telemetry proves the recorded environment episode reached its configured train-split boundary; it does not establish valid evaluation or policy quality.",
            "The campaign's historical test data is previously inspected and selection-acknowledged, not pristine confirmatory OOS evidence.",
            "The status snapshot is included with its own timestamp and is not substituted for the per-attempt logs and provenance.",
        ],
        "runs": run_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile_campaign(args.campaign_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
