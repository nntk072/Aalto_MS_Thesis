#!/usr/bin/env python3
"""Build an appendable run/attempt ledger from preserved campaign evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_RE = re.compile(r"ppo_ckpt_(\d+)_steps\.zip$")
TERMINAL_INTERRUPTION_STATES = (
    "CANCELLED",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _scheduler_state(job_id: Any) -> str | None:
    if job_id is None:
        return None
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobID,State",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 2 and fields[0] == str(job_id):
            return fields[1].strip()
    return None


def _job_id(value: Any) -> int | None:
    if isinstance(value, (int, str)) and str(value).isdigit():
        return int(value)
    return None


def _attempt_paths(attempt_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    provenance = attempt_dir / "attempt_provenance.json"
    config_paths = sorted(attempt_dir.rglob("resolved_config.json"))
    checkpoints: list[tuple[int, Path]] = []
    for candidate in attempt_dir.rglob("ppo_ckpt_*_steps.zip"):
        match = CHECKPOINT_RE.fullmatch(candidate.name)
        if match:
            checkpoints.append((int(match.group(1)), candidate))
    checkpoint = max(checkpoints, default=(0, None), key=lambda item: item[0])[1]
    return (
        provenance if provenance.is_file() else None,
        config_paths[-1] if config_paths else None,
        checkpoint,
    )


def _data_hashes(value: Any) -> list[str]:
    if isinstance(value, dict):
        candidates = list(value.values())
    elif isinstance(value, list):
        candidates = value
    else:
        return []
    return sorted(
        {
            item
            for item in candidates
            if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
        }
    )


def _attempt_failure_map(campaign_dir: Path) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for report in campaign_dir.glob("reports/*infrastructure*reconciliation*.json"):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = payload.get("variants", [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("key"), str):
                    attempt = item.get("attempts")
                    if isinstance(attempt, int):
                        result[(item["key"], attempt)] = str(
                            item.get(
                                "classification",
                                item.get("failure_signature", "infrastructure failure"),
                            )
                        )
        if isinstance(payload.get("key"), str) and isinstance(payload.get("attempts"), int):
            result[(payload["key"], payload["attempts"])] = str(
                payload.get(
                    "classification", payload.get("failure_signature", "infrastructure failure")
                )
            )
    return result


def _reconciled_run_status(
    reported: str, scheduler: str | None, completed_steps: int | None, requested_steps: int
) -> tuple[str, str]:
    if reported != "running" or scheduler is None:
        return reported, "Campaign manifest status retained."
    if scheduler.startswith(TERMINAL_INTERRUPTION_STATES):
        if (
            completed_steps is not None
            and completed_steps >= requested_steps
            and scheduler.startswith("COMPLETED")
        ):
            return "succeeded", "Completed budget checkpoint and terminal Slurm allocation."
        return (
            "interrupted",
            f"Manifest said running, but Slurm reports terminal allocation state {scheduler}.",
        )
    return reported, f"Slurm state observed as {scheduler}; run state remains as recorded."


def build_trial_ledger(
    campaign_dirs: list[Path], scheduler_states: dict[int, str] | None = None
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    limitations: list[str] = []
    observed_campaigns: list[str] = []
    for campaign_dir_raw in sorted(campaign_dirs, key=lambda path: path.name):
        campaign_dir = campaign_dir_raw.resolve()
        manifest_path = campaign_dir / "campaign.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"campaign manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.get("fingerprint")
        campaign_id = campaign_dir.name
        observed_campaigns.append(campaign_id)
        manifest_job_id = _job_id(manifest.get("job_id"))
        observed_job_ids: set[int] = set()
        scheduler_cache: dict[int, str | None] = {}

        def scheduler_for(job_id: int | None) -> str | None:
            if job_id is None:
                return None
            observed_job_ids.add(job_id)
            if job_id in scheduler_cache:
                return scheduler_cache[job_id]
            if scheduler_states is not None:
                scheduler_cache[job_id] = scheduler_states.get(job_id)
            else:
                scheduler_cache[job_id] = _scheduler_state(job_id)
            return scheduler_cache[job_id]

        manifest_scheduler = scheduler_for(manifest_job_id)
        failures = _attempt_failure_map(campaign_dir)
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"campaign has no jobs list: {campaign_id}")
        for job in sorted(jobs, key=lambda item: str(item.get("key", ""))):
            key = str(job["key"])
            attempts = int(job.get("attempts", 0))
            variant = str(job.get("variant", key.rsplit("__seed", maxsplit=1)[0]))
            seed = int(job.get("seed", key.rsplit("__seed", maxsplit=1)[1]))
            expected = int(manifest.get("timesteps", 0))
            attempt_entries: list[dict[str, Any]] = []
            latest_steps: int | None = None
            latest_checkpoint_sha: str | None = None
            latest_config_sha: str | None = None
            latest_source_sha: str | None = None
            run_artifacts: set[str] = set()
            missing_identity_attempts = 0
            latest_attempt_job_id: int | None = None
            latest_attempt_scheduler: str | None = None
            attempt_job_ids: set[int] = set()

            for number in range(1, attempts + 1):
                attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{number}"
                provenance_path, config_path, checkpoint_path = _attempt_paths(attempt_dir)
                provenance: dict[str, Any] = {}
                if provenance_path is not None:
                    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                identity = provenance.get("identity", {}) if isinstance(provenance, dict) else {}
                if not isinstance(identity, dict):
                    identity = {}
                attempt_job_id = _job_id(provenance.get("slurm_job_id")) or manifest_job_id
                attempt_scheduler = scheduler_for(attempt_job_id)
                if attempt_job_id is not None:
                    attempt_job_ids.add(attempt_job_id)
                if number == attempts:
                    latest_attempt_job_id = attempt_job_id
                    latest_attempt_scheduler = attempt_scheduler
                source_sha = identity.get("git_sha")
                if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
                    source_sha = manifest.get("git_sha")
                data_sha256 = _data_hashes(identity.get("data_inputs", manifest.get("data_inputs")))
                config_sha = _sha256(config_path) if config_path is not None else None
                checkpoint_sha = _sha256(checkpoint_path) if checkpoint_path is not None else None
                checkpoint_match = (
                    CHECKPOINT_RE.fullmatch(checkpoint_path.name)
                    if checkpoint_path is not None
                    else None
                )
                steps = int(checkpoint_match.group(1)) if checkpoint_match is not None else None
                if config_sha is None or provenance_path is None:
                    missing_identity_attempts += 1
                for path in (provenance_path, config_path, checkpoint_path):
                    artifact = _artifact(path)
                    if artifact:
                        run_artifacts.add(artifact)
                log_path = campaign_dir / "logs" / "workers" / f"{key}__attempt{number}.log"
                log_artifact = _artifact(log_path)
                if log_artifact:
                    run_artifacts.add(log_artifact)

                if steps is not None:
                    latest_steps = max(latest_steps or 0, steps)
                    if steps == latest_steps:
                        latest_checkpoint_sha = checkpoint_sha
                if config_sha is not None:
                    latest_config_sha = config_sha
                if isinstance(source_sha, str):
                    latest_source_sha = source_sha

                if number < attempts:
                    attempt_status = "interrupted"
                    disposition = f"Superseded by attempt {number + 1}; artifacts retained."
                elif (key, number) in failures:
                    attempt_status = "failed"
                    disposition = failures[(key, number)]
                else:
                    attempt_status = str(job.get("status", "interrupted"))
                    disposition = str(
                        job.get("failure_reason")
                        or "Terminal disposition follows reconciled run state."
                    )
                identity_has_source = (
                    isinstance(source_sha, str)
                    and re.fullmatch(r"[0-9a-f]{40}", source_sha) is not None
                )
                identity_has_fingerprint = (
                    isinstance(fingerprint, str)
                    and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
                )
                completeness = (
                    "complete"
                    if (
                        provenance_path is not None
                        and config_sha is not None
                        and identity_has_source
                        and identity_has_fingerprint
                        and bool(data_sha256)
                    )
                    else "partial"
                    if (
                        provenance_path is not None or identity_has_source or config_sha is not None
                    )
                    else "missing"
                )
                attempt_entries.append(
                    {
                        "record_id": f"{campaign_id}::{key}::attempt{number}",
                        "record_type": "attempt",
                        "campaign_id": campaign_id,
                        "campaign_fingerprint": fingerprint,
                        "run_id": key,
                        "attempt": number,
                        "variant": variant,
                        "seed": seed,
                        "evidence_role": "runtime_benchmark"
                        if "probe" in campaign_id
                        else "selection",
                        "status": attempt_status,
                        "source_sha": source_sha if isinstance(source_sha, str) else None,
                        "config_sha256": config_sha,
                        "data_sha256": data_sha256,
                        "cache_sha256": [],
                        "checkpoint_sha256": checkpoint_sha,
                        "requested_timesteps": expected or None,
                        "completed_timesteps": steps,
                        "n_envs": int(manifest["n_envs"]) if manifest.get("n_envs") else None,
                        "started_at_utc": provenance.get("started") if provenance else None,
                        "ended_at_utc": None,
                        "artifacts": sorted(
                            artifact
                            for artifact in (
                                _artifact(provenance_path),
                                _artifact(config_path),
                                _artifact(checkpoint_path),
                                log_artifact,
                            )
                            if artifact is not None
                        ),
                        "disposition": disposition,
                        "supersedes_record_id": (
                            f"{campaign_id}::{key}::attempt{number - 1}" if number > 1 else None
                        ),
                        "slurm_job_id": attempt_job_id,
                        "scheduler_state": attempt_scheduler,
                        "identity_completeness": completeness,
                    }
                )

            reported_status = str(job.get("status", "interrupted"))
            run_status, run_disposition = _reconciled_run_status(
                reported_status,
                latest_attempt_scheduler or manifest_scheduler,
                latest_steps,
                expected,
            )
            if (
                attempt_entries
                and attempt_entries[-1]["status"] == "running"
                and run_status != "running"
            ):
                attempt_entries[-1]["status"] = run_status
                attempt_entries[-1]["disposition"] = run_disposition
            config_sha = latest_config_sha
            source_sha = latest_source_sha or manifest.get("git_sha")
            if config_sha is None or not latest_source_sha:
                missing_identity_attempts += 1
            records.append(
                {
                    "record_id": f"{campaign_id}::{key}::run",
                    "record_type": "run",
                    "campaign_id": campaign_id,
                    "campaign_fingerprint": fingerprint,
                    "run_id": key,
                    "attempt": None,
                    "variant": variant,
                    "seed": seed,
                    "evidence_role": "runtime_benchmark" if "probe" in campaign_id else "selection",
                    "status": run_status,
                    "source_sha": source_sha if isinstance(source_sha, str) else None,
                    "config_sha256": config_sha,
                    "data_sha256": _data_hashes(manifest.get("data_inputs")),
                    "cache_sha256": [],
                    "checkpoint_sha256": latest_checkpoint_sha,
                    "requested_timesteps": expected or None,
                    "completed_timesteps": latest_steps,
                    "n_envs": int(manifest["n_envs"]) if manifest.get("n_envs") else None,
                    "started_at_utc": job.get("started"),
                    "ended_at_utc": job.get("ended") if run_status != "interrupted" else None,
                    "artifacts": sorted(
                        {
                            artifact
                            for artifact in (
                                _artifact(manifest_path),
                                _artifact(
                                    campaign_dir
                                    / str(manifest.get("source_snapshot", {}).get("path", ""))
                                ),
                                *run_artifacts,
                            )
                            if artifact is not None
                        }
                    ),
                    "disposition": (
                        f"{run_disposition} Reported manifest state={reported_status}; "
                        f"attempt records follow."
                    ),
                    "supersedes_record_id": None,
                    "slurm_job_id": latest_attempt_job_id or manifest_job_id,
                    "slurm_job_ids": sorted(observed_job_ids),
                    "scheduler_state": latest_attempt_scheduler or manifest_scheduler,
                    "identity_completeness": (
                        "complete"
                        if config_sha and latest_source_sha
                        else "partial"
                        if source_sha or config_sha
                        else "missing"
                    ),
                }
            )
            records.extend(attempt_entries)
            if missing_identity_attempts:
                limitations.append(
                    f"{campaign_id}/{key}: {missing_identity_attempts} record(s) lack per-attempt config or provenance."
                )

    return {
        "schema": "aalto.thesis_trial_ledger.v1",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
        "limitations": sorted(
            set(
                [
                    "This snapshot covers the six structured campaign manifests present under outputs/gpu_campaigns; standalone legacy runs and deleted artifacts are not enumerated.",
                    "Manifest job state is reconciled with the parent Slurm state where that job ID is recorded; historical child-step and allocation reuse relationships may require manual review.",
                    "A scheduler state is a point-in-time observation and does not establish full chronological split exposure or strategy validity.",
                    "Missing provenance/configuration is represented as incomplete; no hashes are invented for absent artifacts.",
                    *limitations,
                ]
            )
        ),
        "campaign_ids": observed_campaigns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", action="append", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace the requested output file")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to replace existing trial ledger: {args.output}")
    ledger = build_trial_ledger(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = Counter(
        record["status"] for record in ledger["records"] if record["record_type"] == "run"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(ledger["records"]),
                "run_statuses": dict(statuses),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
