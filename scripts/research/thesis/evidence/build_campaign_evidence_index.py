#!/usr/bin/env python3
"""Build a provenance-first artifact index for one campaign directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_RE = re.compile(r"ppo_ckpt_(\d+)_steps\.zip$")
RAW_ARTIFACT_NAMES = {
    "config.yaml",
    "config.json",
    "resolved_config.json",
    "run_manifest.json",
    "training_log.json",
    "progress.csv",
    "metrics.json",
    "equity.csv",
    "trades.csv",
    "fills.csv",
    "positions.csv",
    "breaches.csv",
    "actions.csv",
    "reward_attribution.csv",
    "episode_equity.csv",
    "train_equity_gate.json",
    "ppo_final.zip",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_artifacts(attempt_dir: Path, root: Path) -> list[dict[str, Any]]:
    files = [
        path
        for path in attempt_dir.rglob("*")
        if path.is_file()
        and (path.name in RAW_ARTIFACT_NAMES or CHECKPOINT_RE.fullmatch(path.name))
    ]
    return [_artifact(path, root) for path in sorted(files)]


def _reported_completed_steps(attempt_dir: Path) -> int | None:
    values: list[int] = []
    for name in ("run_manifest.json", "training_log.json"):
        for path in attempt_dir.rglob(name):
            record = _read_json(path)
            if not record:
                continue
            candidates = [record.get("completed_steps"), record.get("timesteps_completed")]
            protocol = record.get("training_protocol")
            if isinstance(protocol, dict):
                candidates.append(protocol.get("completed_steps"))
            for candidate in candidates:
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    values.append(candidate)
    return max(values) if values else None


def _campaign_artifacts(campaign_dir: Path, root: Path) -> list[dict[str, Any]]:
    roots = (campaign_dir / "logs", campaign_dir / "reports", campaign_dir / "cache")
    files = [
        path
        for directory in roots
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "thesis_evidence_index.json"
        and not path.name.startswith("thesis_evidence_index_")
    ]
    return [_artifact(path, root) for path in sorted(files)]


def _attempt_record(campaign_dir: Path, key: str, number: int) -> dict[str, Any]:
    root = campaign_dir.parent.parent.parent
    attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{number}"
    provenance_path = attempt_dir / "attempt_provenance.json"
    provenance = _read_json(provenance_path)
    log_path = campaign_dir / "logs" / "workers" / f"{key}__attempt{number}.log"
    checkpoints: list[tuple[int, Path]] = []
    for path in attempt_dir.rglob("ppo_ckpt_*_steps.zip"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    final_checkpoints = [path for path in attempt_dir.rglob("ppo_final.zip") if path.is_file()]
    resume_checkpoint = None
    if provenance:
        checkpoint_value = provenance.get("resume_checkpoint")
        checkpoint_hash = provenance.get("resume_checkpoint_sha256")
        if isinstance(checkpoint_value, str) and isinstance(checkpoint_hash, str):
            checkpoint_path = Path(checkpoint_value)
            try:
                checkpoint_path.resolve().relative_to(campaign_dir.resolve())
                checkpoint_is_verified = (
                    checkpoint_path.is_file() and _sha256(checkpoint_path) == checkpoint_hash
                )
            except (OSError, ValueError):
                checkpoint_is_verified = False
            resume_checkpoint = {
                "path": checkpoint_value,
                "sha256_recorded": checkpoint_hash,
                "resume_from_attempt": provenance.get("resume_from_attempt"),
                "origin": provenance.get("resume_origin"),
                "file_hash_verified": checkpoint_is_verified,
            }
    return {
        "attempt": number,
        "provenance": _artifact(provenance_path, root) if provenance_path.is_file() else None,
        "worker_log": _artifact(log_path, root) if log_path.is_file() else None,
        "resume_checkpoint": resume_checkpoint,
        "completed_steps_from_checkpoints": [steps for steps, _path in checkpoints],
        "checkpoints": [_artifact(path, root) for _steps, path in checkpoints],
        "final_checkpoints": [_artifact(path, root) for path in sorted(final_checkpoints)],
        "reported_completed_steps": _reported_completed_steps(attempt_dir),
        "raw_run_artifacts": _run_artifacts(attempt_dir, root),
    }


def _classify(job: dict[str, Any], attempts: list[dict[str, Any]], budget: int) -> str:
    status = str(job.get("status", "unknown"))
    step_values = [
        step for attempt in attempts for step in attempt["completed_steps_from_checkpoints"]
    ]
    if status == "succeeded":
        completed = [
            int(attempt["reported_completed_steps"])
            for attempt in attempts
            if isinstance(attempt.get("reported_completed_steps"), int)
        ]
        has_final_checkpoint = any(attempt.get("final_checkpoints") for attempt in attempts)
        if completed and max(completed) >= budget and has_final_checkpoint:
            return "completed_requested_budget_with_final_checkpoint"
        return "controller_succeeded_but_budget_or_final_checkpoint_unverified"
    if status == "failed":
        latest = attempts[-1] if attempts else {}
        candidate = latest.get("resume_checkpoint")
        if isinstance(candidate, dict) and candidate.get("file_hash_verified"):
            return "failed_attempt_with_hash_verified_resume_candidate"
        if candidate:
            return "failed_attempt_with_unverified_resume_candidate"
        return "failed_requires_failure_review"
    if status == "interrupted" and step_values:
        return "infrastructure_interrupted_with_checkpoint"
    if status == "interrupted":
        return "interrupted_without_indexed_checkpoint"
    if status == "running":
        return "running"
    if status == "pending":
        return "pending"
    return f"unclassified_status:{status}"


def build_campaign_evidence_index(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    root = campaign_dir.parent.parent.parent
    manifest_path = campaign_dir / "campaign.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise ValueError(f"campaign manifest is missing or invalid: {manifest_path}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("campaign manifest jobs must be a list")

    run_records: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        key = str(job.get("key", ""))
        if not key:
            continue
        attempt_count = max(0, int(job.get("attempts", 0)))
        attempts = [
            _attempt_record(campaign_dir, key, number) for number in range(1, attempt_count + 1)
        ]
        valid_checkpoint_candidates = [
            artifact
            for attempt in attempts
            for artifact in [*attempt["checkpoints"], *attempt["final_checkpoints"]]
        ]
        ranked_checkpoint_candidates = [
            (step, artifact)
            for attempt in attempts
            for step, artifact in zip(
                attempt["completed_steps_from_checkpoints"],
                attempt["checkpoints"],
                strict=True,
            )
        ]
        ranked_checkpoint_candidates.extend(
            (int(attempt["reported_completed_steps"]), artifact)
            for attempt in attempts
            if isinstance(attempt.get("reported_completed_steps"), int)
            for artifact in attempt["final_checkpoints"]
        )
        latest_checkpoint = None
        if ranked_checkpoint_candidates:
            latest_checkpoint = max(ranked_checkpoint_candidates, key=lambda item: item[0])[1]
        elif valid_checkpoint_candidates:
            latest_checkpoint = valid_checkpoint_candidates[-1]
        run_records.append(
            {
                "key": key,
                "variant": job.get("variant"),
                "seed": job.get("seed"),
                "controller_status": job.get("status"),
                "classification": _classify(job, attempts, int(manifest.get("timesteps", 0))),
                "attempt_count": attempt_count,
                "attempts": attempts,
                "checkpoint_artifact_count": len(valid_checkpoint_candidates),
                "latest_checkpoint_by_step": latest_checkpoint,
                "full_split_exposure": "not_verified_by_this_index",
            }
        )

    source_hashes = manifest.get("source_hashes", {})
    verified_sources = []
    missing_sources = []
    mismatched_sources = []
    for relative, expected in sorted(source_hashes.items()):
        path = root / relative
        if not path.is_file():
            missing_sources.append(relative)
        elif _sha256(path) != expected:
            mismatched_sources.append(relative)
        else:
            verified_sources.append(relative)

    return {
        "schema": "aalto.campaign_evidence_index.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evidence_role": "inventory_and_provenance_index; not a strategy validation",
        "campaign": {
            "campaign_id": campaign_dir.name,
            "campaign_path": campaign_dir.relative_to(root).as_posix(),
            "campaign_manifest": _artifact(manifest_path, root),
            "fingerprint": manifest.get("fingerprint"),
            "recorded_git_sha": manifest.get("git_sha"),
            "requested_timesteps": manifest.get("timesteps"),
            "data_source": manifest.get("data_source"),
            "n_envs": manifest.get("n_envs"),
            "source_hash_count": len(source_hashes),
            "verified_source_hash_count": len(verified_sources),
            "missing_source_paths": missing_sources,
            "mismatched_source_paths": mismatched_sources,
            "data_inputs_recorded": manifest.get("data_inputs", {}),
        },
        "run_status_counts": dict(Counter(str(job.get("status", "unknown")) for job in jobs)),
        "campaign_artifacts": _campaign_artifacts(campaign_dir, root),
        "runs": run_records,
        "limitations": [
            "This index does not establish complete chronological split exposure.",
            "Checkpoint file hashes do not establish successful model deserialization.",
            "Metrics and strategy validity must be reconstructed and audited separately.",
        ],
    }


def write_index(index: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    index = build_campaign_evidence_index(args.campaign_dir)
    output = args.output or args.campaign_dir / "reports" / "thesis_evidence_index.json"
    write_index(index, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
