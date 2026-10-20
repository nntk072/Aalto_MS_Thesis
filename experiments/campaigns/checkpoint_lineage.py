"""Focused campaign execution checkpoint lineage services."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import sys
import zipfile
from typing import Any

from experiments.campaigns.constants import ROOT
from experiments.campaigns.source_snapshot import sha256_file, verify_campaign_source_snapshot
from experiments.campaigns.state_store import read_manifest


def latest_resumable_checkpoint(
    campaign_out: pathlib.Path,
    key: str,
    previous_attempt: int,
    fingerprint: str,
    max_steps: int | None = None,
) -> pathlib.Path | None:
    """Return the newest intact PPO checkpoint from a verified interrupted attempt."""
    attempt_out = campaign_out / "runs" / "attempts" / key / f"attempt{previous_attempt}"
    try:
        provenance = json.loads(
            (attempt_out / "attempt_provenance.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if (
        not isinstance(provenance, dict)
        or provenance.get("campaign_fingerprint") != fingerprint
        or provenance.get("run_key") != key
        or provenance.get("attempt") != previous_attempt
    ):
        return None
    checkpoints: list[tuple[int, pathlib.Path]] = []
    for path in attempt_out.rglob("ppo_ckpt_*_steps.zip"):
        match = re.fullmatch(r"ppo_ckpt_(\d+)_steps\.zip", path.name)
        if match:
            step = int(match.group(1))
            if max_steps is None or step <= max_steps:
                checkpoints.append((step, path))
    if max_steps is None:
        latest = list(attempt_out.rglob("ppo_latest.zip"))
        checkpoints.extend((0, path) for path in latest)
    for _step, path in sorted(checkpoints, key=lambda item: item[0], reverse=True):
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is None:
                        return path
        except (OSError, zipfile.BadZipFile):
            continue
    return None


def validate_initial_checkpoints(
    map_path: pathlib.Path | None, jobs: list[dict[str, Any]], target_steps: int
) -> dict[str, dict[str, Any]]:
    """Validate explicit parent-campaign checkpoints for a new lineage campaign."""
    if map_path is None:
        return {}
    raw = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("initial checkpoint map must use schema_version=1")
    entries = raw.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("initial checkpoint map must contain a checkpoints list")
    expected = {str(job["key"]): job for job in jobs}
    by_key: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("initial checkpoint entries must be mappings")
        key = str(item.get("key", ""))
        if key not in expected or key in by_key:
            raise ValueError(f"initial checkpoint map has unknown or duplicate run key: {key}")
        parent_dir_raw = item.get("source_campaign_dir")
        checkpoint_raw = item.get("checkpoint_path")
        if not isinstance(parent_dir_raw, str) or not isinstance(checkpoint_raw, str):
            raise ValueError(f"initial checkpoint paths are missing for {key}")
        parent_dir = pathlib.Path(parent_dir_raw)
        if not parent_dir.is_absolute():
            parent_dir = map_path.parent / parent_dir
        parent_dir = parent_dir.resolve()
        parent_manifest_path = parent_dir / "campaign.json"
        try:
            parent_manifest = read_manifest(parent_manifest_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read source campaign manifest for {key}") from exc
        source_fingerprint = str(item.get("source_campaign_fingerprint", ""))
        if parent_manifest.get("fingerprint") != source_fingerprint:
            raise ValueError(f"source campaign fingerprint mismatch for {key}")
        if item.get("source_campaign_id") not in (None, parent_dir.name):
            raise ValueError(f"source campaign ID mismatch for {key}")
        source_git_sha = str(parent_manifest.get("git_sha", ""))
        if item.get("source_campaign_git_sha") not in (None, source_git_sha):
            raise ValueError(f"source campaign revision mismatch for {key}")
        source_budget = int(parent_manifest.get("timesteps", 0))
        if item.get("source_requested_timesteps") not in (None, source_budget):
            raise ValueError(f"source campaign budget mismatch for {key}")
        parent_jobs = [job for job in parent_manifest.get("jobs", []) if job.get("key") == key]
        if len(parent_jobs) != 1:
            raise ValueError(f"source campaign does not contain exactly one {key}")
        parent_job = parent_jobs[0]
        expected_job = expected[key]
        if (parent_job.get("variant"), parent_job.get("seed")) != (
            expected_job["variant"],
            expected_job["seed"],
        ):
            raise ValueError(f"source campaign run identity mismatch for {key}")
        attempt = int(item.get("source_attempt", 0))
        latest_attempt = int(parent_job.get("attempts", 0))
        if attempt < 1 or attempt > latest_attempt:
            raise ValueError(f"source attempt is outside recorded campaign attempts for {key}")
        attempt_dir = parent_dir / "runs" / "attempts" / key / f"attempt{attempt}"
        provenance_path = attempt_dir / "attempt_provenance.json"
        if not provenance_path.is_file():
            raise ValueError(f"source attempt provenance is missing for {key}")
        provenance_sha = sha256_file(provenance_path)
        if provenance_sha != item.get("source_attempt_provenance_sha256"):
            raise ValueError(f"source attempt provenance hash mismatch for {key}")
        provenance = read_manifest(provenance_path)
        provenance_identity = provenance.get("identity")
        provenance_fingerprint = (
            hashlib.sha256(json.dumps(provenance_identity, sort_keys=True).encode()).hexdigest()
            if isinstance(provenance_identity, dict)
            else ""
        )
        if (
            provenance.get("campaign_fingerprint") != source_fingerprint
            or provenance_fingerprint != source_fingerprint
            or provenance.get("run_key") != key
            or provenance.get("attempt") != attempt
            or provenance.get("variant") != expected_job["variant"]
            or provenance.get("seed") != expected_job["seed"]
        ):
            raise ValueError(f"source attempt provenance identity mismatch for {key}")
        checkpoint = pathlib.Path(checkpoint_raw)
        if not checkpoint.is_absolute():
            checkpoint = map_path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        try:
            checkpoint.relative_to(attempt_dir.resolve())
        except ValueError:
            raise ValueError(
                f"source checkpoint is outside the recorded attempt for {key}"
            ) from None
        match = re.fullmatch(r"ppo_ckpt_(\d+)_steps\.zip", checkpoint.name)
        if match is None:
            raise ValueError(f"source checkpoint must be a numbered PPO checkpoint for {key}")
        source_steps = int(match.group(1))
        if source_steps != int(item.get("source_num_timesteps", -1)):
            raise ValueError(f"source checkpoint step metadata mismatch for {key}")
        if source_steps > int(provenance.get("requested_timesteps", 0)):
            raise ValueError(f"source checkpoint exceeds its parent budget for {key}")
        if source_steps <= 0 or source_steps > target_steps:
            raise ValueError(f"source checkpoint step count is outside target budget for {key}")
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != item.get("checkpoint_sha256") or not zipfile.is_zipfile(checkpoint):
            raise ValueError(f"source checkpoint is corrupt or hash-mismatched for {key}")
        with zipfile.ZipFile(checkpoint) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"source checkpoint zip CRC check failed for {key}")
            try:
                checkpoint_data = json.loads(archive.read("data"))
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"source checkpoint metadata is unreadable for {key}") from exc
        if (
            not isinstance(checkpoint_data, dict)
            or checkpoint_data.get("num_timesteps") != source_steps
        ):
            raise ValueError(f"source checkpoint step count disagrees with its filename for {key}")
        by_key[key] = {
            "key": key,
            "source_campaign_dir": str(parent_dir),
            "source_campaign_id": parent_dir.name,
            "source_campaign_git_sha": source_git_sha,
            "source_requested_timesteps": source_budget,
            "source_campaign_fingerprint": source_fingerprint,
            "source_attempt": attempt,
            "source_attempt_provenance_sha256": provenance_sha,
            "source_num_timesteps": source_steps,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
    missing = set(expected) - set(by_key)
    if missing:
        raise ValueError(f"initial checkpoint map is missing run keys: {sorted(missing)}")
    return by_key


def build_initial_checkpoint_map(
    source_campaign_dirs: list[pathlib.Path],
    required_keys: set[str],
    target_steps: int,
) -> dict[str, Any]:
    """Build an explicit, checksummed parent-lineage map for incomplete runs."""
    records: dict[str, dict[str, Any]] = {}
    for source_dir_raw in source_campaign_dirs:
        source_dir = source_dir_raw.resolve()
        manifest = read_manifest(source_dir / "campaign.json")
        snapshot = manifest.get("source_snapshot")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
            raise ValueError(f"source campaign snapshot is missing: {source_dir}")
        snapshot_path = (source_dir / snapshot["path"]).resolve()
        if snapshot_path.parent != source_dir:
            raise ValueError(f"source snapshot path escapes campaign: {source_dir}")
        valid, reason = verify_campaign_source_snapshot(snapshot_path, snapshot)
        if not valid:
            raise ValueError(f"source campaign snapshot is invalid: {reason}")
        try:
            with zipfile.ZipFile(snapshot_path) as archive:
                archived_source = json.loads(archive.read("source_manifest.json"))
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            raise ValueError(
                f"source campaign snapshot manifest is unreadable: {source_dir}"
            ) from exc
        if archived_source.get("files") != manifest.get("source_hashes") or archived_source.get(
            "git_sha"
        ) != manifest.get("git_sha"):
            raise ValueError(f"source campaign snapshot/manifest disagree: {source_dir}")
        fingerprint = str(manifest.get("fingerprint", ""))
        if not fingerprint:
            raise ValueError(f"source campaign fingerprint is missing: {source_dir}")
        for job in manifest.get("jobs", []):
            key = str(job.get("key", ""))
            if key not in required_keys:
                continue
            latest_attempt = int(job.get("attempts", 0))
            if latest_attempt < 1:
                continue
            attempt = 0
            checkpoint = None
            for candidate_attempt in range(latest_attempt, 0, -1):
                candidate_checkpoint = latest_resumable_checkpoint(
                    source_dir, key, candidate_attempt, fingerprint, max_steps=target_steps
                )
                if candidate_checkpoint is not None:
                    attempt = candidate_attempt
                    checkpoint = candidate_checkpoint
                    break
            if checkpoint is None:
                continue
            match = re.fullmatch(r"ppo_ckpt_(\d+)_steps\.zip", checkpoint.name)
            if match is None:
                continue
            provenance_path = (
                source_dir
                / "runs"
                / "attempts"
                / key
                / f"attempt{attempt}"
                / "attempt_provenance.json"
            )
            candidate = {
                "key": key,
                "source_campaign_dir": str(source_dir),
                "source_campaign_id": source_dir.name,
                "source_campaign_git_sha": str(manifest.get("git_sha", "")),
                "source_requested_timesteps": int(manifest.get("timesteps", 0)),
                "source_campaign_fingerprint": fingerprint,
                "source_attempt": attempt,
                "source_attempt_provenance_sha256": sha256_file(provenance_path),
                "source_num_timesteps": int(match.group(1)),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            previous = records.get(key)
            if (
                previous is None
                or candidate["source_num_timesteps"] > previous["source_num_timesteps"]
            ):
                records[key] = candidate
    missing = required_keys - set(records)
    extra = set(records) - required_keys
    if missing or extra:
        raise ValueError(
            f"source campaigns do not cover required keys exactly; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return {"schema_version": 1, "checkpoints": [records[key] for key in sorted(records)]}


def archive_initial_checkpoints(
    campaign_out: pathlib.Path, checkpoints: dict[str, dict[str, Any]]
) -> dict[str, pathlib.Path]:
    """Copy validated bootstrap checkpoints into the successor campaign."""
    archived: dict[str, pathlib.Path] = {}
    for key, checkpoint in checkpoints.items():
        destination = campaign_out / "bootstrap_checkpoints" / f"{key}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = str(checkpoint["checkpoint_sha256"])
        if destination.is_file():
            if sha256_file(destination) != expected_sha:
                raise RuntimeError(f"archived bootstrap checkpoint changed for {key}")
        else:
            shutil.copy2(pathlib.Path(str(checkpoint["checkpoint_path"])), destination)
            if sha256_file(destination) != expected_sha:
                destination.unlink(missing_ok=True)
                raise RuntimeError(f"bootstrap checkpoint copy failed verification for {key}")
        archived[key] = destination
    return archived


def choose_resume_checkpoint(
    campaign_out: pathlib.Path,
    key: str,
    previous_attempt: int | None,
    fingerprint: str,
    archived_bootstrap: dict[str, pathlib.Path],
) -> tuple[pathlib.Path | None, str | None]:
    """Prefer this campaign's latest checkpoint, else its verified parent input."""
    if previous_attempt is not None:
        checkpoint = latest_resumable_checkpoint(campaign_out, key, previous_attempt, fingerprint)
        if checkpoint is not None:
            return checkpoint, "interrupted_campaign_attempt"
    checkpoint = archived_bootstrap.get(key)
    if checkpoint is not None:
        return checkpoint, "parent_campaign_checkpoint"
    return None, None


def verify_initial_checkpoint_models(
    archived: dict[str, pathlib.Path], lineage: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Deserialize migrated PPO checkpoints on CPU and check saved step counts."""
    import gc

    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from learning.rl.checkpoint import load_sb3_checkpoint

    loaded_steps: dict[str, int] = {}
    for key, path in archived.items():
        model = load_sb3_checkpoint(path, device="cpu")
        actual_steps = int(model.num_timesteps)
        expected_steps = int(lineage[key]["source_num_timesteps"])
        if actual_steps != expected_steps:
            raise ValueError(
                f"loaded checkpoint step mismatch for {key}: "
                f"expected {expected_steps}, got {actual_steps}"
            )
        loaded_steps[key] = actual_steps
        del model
        gc.collect()
    return loaded_steps
