#!/usr/bin/env python3
"""Check whether the reconciled seed-50 screen's exact checkpoints remain in outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configuration.paths import repository_root

ROOT = repository_root(__file__)
OUTPUTS = ROOT / "outputs"
RECOVERY_DIR = (
    OUTPUTS / "gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
)
SOURCE_DIR = OUTPUTS / "gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_v3_20261016"
MAINPLAN_DIR = OUTPUTS / "gpu_campaigns/ladder18_seed50_20m_full_split_n64_mainplan_20261016"
SUMMARY_PATH = RECOVERY_DIR / "reports/campaign_results_summary_v2.json"
DEFAULT_REPORT = ROOT / "docs/thesis/evidence/p3_seed50_report_pack_recoverability_20261018.json"
MAINPLAN_CHECKPOINT = "ppo_ckpt_1000000_steps.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
    ).splitlines()
    return {"head": head, "dirty_paths": dirty}


def mainplan_checkpoints(mainplan_dir: Path) -> list[Path]:
    """Find saved 1M-step checkpoints despite variable run-directory depth."""
    return sorted(mainplan_dir.rglob(MAINPLAN_CHECKPOINT))


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    campaign_path = RECOVERY_DIR / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    lineage = campaign.get("initial_checkpoint_lineage")
    if not isinstance(lineage, dict) or len(lineage) != 18:
        raise ValueError("recovery campaign must record exactly 18 source checkpoint hashes")
    if len(summary.get("jobs", [])) != 18:
        raise ValueError("screen summary must record exactly 18 variants")

    source_hashes = {
        key: record.get("checkpoint_sha256")
        for key, record in lineage.items()
        if isinstance(record, dict)
    }
    if len(source_hashes) != 18 or any(
        not isinstance(value, str) for value in source_hashes.values()
    ):
        raise ValueError("source checkpoint lineage contains malformed hashes")

    recovery_hashes: dict[str, str] = {}
    for job in campaign.get("jobs", []):
        if not isinstance(job, dict):
            continue
        for relative_path, digest in job.get("artifact_hashes", {}).items():
            if relative_path.endswith(".zip") and isinstance(digest, str):
                recovery_hashes[str(job.get("key"))] = digest
    if len(recovery_hashes) != 18:
        raise ValueError("recovery campaign must record 18 final checkpoint hashes")

    run_dirs: list[dict[str, Any]] = []
    summary_dir = Path(str(summary.get("campaign_dir", RECOVERY_DIR)))
    for job in summary["jobs"]:
        if not isinstance(job, dict):
            continue
        relative = job.get("run_artifact_dir")
        if not isinstance(relative, str):
            raise ValueError("screen summary has a malformed run_artifact_dir")
        path = summary_dir / relative
        run_dirs.append(
            {
                "variant": job.get("variant"),
                "controller_status": job.get("controller_status"),
                "path": str(path),
                "exists": path.is_dir(),
                "summary_has_train_metrics": isinstance(job.get("train_metrics"), dict),
                "summary_has_report_only_test_metrics": isinstance(
                    job.get("test_metrics_report_only"), dict
                ),
                "summary_evidence_hash_count": len(job.get("evidence_sha256", {})),
            }
        )

    expected = {
        **source_hashes,
        **{f"final::{key}": value for key, value in recovery_hashes.items()},
    }
    candidates = sorted(OUTPUTS.rglob("*.zip"))
    archive_matches: dict[str, list[str]] = {key: [] for key in expected}
    archive_bytes = 0
    for path in candidates:
        archive_bytes += path.stat().st_size
        digest = sha256(path)
        for key, expected_digest in expected.items():
            if digest == expected_digest:
                archive_matches[key].append(str(path))

    checkpoint_match_counts = {
        "source_attempt3": sum(bool(archive_matches[key]) for key in source_hashes),
        "recovery_final": sum(bool(archive_matches[f"final::{key}"]) for key in recovery_hashes),
    }
    final_checkpoint_match_map = {
        key.removeprefix("final::"): paths
        for key, paths in archive_matches.items()
        if key.startswith("final::")
    }
    source_checkpoint_match_map = {key: archive_matches[key] for key in source_hashes}
    existing_mainplan_checkpoints = mainplan_checkpoints(MAINPLAN_DIR)
    replay_manifests = sorted(
        {
            *RECOVERY_DIR.rglob("reanalysis_manifest.json"),
            *SOURCE_DIR.rglob("reanalysis_manifest.json"),
        }
    )

    return {
        "schema": "aalto.seed50_screen_report_recoverability.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository": source_state(),
        "scope": {
            "archive_inventory_root": "outputs",
            "archive_glob": "outputs/**/*.zip",
            "archive_count": len(candidates),
            "archive_bytes_scanned": archive_bytes,
            "external_paths_searched": False,
            "interpretation": (
                "This inventory tests recovery inside this checkout's outputs tree only; "
                "it does not establish that no external backup exists."
            ),
        },
        "inputs": {
            "campaign_path": str(campaign_path.relative_to(ROOT)),
            "campaign_sha256": sha256(campaign_path),
            "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
            "summary_sha256": sha256(SUMMARY_PATH),
            "audit_script_path": "scripts/research/audits/audit_seed50_screen_report_recoverability.py",
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "source_campaign_path": str(SOURCE_DIR.relative_to(ROOT)),
            "source_campaign_fingerprint": campaign.get("initial_checkpoint_lineage", {})
            .get("ladder_a0_flat__seed50", {})
            .get("source_campaign_fingerprint"),
        },
        "screen": {
            "variant_count": len(summary["jobs"]),
            "summary_train_metric_rows": sum(
                isinstance(job.get("train_metrics"), dict) for job in summary["jobs"]
            ),
            "summary_report_only_test_metric_rows": sum(
                isinstance(job.get("test_metrics_report_only"), dict) for job in summary["jobs"]
            ),
            "expected_run_artifact_directories": len(run_dirs),
            "present_run_artifact_directories": sum(item["exists"] for item in run_dirs),
            "run_artifact_directory_inventory": run_dirs,
            "source_campaign_runs_directory_exists": (SOURCE_DIR / "runs").is_dir(),
            "recovery_campaign_runs_directory_exists": (RECOVERY_DIR / "runs").is_dir(),
            "present_1m_mainplan_checkpoints": len(existing_mainplan_checkpoints),
            "mainplan_checkpoint_paths": [str(path) for path in existing_mainplan_checkpoints],
            "available_replay_manifest_count": len(replay_manifests),
            "available_replay_manifest_paths": [str(path) for path in replay_manifests],
        },
        "checkpoint_recoverability": {
            "source_attempt3_expected_count": len(source_hashes),
            "source_attempt3_hash_matches_in_outputs": checkpoint_match_counts["source_attempt3"],
            "source_attempt3_hash_matches": source_checkpoint_match_map,
            "recovery_final_expected_count": len(recovery_hashes),
            "recovery_final_hash_matches_in_outputs": checkpoint_match_counts["recovery_final"],
            "recovery_final_hash_matches": final_checkpoint_match_map,
            "disposition": (
                "exact recovery of the 18-variant raw report packs is unavailable from the "
                "recorded run directories and matching checkpoints currently present under outputs"
                if checkpoint_match_counts["source_attempt3"] == 0
                and checkpoint_match_counts["recovery_final"] == 0
                and not any(item["exists"] for item in run_dirs)
                else "partial or exact recovery material was found; inspect the recorded matches"
            ),
        },
        "limitations": [
            "The seed-50 screen remains exploratory and invalid for variant selection due to its recorded lookahead and survival-gate findings.",
            "Campaign summary metrics and their hashes are not substitutes for raw fills, positions, equity, diagnostics, manifests, and checkpoint-backed deterministic report packs.",
            "The 1M-step mainplan checkpoints do not match the recorded attempt-3 checkpoint hashes and cannot reproduce the 12M-14M-step recovery outputs.",
            "This audit performs no policy evaluation, training, deletion, overwrite, or modification of campaign files.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"report": str(output), **report["checkpoint_recoverability"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
