"""Snapshot full-split exposure headroom for the active resumed campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = (
    ROOT / "outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
)
CAMPAIGN_PATH = CAMPAIGN_DIR / "campaign.json"
BUDGET_EVIDENCE_PATH = (
    ROOT / "docs/thesis/evidence/g1_2026-10-16/g1_full_split_budget_math_20261016.json"
)
RAW_AUDIT_PATH = ROOT / "docs/thesis/evidence/g1_2026-10-16/g1_raw_m1_quality_audit_20261016.json"
FOLD_AUDIT_PATH = ROOT / "docs/thesis/g1_real_market_fold_fit_scope_2026-10-16.json"
OUTPUT_PATH = ROOT / "docs/thesis/p1_checkpoint_full_split_budget_2026-10-16.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_total_timesteps(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    values = re.findall(r"total_timesteps\s*\|\s*([0-9,]+)", text)
    return int(values[-1].replace(",", "")) if values else None


def episode_coverage(key: str) -> tuple[list[Path], int]:
    paths = sorted(CAMPAIGN_DIR.glob(f"runs/attempts/{key}/**/episode_equity.csv"))
    completed = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                value = str(row.get("completed_data_split", "")).strip().lower()
                completed += value in {"1", "true", "yes"}
    return paths, completed


def source_lineage_coverage(
    key: str, lineage: dict[str, Any], steps_per_env: int
) -> dict[str, Any]:
    source_dir = Path(lineage["source_campaign_dir"])
    source_attempt = as_int(lineage["source_attempt"])
    completed_attempt = source_attempt - 1
    attempt_dir = source_dir / "runs/attempts" / key / f"attempt{completed_attempt}"
    episode_paths = sorted(attempt_dir.glob(f"{key}/**/model/episode_equity.csv"))
    provenance_path = (
        source_dir / "runs/attempts" / key / f"attempt{source_attempt}" / "attempt_provenance.json"
    )
    attempt_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    predecessor_provenance_path = (
        source_dir
        / "runs/attempts"
        / key
        / f"attempt{completed_attempt}"
        / "attempt_provenance.json"
    )
    predecessor_provenance = json.loads(predecessor_provenance_path.read_text(encoding="utf-8"))

    episode_rows: list[dict[str, str]] = []
    config_paths: list[Path] = []
    for episode_path in episode_paths:
        config_path = episode_path.parent.parent / "config.yaml"
        config_paths.append(config_path)
        with episode_path.open(newline="", encoding="utf-8") as stream:
            episode_rows.extend(csv.DictReader(stream))

    config_text = config_paths[0].read_text(encoding="utf-8") if config_paths else ""
    mode_match = re.search(r"(?m)^\s*episode_mode:\s*([^\s#]+)", config_text)
    cap_match = re.search(r"(?m)^\s*max_episode_steps:\s*([^\s#]+)", config_text)
    envs_match = re.search(r"(?m)^\s*n_envs:\s*(\d+)\s*$", config_text)
    configured_mode = mode_match.group(1) if mode_match else None
    configured_cap = cap_match.group(1) if cap_match else None
    configured_envs = int(envs_match.group(1)) if envs_match else None

    completed_rows = [
        row
        for row in episode_rows
        if str(row.get("completed_data_split", "")).strip().lower() in {"1", "true", "yes"}
    ]
    environment_ids = sorted(
        {
            int(row["environment_index"])
            for row in completed_rows
            if str(row.get("environment_index", "")).isdigit()
        }
    )
    episode_steps = sorted(
        {
            int(row["episode_steps"])
            for row in completed_rows
            if str(row.get("episode_steps", "")).isdigit()
        }
    )
    end_reasons = sorted({str(row.get("episode_end_reason", "")) for row in completed_rows})
    trade_counts = [int(row["n_trades"]) for row in completed_rows]
    equity_deltas = [float(row["equity_delta"]) for row in completed_rows]
    breach_counts: dict[str, int] = {}
    for row in completed_rows:
        reason = str(row.get("breach_reason", "")).strip() or "none"
        breach_counts[reason] = breach_counts.get(reason, 0) + 1

    attempt_checkpoint_paths = sorted(
        (source_dir / "runs/attempts" / key / f"attempt{source_attempt}").glob(
            f"{key}/**/model/ppo_ckpt_*_steps.zip"
        )
    )
    matching_checkpoint_paths = [
        path for path in attempt_checkpoint_paths if sha256(path) == lineage["checkpoint_sha256"]
    ]
    attempt3_resume_path = Path(attempt_provenance["resume_checkpoint"])
    if not attempt3_resume_path.exists():
        attempt3_resume_path = source_dir / attempt3_resume_path.relative_to(
            "/scratch/work/nguyenl37/Aalto_MS_Thesis-main"
        )
    resume_hash_matches = (
        attempt3_resume_path.is_file()
        and sha256(attempt3_resume_path) == attempt_provenance["resume_checkpoint_sha256"]
    )
    provenance_hash_matches = sha256(provenance_path) == lineage["source_attempt_provenance_sha256"]
    full_split_verified = (
        len(episode_paths) == 1
        and configured_mode == "full_split"
        and configured_cap == "null"
        and configured_envs == 64
        and len(episode_rows) == 64
        and len(completed_rows) == 64
        and environment_ids == list(range(64))
        and episode_steps == [steps_per_env]
        and end_reasons == ["data_end"]
        and attempt_provenance.get("resume_from_attempt") == completed_attempt
        and predecessor_provenance.get("attempt") == completed_attempt
        and resume_hash_matches
        and provenance_hash_matches
        and len(matching_checkpoint_paths) == 1
    )
    return {
        "source_campaign_id": lineage["source_campaign_id"],
        "source_campaign_fingerprint": lineage["source_campaign_fingerprint"],
        "source_campaign_git_sha": lineage["source_campaign_git_sha"],
        "completed_split_attempt": completed_attempt,
        "completed_split_config_path": str(config_paths[0].relative_to(ROOT))
        if config_paths
        else None,
        "completed_split_config_sha256": sha256(config_paths[0]) if config_paths else None,
        "configured_episode_mode": configured_mode,
        "configured_max_episode_steps": configured_cap,
        "configured_n_envs": configured_envs,
        "episode_equity_paths": [str(path.relative_to(ROOT)) for path in episode_paths],
        "episode_equity_sha256": [sha256(path) for path in episode_paths],
        "episode_row_count": len(episode_rows),
        "completed_data_split_row_count": len(completed_rows),
        "completed_environment_indices": environment_ids,
        "completed_episode_steps_values": episode_steps,
        "completed_episode_end_reasons": end_reasons,
        "training_episode_trade_counts": {
            "total": sum(trade_counts),
            "min": min(trade_counts) if trade_counts else None,
            "median": statistics.median(trade_counts) if trade_counts else None,
            "max": max(trade_counts) if trade_counts else None,
            "zero_trade_episode_count": sum(count == 0 for count in trade_counts),
        },
        "training_episode_breach_reason_counts": breach_counts,
        "training_episode_equity_delta_usd": {
            "mean": round(statistics.mean(equity_deltas), 2) if equity_deltas else None,
            "median": round(statistics.median(equity_deltas), 2) if equity_deltas else None,
        },
        "next_attempt_provenance_path": str(provenance_path.relative_to(ROOT)),
        "next_attempt_provenance_sha256": sha256(provenance_path),
        "next_attempt_provenance_hash_matches_lineage": provenance_hash_matches,
        "next_attempt_resume_from_attempt": attempt_provenance.get("resume_from_attempt"),
        "next_attempt_resume_checkpoint": str(attempt3_resume_path),
        "next_attempt_resume_checkpoint_hash_matches": resume_hash_matches,
        "current_lineage_checkpoint_matches_source_attempt": len(matching_checkpoint_paths) == 1,
        "verified_full_split_training_exposure": full_split_verified,
    }


def as_int(value: Any) -> int:
    return int(value)


def main() -> None:
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    prior_budget = json.loads(BUDGET_EVIDENCE_PATH.read_text(encoding="utf-8"))
    raw_audit = json.loads(RAW_AUDIT_PATH.read_text(encoding="utf-8"))
    fold_audit = json.loads(FOLD_AUDIT_PATH.read_text(encoding="utf-8"))
    n_envs = as_int(campaign["n_envs"])
    target = as_int(campaign["timesteps"])
    steps_per_env = as_int(prior_budget["full_split_episode_steps_per_env"])
    aggregate_episode_steps = n_envs * steps_per_env
    jobs = {job["key"]: job for job in campaign["jobs"]}
    rows: list[dict[str, Any]] = []

    for key, lineage in sorted(campaign["initial_checkpoint_lineage"].items()):
        start = as_int(lineage["source_num_timesteps"])
        current = latest_total_timesteps(Path(jobs[key]["log"]))
        fresh_env_steps = None if current is None else max(current - start, 0) // n_envs
        attainable_env_steps = max(target - start, 0) // n_envs
        telemetry_paths, completed_episodes = episode_coverage(key)
        prior_coverage = source_lineage_coverage(key, lineage, steps_per_env)
        rows.append(
            {
                "variant_key": key,
                "source_campaign_id": lineage["source_campaign_id"],
                "source_campaign_fingerprint": lineage["source_campaign_fingerprint"],
                "source_campaign_git_sha": lineage["source_campaign_git_sha"],
                "source_attempt": as_int(lineage["source_attempt"]),
                "checkpoint_sha256": lineage["checkpoint_sha256"],
                "resume_start_timesteps": start,
                "requested_final_timesteps": target,
                "latest_logged_timesteps": current,
                "observed_fresh_environment_steps_per_env": fresh_env_steps,
                "maximum_fresh_environment_steps_per_env_at_budget": attainable_env_steps,
                "train_split_steps_per_env": steps_per_env,
                "full_split_budget_possible_after_resume": attainable_env_steps >= steps_per_env,
                "current_attempt_completed_full_split_episodes": completed_episodes,
                "full_split_completed_evidence_in_current_attempt": completed_episodes > 0,
                "prior_checkpoint_lineage_coverage": prior_coverage,
                "episode_telemetry_paths": [
                    str(path.relative_to(ROOT)) for path in telemetry_paths
                ],
                "worker_log_path": str(Path(jobs[key]["log"]).relative_to(ROOT)),
                "worker_log_sha256_at_snapshot": sha256(Path(jobs[key]["log"]))
                if Path(jobs[key]["log"]).is_file()
                else None,
                "campaign_status_at_snapshot": jobs[key]["status"],
            }
        )

    episode_files = sorted(CAMPAIGN_DIR.glob("runs/attempts/**/episode_equity.csv"))
    verified_lineages = sum(
        row["prior_checkpoint_lineage_coverage"]["verified_full_split_training_exposure"]
        for row in rows
    )
    completed_prior_episodes = sum(
        row["prior_checkpoint_lineage_coverage"]["completed_data_split_row_count"] for row in rows
    )
    statuses: dict[str, int] = {}
    for job in campaign["jobs"]:
        statuses[job["status"]] = statuses.get(job["status"], 0) + 1

    now_utc = datetime.now(UTC)
    now_helsinki = now_utc.astimezone(ZoneInfo("Europe/Helsinki"))
    output = {
        "schema": "aalto.thesis.p1_checkpoint_full_split_budget.v2",
        "generated_at_utc": now_utc.isoformat(timespec="seconds"),
        "generated_at_helsinki": now_helsinki.isoformat(timespec="seconds"),
        "repository_path": str(ROOT),
        "repository_branch": "main",
        "campaign_id": CAMPAIGN_DIR.name,
        "campaign_fingerprint": campaign["fingerprint"],
        "campaign_git_sha": campaign["git_sha"],
        "campaign_manifest_path": str(CAMPAIGN_PATH.relative_to(ROOT)),
        "campaign_manifest_sha256": sha256(CAMPAIGN_PATH),
        "budget_evidence_path": str(BUDGET_EVIDENCE_PATH.relative_to(ROOT)),
        "budget_evidence_sha256": sha256(BUDGET_EVIDENCE_PATH),
        "raw_data_audit_path": str(RAW_AUDIT_PATH.relative_to(ROOT)),
        "raw_data_audit_sha256": sha256(RAW_AUDIT_PATH),
        "raw_input_sha256": {
            symbol: data["sha256"] for symbol, data in raw_audit["symbols"].items()
        },
        "fold_audit_path": str(FOLD_AUDIT_PATH.relative_to(ROOT)),
        "fold_audit_sha256": sha256(FOLD_AUDIT_PATH),
        "generator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "n_envs": n_envs,
        "requested_timesteps_per_variant": target,
        "train_split_steps_per_env": steps_per_env,
        "aggregate_steps_required_for_one_full_split": aggregate_episode_steps,
        "checkpoint_restart_semantics": (
            "PPO checkpoints restore policy/optimizer state, not vector-environment cursors; "
            "each allocation attempt starts fresh environments at the train-split beginning."
        ),
        "campaign_status_counts_at_snapshot": statuses,
        "episode_equity_file_count_at_snapshot": len(episode_files),
        "episode_equity_paths_at_snapshot": [str(path.relative_to(ROOT)) for path in episode_files],
        "source_lineages_with_verified_complete_train_split": verified_lineages,
        "prior_lineage_completed_split_episode_records": completed_prior_episodes,
        "fold_audit_status": "passed" if fold_audit.get("all_profiles_pass") else "failed",
        "fold_audit_affected_configs": fold_audit.get("affected_campaign_run_count"),
        "interpretation": (
            "The source lineage contains verified 64-environment, 99,504-step completed "
            "train-split episodes for every variant before the current attempt-3 checkpoints. "
            "This proves one chronological train-split traversal per environment in that "
            "attempt, not all-year active trading, performance, or a valid strategy result. "
            "The current resumed attempt has no completion telemetry; its 14M-start baseline "
            "has budget for at most 93,750 new steps/environment, so cannot complete a second "
            "full split before the 20M cap. Episode trade, breach, and equity fields are "
            "training behavior, not final evaluation results. The campaign remains exploratory "
            "because the real-market feature-causality audit fails."
        ),
        "variants": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
