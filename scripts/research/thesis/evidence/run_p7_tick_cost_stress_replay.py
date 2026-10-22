#!/usr/bin/env python3
"""Replay the frozen P7 checkpoints at 5- and 10-point tick-fill costs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1"
CAMPAIGN_FINGERPRINT = "10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab"
SOURCE_PAYLOAD = "p7_paired_seed_metrics_exploratory_20261022_v2.json"
SEEDS = (101, 102, 103, 104, 105)
COST_LEVELS = {"baseline_5pt": 5.0, "stress_10pt": 10.0}
REPRODUCIBILITY_SEED = 101
CORE_OUTPUTS = (
    "metrics.json",
    "equity.csv",
    "trades.csv",
    "fills.csv",
    "breach_events.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(arm: dict[str, Any], suffix: str) -> tuple[Path, str]:
    records = arm.get("artifact_sources")
    if not isinstance(records, list):
        raise ValueError(f"{arm.get('run_id')}: artifact source list is missing")
    found = [item for item in records if str(item.get("path", "")).endswith(suffix)]
    if len(found) != 1:
        raise ValueError(f"{arm.get('run_id')}: expected one {suffix}, found {len(found)}")
    return Path(found[0]["path"]), str(found[0]["sha256"])


def _validate_source(
    repo: Path,
    campaign: dict[str, Any],
    arm: dict[str, Any],
    seed: int,
) -> tuple[Path, Path, str]:
    if arm.get("status") != "succeeded" or arm.get("evidence_role") != "matched_exploratory":
        raise ValueError(f"seed {seed}: source run is not a succeeded exploratory artifact")
    provenance = arm.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
    ):
        raise ValueError(f"seed {seed}: source campaign fingerprint does not match")
    if provenance.get("seed") != seed:
        raise ValueError(f"seed {seed}: payload seed differs from run provenance")

    manifest_rel, manifest_sha = _artifact(arm, "/run_manifest.json")
    checkpoint_rel, checkpoint_sha = _artifact(arm, "/model/ppo_final.zip")
    manifest_path = (repo / manifest_rel).resolve()
    checkpoint_path = (repo / checkpoint_rel).resolve()
    campaign_root = (repo / "outputs/gpu_campaigns" / CAMPAIGN_ID).resolve()
    if campaign_root not in manifest_path.parents or campaign_root not in checkpoint_path.parents:
        raise ValueError(f"seed {seed}: source artifact escapes campaign directory")
    if _sha256(manifest_path) != manifest_sha or _sha256(checkpoint_path) != checkpoint_sha:
        raise ValueError(f"seed {seed}: source manifest/checkpoint hash changed")

    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("training", {}).get("seed", seed) not in (seed, None):
        raise ValueError(f"seed {seed}: run manifest records a different training seed")
    job_key = checkpoint_path.parents[2].name
    job = next((item for item in campaign["jobs"] if item.get("key") == job_key), None)
    if job is None or job.get("status") != "succeeded" or int(job.get("seed", -1)) != seed:
        raise ValueError(f"seed {seed}: campaign terminal record does not match checkpoint")
    attempt_dir = checkpoint_path.parents[3]
    if attempt_dir.name != f"attempt{job.get('attempts')}":
        raise ValueError(f"seed {seed}: checkpoint is not from the latest campaign attempt")
    return manifest_path.parent, checkpoint_path, checkpoint_sha


def _run_one(
    repo: Path,
    source_run: Path,
    checkpoint: Path,
    checkpoint_sha: str,
    seed: int,
    arm_name: str,
    cost_name: str,
    slippage_points: float,
    repetition: int,
) -> dict[str, Any]:
    output = (
        repo
        / "outputs/gpu_campaigns"
        / CAMPAIGN_ID
        / "reports/p7_tick_cost_stress_20261022"
        / cost_name
        / f"replay{repetition}"
        / f"{arm_name}_seed{seed}"
    )
    manifest_path = output / "reanalysis_manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"incomplete prior output exists; refusing to overwrite: {output}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        override = existing.get("execution_cost_override", {})
        if (
            existing.get("checkpoint_sha256") != checkpoint_sha
            or existing.get("source_run") != str(source_run)
            or float(override.get("effective_tick_fill_slippage_points", -1)) != slippage_points
        ):
            raise FileExistsError(f"existing reanalysis has different identity: {output}")
        metrics_path = output / "testing/metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"completed reanalysis metrics are missing: {metrics_path}")
    else:
        command = [
            sys.executable,
            "-m",
            "evaluation.reanalysis.runner",
            "--run",
            str(source_run),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--allow-historical",
            "--seed",
            str(seed),
            "--tick-fill-slippage-points",
            str(slippage_points),
            "--max-charts",
            "0",
            "--no-save-html",
        ]
        print(f"RUN {cost_name} {arm_name} seed={seed}", flush=True)
        subprocess.run(command, cwd=repo, check=True)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"reanalysis did not produce its manifest: {manifest_path}")
        if _sha256(checkpoint) != checkpoint_sha:
            raise ValueError("source checkpoint changed during reanalysis")

    metrics_path = output / "testing/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    core_hashes: dict[str, str] = {}
    for split in ("training", "testing"):
        for name in CORE_OUTPUTS:
            artifact = output / split / name
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"required numerical replay artifact is missing: {artifact}"
                )
            core_hashes[f"{split}/{name}"] = _sha256(artifact)
    return {
        "seed": seed,
        "training_arm": arm_name,
        "cost_level": cost_name,
        "slippage_points": slippage_points,
        "repetition": repetition,
        "source_run": str(source_run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "reanalysis_manifest": str(manifest_path),
        "reanalysis_manifest_sha256": _sha256(manifest_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "metrics": metrics,
        "core_output_hashes": core_hashes,
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[4]
    campaign_root = repo / "outputs/gpu_campaigns" / CAMPAIGN_ID
    campaign = json.loads((campaign_root / "campaign.json").read_text(encoding="utf-8"))
    if campaign.get("fingerprint") != CAMPAIGN_FINGERPRINT:
        raise ValueError("P7 campaign fingerprint changed")
    if any(job.get("status") != "succeeded" for job in campaign.get("jobs", [])):
        raise ValueError("P7 campaign is not terminal with all succeeded jobs")

    source_path = campaign_root / "reports" / SOURCE_PAYLOAD
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if source_payload.get("expected_seeds") != list(SEEDS):
        raise ValueError("P7 source payload seed set changed")
    pairs = {int(pair["seed"]): pair for pair in source_payload["pairs"]}
    if set(pairs) != set(SEEDS):
        raise ValueError("P7 source payload does not contain exactly seeds 101-105")

    ledger_path = campaign_root / "reports/p7_tick_cost_stress_20261022/replay_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if previous.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT or previous.get(
            "source_payload_sha256"
        ) != _sha256(source_path):
            raise ValueError("existing replay ledger does not match the frozen P7 input")
        results = list(previous.get("results", []))
    complete_keys = {
        (
            item["cost_level"],
            item["training_arm"],
            int(item["seed"]),
            int(item["repetition"]),
        )
        for item in results
    }

    for seed in SEEDS:
        for arm_name in ("control", "treatment"):
            arm = pairs[seed][arm_name]
            source_run, checkpoint, checkpoint_sha = _validate_source(repo, campaign, arm, seed)
            for cost_name, slippage_points in COST_LEVELS.items():
                repetitions = (1, 2) if seed == REPRODUCIBILITY_SEED else (1,)
                for repetition in repetitions:
                    key = (cost_name, arm_name, seed, repetition)
                    if key in complete_keys:
                        continue
                    result = _run_one(
                        repo,
                        source_run,
                        checkpoint,
                        checkpoint_sha,
                        seed,
                        arm_name,
                        cost_name,
                        slippage_points,
                        repetition,
                    )
                    results.append(result)
                    complete_keys.add(key)
                    ledger = {
                        "schema": "aalto.p7_tick_cost_stress_replay_ledger.v1",
                        "campaign_id": CAMPAIGN_ID,
                        "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
                        "source_payload": str(source_path.relative_to(repo)),
                        "source_payload_sha256": _sha256(source_path),
                        "requested_cost_levels": COST_LEVELS,
                        "reproducibility_seeds": [REPRODUCIBILITY_SEED],
                        "completed_replays": len(results),
                        "expected_replays": len(SEEDS) * 2 * len(COST_LEVELS)
                        + len(COST_LEVELS) * 2,
                        "results": results,
                    }
                    temp_path = ledger_path.with_suffix(".json.tmp")
                    temp_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
                    temp_path.replace(ledger_path)
    expected_replays = len(SEEDS) * 2 * len(COST_LEVELS) + len(COST_LEVELS) * 2
    if len(results) != expected_replays:
        raise ValueError(f"expected {expected_replays} completed replays, found {len(results)}")
    for arm_name in ("control", "treatment"):
        for cost_name in COST_LEVELS:
            first = next(
                item
                for item in results
                if item["training_arm"] == arm_name
                and item["cost_level"] == cost_name
                and int(item["seed"]) == REPRODUCIBILITY_SEED
                and int(item["repetition"]) == 1
            )
            repeated = next(
                item
                for item in results
                if item["training_arm"] == arm_name
                and item["cost_level"] == cost_name
                and int(item["seed"]) == REPRODUCIBILITY_SEED
                and int(item["repetition"]) == 2
            )
            if first["core_output_hashes"] != repeated["core_output_hashes"]:
                raise ValueError(
                    f"repeat replay mismatch: {arm_name} {cost_name} seed {REPRODUCIBILITY_SEED}"
                )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["reproducibility_checks"] = {
        f"{arm_name}/{cost_name}/seed{REPRODUCIBILITY_SEED}": "exact_core_artifact_hash_match"
        for arm_name in ("control", "treatment")
        for cost_name in COST_LEVELS
    }
    temp_path = ledger_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    temp_path.replace(ledger_path)
    print(f"P7 cost-stress replays complete: {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
