#!/usr/bin/env python3
"""Build a provenance-checked exploratory payload for the frozen P7 seed pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1"
CAMPAIGN_FINGERPRINT = "10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab"
EXPECTED_SEEDS = (101, 102, 103, 104, 105)
REQUESTED_STEPS = 20_000_000
CONTROL_PREFIX = "ladder_a4b_entry_fsm__seed"
TREATMENT_PREFIX = "ladder_a4a_entry_fsm_hidden__seed"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash required artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _identity_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True).encode("utf-8"))


def _source_record(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display_path = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _run_paths(campaign_dir: Path, job: dict[str, Any]) -> dict[str, Path]:
    key = str(job["key"])
    attempt_number = int(job["attempts"])
    attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{attempt_number}"
    provenance_path = attempt_dir / "attempt_provenance.json"
    run_parent = attempt_dir / key
    manifests = sorted(run_parent.glob("*/run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one final run manifest for {key}, found {len(manifests)}")
    run_manifest_path = manifests[0]
    run_root = run_manifest_path.parent
    return {
        "attempt_dir": attempt_dir,
        "provenance": provenance_path,
        "run_manifest": run_manifest_path,
        "metrics": run_root / "testing" / "metrics.json",
        "resolved_config": run_parent / "resolved_config.json",
    }


def _arm_name_and_seed(key: str) -> tuple[str, int]:
    for prefix, arm in ((CONTROL_PREFIX, "control"), (TREATMENT_PREFIX, "treatment")):
        if key.startswith(prefix):
            suffix = key.removeprefix(prefix)
            if suffix.isdigit():
                return arm, int(suffix)
    raise ValueError(f"unexpected P7 run key: {key}")


def _base_provenance(
    *,
    seed: int,
    run_id: str,
    run_manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    training = run_manifest["training_protocol"]
    datasets = run_manifest["datasets"]
    execution = run_manifest["execution_protocol"]
    architecture = {
        key: value
        for key, value in run_manifest["training_architecture"].items()
        if key != "policy_parameter_count"
    }
    execution_without_costs = {key: value for key, value in execution.items() if key != "costs"}
    test_bars = datasets["test_bars"]
    return {
        "dataset_sha256": _identity_hash(datasets),
        "split_id": _identity_hash(run_manifest["split"]),
        "features_sha256": str(run_manifest["sequence_feature_schema"]["sha256"]),
        "reward_sha256": _identity_hash(
            {"reward": training["reward"], "constrained_rl": training["constrained_rl"]}
        ),
        "costs_sha256": _identity_hash(
            {"execution_costs": execution["costs"], "risk": training["risk"]}
        ),
        "execution_sha256": _identity_hash(execution_without_costs),
        "timesteps": REQUESTED_STEPS,
        "seed": seed,
        "initial_capital": float(metrics["initial_balance"]),
        "model_settings_sha256": _identity_hash(architecture),
        "environment_sha256": _identity_hash(training["environment"]),
        "evaluation_start": str(test_bars["start"]),
        "evaluation_end": str(test_bars["end"]),
        "run_id": run_id,
    }


def build_p7_payload(campaign_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Validate complete campaign lineage before reading or deriving any metrics."""
    campaign_path = campaign_dir / "campaign.json"
    campaign = _read_json(campaign_path)
    if campaign.get("fingerprint") != CAMPAIGN_FINGERPRINT:
        raise ValueError("campaign fingerprint does not match the frozen P7 campaign")
    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2 * len(EXPECTED_SEEDS):
        raise ValueError("P7 campaign must contain exactly ten runs")
    job_by_arm_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict) or job.get("status") != "succeeded":
            raise ValueError("every P7 job must be terminal and succeeded")
        arm, seed = _arm_name_and_seed(str(job.get("key", "")))
        pair_key = (arm, seed)
        if pair_key in job_by_arm_seed:
            raise ValueError(f"duplicate P7 run for {arm}, seed {seed}")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected P7 seed: {seed}")
        job_by_arm_seed[pair_key] = job
    expected_keys = {(arm, seed) for arm in ("control", "treatment") for seed in EXPECTED_SEEDS}
    if set(job_by_arm_seed) != expected_keys:
        raise ValueError("P7 campaign does not contain exactly the frozen arms and seeds")

    source_records: list[dict[str, Any]] = [_source_record(campaign_path, repo_root)]
    source_snapshot = campaign.get("source_snapshot")
    if not isinstance(source_snapshot, dict) or not isinstance(source_snapshot.get("path"), str):
        raise ValueError("campaign source snapshot record is missing")
    snapshot_path = campaign_dir / source_snapshot["path"]
    snapshot_record = _source_record(snapshot_path, repo_root)
    if snapshot_record["sha256"] != source_snapshot.get("sha256"):
        raise ValueError("campaign source snapshot hash mismatch")
    source_records.append(snapshot_record)

    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for arm_seed, job in job_by_arm_seed.items():
        arm, seed = arm_seed
        paths = _run_paths(campaign_dir, job)
        provenance = _read_json(paths["provenance"])
        run_manifest = _read_json(paths["run_manifest"])
        if (
            provenance.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
            or _identity_hash(provenance.get("identity")) != CAMPAIGN_FINGERPRINT
            or provenance.get("campaign_id") != CAMPAIGN_ID
            or provenance.get("run_key") != job["key"]
            or int(provenance.get("attempt", -1)) != int(job["attempts"])
            or int(provenance.get("seed", -1)) != seed
            or int(run_manifest.get("seed", -1)) != seed
        ):
            raise ValueError(f"campaign provenance identity mismatch for {job['key']}")
        if int(provenance.get("requested_timesteps", -1)) != REQUESTED_STEPS:
            raise ValueError(f"campaign request budget mismatch for {job['key']}")
        training = run_manifest.get("training_protocol", {})
        if (
            int(training.get("requested_steps", -1)) != REQUESTED_STEPS
            or int(training.get("completed_steps", 0)) < REQUESTED_STEPS
        ):
            raise ValueError(f"run did not reach its requested training budget: {job['key']}")
        expected_observation = arm == "control"
        environment = training.get("environment", {})
        if environment.get("entry_state_observation") is not expected_observation:
            raise ValueError(f"entry-state intervention mismatch for {job['key']}")

        checkpoint = run_manifest.get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        checkpoint_record = _source_record(checkpoint_path, repo_root)
        if checkpoint_record["sha256"] != checkpoint.get("sha256"):
            raise ValueError(f"final checkpoint hash mismatch for {job['key']}")
        learning_artifact = run_manifest.get("learning_result", {}).get("artifact", {})
        if learning_artifact.get("sha256") != checkpoint_record["sha256"]:
            raise ValueError(f"learning-result checkpoint hash mismatch for {job['key']}")

        resume_checkpoint = provenance.get("resume_checkpoint")
        if resume_checkpoint:
            resume_record = _source_record(Path(str(resume_checkpoint)), repo_root)
            if resume_record["sha256"] != provenance.get("resume_checkpoint_sha256"):
                raise ValueError(f"resume checkpoint hash mismatch for {job['key']}")
            source_records.append(resume_record)

        for source_name, source_path in paths.items():
            if source_name in {"attempt_dir", "metrics"}:
                continue
            source_records.append(_source_record(source_path, repo_root))
        source_records.append(checkpoint_record)
        loaded[arm_seed] = {
            "arm": arm,
            "seed": seed,
            "job": job,
            "paths": paths,
            "provenance_record": _source_record(paths["provenance"], repo_root),
            "run_manifest_record": _source_record(paths["run_manifest"], repo_root),
            "checkpoint_record": checkpoint_record,
            "provenance": provenance,
            "run_manifest": run_manifest,
        }

    # Validate paired settings before opening the report metrics.
    for seed in EXPECTED_SEEDS:
        control = loaded[("control", seed)]["run_manifest"]
        treatment = loaded[("treatment", seed)]["run_manifest"]
        for field in ("datasets", "split", "sequence_feature_schema", "execution_protocol"):
            if control.get(field) != treatment.get(field):
                raise ValueError(f"control/treatment mismatch in {field} for seed {seed}")
        control_environment = control["training_protocol"]["environment"]
        treatment_environment = treatment["training_protocol"]["environment"]
        if set(control_environment) != set(treatment_environment):
            raise ValueError(f"environment schema differs within seed pair {seed}")
        differing = {
            key
            for key in control_environment
            if control_environment[key] != treatment_environment[key]
        }
        if differing != {"entry_state_observation"}:
            raise ValueError(
                f"unexpected environment differences for seed {seed}: {sorted(differing)}"
            )
        control_arch = {
            key: value
            for key, value in control["training_architecture"].items()
            if key != "policy_parameter_count"
        }
        treatment_arch = {
            key: value
            for key, value in treatment["training_architecture"].items()
            if key != "policy_parameter_count"
        }
        if control_arch != treatment_arch:
            raise ValueError(
                f"architecture mismatch beyond input-induced parameter count for seed {seed}"
            )

    # Metrics are read only after the campaign, attempt, data, config, and checkpoint gates pass.
    pairs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        pair: dict[str, Any] = {"seed": seed}
        for arm in ("control", "treatment"):
            item = loaded[(arm, seed)]
            metrics = _read_json(item["paths"]["metrics"])
            pnl = float(metrics["total_pnl"])
            capital = float(metrics["initial_balance"])
            stored_return = float(metrics["total_return_pct"])
            if not (math.isfinite(pnl) and math.isfinite(capital) and capital > 0):
                raise ValueError(f"invalid PnL or initial capital for seed {seed}, arm {arm}")
            computed_return = 100.0 * pnl / capital
            if not math.isclose(computed_return, stored_return, rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(
                    f"stored total_return_pct does not reconcile for seed {seed}, arm {arm}"
                )

            provenance = _base_provenance(
                seed=seed,
                run_id=item["paths"]["run_manifest"].parent.name,
                run_manifest=item["run_manifest"],
                metrics=metrics,
            )
            provenance.update(
                {
                    "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
                    "attempt": int(item["job"]["attempts"]),
                    "slurm_job_id": item["provenance"].get("slurm_job_id"),
                    "resume_from_attempt": item["provenance"].get("resume_from_attempt"),
                    "resume_checkpoint_sha256": item["provenance"].get("resume_checkpoint_sha256"),
                }
            )
            pair[arm] = {
                "run_id": provenance["run_id"],
                "status": "succeeded",
                "evidence_role": "matched_exploratory",
                "provenance": provenance,
                "metrics": {"net_return_pct": computed_return},
                "pnl": pnl,
                "initial_balance": capital,
                "trade_count": int(metrics["n_trades"]),
                "completed_steps": item["run_manifest"]["training_protocol"]["completed_steps"],
                "checkpoint_sha256": item["checkpoint_record"]["sha256"],
                "artifact_sources": [
                    item["provenance_record"],
                    item["run_manifest_record"],
                    _source_record(item["paths"]["resolved_config"], repo_root),
                    _source_record(item["paths"]["metrics"], repo_root),
                    item["checkpoint_record"],
                ],
            }
        control_prov = pair["control"]["provenance"]
        treatment_prov = pair["treatment"]["provenance"]
        for field in (
            "dataset_sha256",
            "split_id",
            "features_sha256",
            "reward_sha256",
            "costs_sha256",
            "execution_sha256",
            "timesteps",
            "seed",
            "initial_capital",
            "model_settings_sha256",
            "evaluation_start",
            "evaluation_end",
        ):
            if control_prov[field] != treatment_prov[field]:
                raise ValueError(f"matched provenance differs for seed {seed}: {field}")
        if control_prov["environment_sha256"] == treatment_prov["environment_sha256"]:
            raise ValueError(f"declared environment intervention is absent for seed {seed}")
        pairs.append(pair)

    return {
        "schema": "aalto.paired_seed_metrics.v1",
        "comparison_id": CAMPAIGN_ID,
        "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
        "evidence_role": "matched_exploratory",
        "evaluation_exposure": "exposed_report_only",
        "expected_seeds": list(EXPECTED_SEEDS),
        "intervention_field": "environment_sha256",
        "metrics": ["net_return_pct"],
        "pairs": pairs,
        "source_artifacts": source_records,
        "selection_note": "Pair frozen before these five-seed runs; the 2026 report interval was previously inspected.",
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=repo_root / "outputs" / "gpu_campaigns" / CAMPAIGN_ID,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_p7_payload(args.campaign_dir, repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "seed_pairs": len(payload["pairs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
