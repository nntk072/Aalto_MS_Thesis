#!/usr/bin/env python3
"""Record and apply a campaign-identity migration before any trainer attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import socket
import time
import zipfile
from typing import Any

import yaml

from experiments.campaigns.execution import (
    ROOT,
    atomic_json,
    campaign_data_inputs,
    campaign_source_files,
    expand_slurm_hosts,
    read_manifest,
    runtime_metadata,
    sha256_file,
    validate_campaign_evaluation,
    verify_campaign_source_snapshot,
    write_campaign_source_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-spec", required=True, type=pathlib.Path)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--worker-cpus", required=True, type=int)
    parser.add_argument("--worker-gpu-type", required=True)
    parser.add_argument("--worker-gpus-per-node", required=True, type=int)
    parser.add_argument("--fast-reports", action="store_true")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not job_id or not node_list or os.environ.get("SLURM_NNODES") != "2":
        raise RuntimeError("preflight migration must run in the approved two-node allocation")
    if args.workers != 2 or args.worker_cpus != 47 or args.worker_gpus_per_node != 1:
        raise RuntimeError("migration parameters differ from the reviewed P7 allocation")

    nodes = expand_slurm_hosts(node_list)
    if nodes != ["gpuarm1", "gpuarm2"]:
        raise RuntimeError(f"unexpected P7 allocation nodes: {nodes}")

    out = ROOT / "outputs" / "gpu_campaigns" / args.campaign_id
    manifest_path = out / "campaign.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"campaign manifest is missing: {manifest_path}")
    manifest = read_manifest(manifest_path)
    spec_path = args.campaign_spec.resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("campaign_id") != args.campaign_id:
        raise ValueError("campaign specification does not match the campaign ID")

    variants_path = (
        ROOT / str(spec.get("variant_file", "config/campaigns/variant_list.txt"))
    ).resolve()
    variants = [
        line.strip()
        for line in variants_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if variants != spec.get("variants"):
        raise ValueError("campaign specification and authoritative variant file disagree")
    seeds = [int(seed) for seed in spec["seeds"]]
    expected_jobs = [f"{variant}__seed{seed}" for variant in variants for seed in seeds]
    if [job.get("key") for job in manifest.get("jobs", [])] != expected_jobs:
        raise ValueError("campaign runs differ from the frozen specification")
    evaluation = validate_campaign_evaluation(spec.get("evaluation", {}))
    n_envs = int(spec.get("n_envs", 64))
    if (
        manifest.get("timesteps") != int(spec["timesteps"])
        or manifest.get("n_envs") != n_envs
        or manifest.get("data_source") != str(spec.get("data_source", "frames"))
        or manifest.get("evaluation") != evaluation
        or manifest.get("fast_reports") is not args.fast_reports
    ):
        raise ValueError("campaign manifest no longer matches the frozen P7 specification")
    original_fingerprint = str(manifest.get("fingerprint", ""))
    if original_fingerprint != args.expected_fingerprint:
        if (
            manifest.get("workers") == args.workers
            and manifest.get("worker_nodes") == nodes
            and manifest.get("worker_cpus") == args.worker_cpus
            and manifest.get("preflight_migration")
        ):
            source_hashes = {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in campaign_source_files(ROOT, spec_path)
            }
            if source_hashes != manifest.get("source_hashes"):
                raise RuntimeError("migrated campaign source changed after fingerprint migration")
            snapshot_record = manifest.get("source_snapshot")
            if not isinstance(snapshot_record, dict):
                raise RuntimeError("migrated campaign source snapshot record is missing")
            snapshot_path = (out / str(snapshot_record.get("path", ""))).resolve()
            if snapshot_path.parent != out.resolve():
                raise RuntimeError("migrated campaign source snapshot escapes campaign directory")
            valid, reason = verify_campaign_source_snapshot(snapshot_path, snapshot_record)
            if not valid:
                raise RuntimeError(f"migrated campaign source snapshot is invalid: {reason}")
            print(
                f"preflight migration already recorded: fingerprint={original_fingerprint}; "
                f"workers={args.workers} nodes={','.join(nodes)} "
                f"cpus_per_worker={args.worker_cpus}",
                flush=True,
            )
            return 0
        raise RuntimeError(
            "manifest fingerprint is neither the preserved preflight identity "
            "nor a recognized migration"
        )

    if any(job.get("status") != "pending" or job.get("attempts") != 0 for job in manifest["jobs"]):
        raise RuntimeError("preflight migration is allowed only before the first trainer attempt")
    attempt_root = out / "runs" / "attempts"
    if attempt_root.exists() and any(attempt_root.rglob("ppo_*.zip")):
        raise RuntimeError("trainer checkpoints exist; refusing preflight-only migration")
    marker_root = out / "markers"
    if marker_root.exists() and any(marker_root.glob("*.done")):
        raise RuntimeError("completed-run markers exist; refusing preflight-only migration")

    if (
        manifest.get("workers") != 1
        or manifest.get("worker_nodes") != ["gpuarm1"]
        or manifest.get("worker_cpus") != 48
    ):
        raise RuntimeError("original campaign identity differs from the reviewed preflight")

    original_snapshot = manifest.get("source_snapshot")
    if not isinstance(original_snapshot, dict):
        raise ValueError("original source snapshot record is missing")
    original_snapshot_path = (out / str(original_snapshot.get("path", ""))).resolve()
    if original_snapshot_path.parent != out.resolve():
        raise ValueError("original source snapshot path escapes the campaign directory")
    valid, reason = verify_campaign_source_snapshot(original_snapshot_path, original_snapshot)
    if not valid:
        raise RuntimeError(f"original source snapshot is invalid: {reason}")
    with zipfile.ZipFile(original_snapshot_path) as archive:
        archived_manifest = json.loads(archive.read("source_manifest.json"))
    if archived_manifest.get("files") != manifest.get("source_hashes"):
        raise RuntimeError("original campaign source snapshot and manifest disagree")

    old_manifest_copy = out / "reports" / f"preflight_manifest_{original_fingerprint}.json"
    old_snapshot_copy = out / f"source_snapshot_preflight_{original_fingerprint}.zip"
    old_manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (manifest_path, old_manifest_copy),
        (original_snapshot_path, old_snapshot_copy),
    ):
        if destination.exists():
            if sha256_file(source) != sha256_file(destination):
                raise RuntimeError(f"preserved preflight artifact differs: {destination}")
        else:
            shutil.copy2(source, destination)

    runtime = runtime_metadata()
    if runtime.get("machine") != "aarch64":
        raise RuntimeError(
            f"P7 workers require the native GH200 architecture: {runtime.get('machine')}"
        )
    raw_config = yaml.safe_load((ROOT / "config" / "base" / "default.yaml").read_text())
    data_inputs = campaign_data_inputs(raw_config["data"])
    if data_inputs != manifest.get("data_inputs"):
        raise RuntimeError("configured market-data inputs changed since the preflight fingerprint")

    source_files = campaign_source_files(ROOT, spec_path)
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    identity: dict[str, Any] = {
        "jobs": expected_jobs,
        "steps": int(spec["timesteps"]),
        "data_source": str(spec.get("data_source", "frames")),
        "workers": args.workers,
        "n_envs": n_envs,
        "worker_placement": {
            "mode": "slurm-node-steps",
            "node_count": len(nodes),
            "gpu_type": args.worker_gpu_type,
            "gpus_per_node": args.worker_gpus_per_node,
            "gpu_binding": "least-loaded-map_gpu",
            "cpus_per_worker": args.worker_cpus,
            "cpu_binding": "unique-mask-per-worker",
        },
        "fast_reports": args.fast_reports,
        "evaluation": evaluation,
        "runtime_compatibility": {
            "python": runtime["python"],
            "machine": runtime["machine"],
            "packages": runtime["packages"],
        },
        "git_sha": str(manifest["git_sha"]),
        "source_hashes": source_hashes,
        "data_inputs": data_inputs,
        "initial_checkpoint_lineage": manifest.get("initial_checkpoint_lineage", {}),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    snapshot_path = out / f"source_snapshot_{fingerprint}.zip"
    snapshot_record = write_campaign_source_snapshot(
        ROOT,
        source_files,
        snapshot_path,
        git_sha=str(manifest["git_sha"]),
        expected_source_hashes=source_hashes,
    )
    migration = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign_id": args.campaign_id,
        "reason": (
            "P7 allocation was corrected to two workers on the approved two-node shape; "
            "each 48-CPU task reserves one CPU for the campaign resource sampler."
        ),
        "trainer_attempts_before_migration": 0,
        "previous_fingerprint": original_fingerprint,
        "new_fingerprint": fingerprint,
        "previous_identity": {
            "workers": 1,
            "worker_nodes": ["gpuarm1"],
            "worker_cpus": 48,
        },
        "new_identity": identity,
        "preserved_manifest": str(old_manifest_copy.relative_to(out)),
        "preserved_source_snapshot": old_snapshot_copy.name,
        "new_source_snapshot": snapshot_record,
        "allocation_job_id": job_id,
        "allocation_nodes": nodes,
    }
    migration_path = (
        out / "reports" / f"preflight_migration_{original_fingerprint}_to_{fingerprint}.json"
    )
    if migration_path.exists():
        existing = json.loads(migration_path.read_text(encoding="utf-8"))
        if (
            existing.get("previous_fingerprint") != original_fingerprint
            or existing.get("new_fingerprint") != fingerprint
        ):
            raise RuntimeError("existing preflight migration record conflicts with this migration")
    else:
        atomic_json(migration_path, migration)

    manifest.update(
        {
            "fingerprint": fingerprint,
            "source_hashes": source_hashes,
            "source_snapshot": snapshot_record,
            "workers": args.workers,
            "worker_nodes": nodes,
            "worker_cpus": args.worker_cpus,
            "worker_gpu_type": args.worker_gpu_type,
            "worker_gpus_per_node": args.worker_gpus_per_node,
            "gpu_binding": "least-loaded-map_gpu",
            "runtime": runtime,
            "job_id": job_id,
            "hostname": socket.gethostname(),
            "preflight_migration": str(migration_path.relative_to(out)),
        }
    )
    atomic_json(manifest_path, manifest)
    print(
        f"preflight migration recorded: {original_fingerprint} -> {fingerprint}; "
        f"attempts=0 workers={args.workers} nodes={','.join(nodes)} "
        f"cpus_per_worker={args.worker_cpus}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
