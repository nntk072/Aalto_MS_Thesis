#!/usr/bin/env python3
"""Durable, allocation-bounded launcher for the fixed Triton variant matrix.

Does not create or cancel Slurm allocations. Runs in an existing compute-node job.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, TextIO, cast

from experiments.campaigns.artifact_validation import (
    hash_run_artifacts as hash_run_artifacts,
)
from experiments.campaigns.artifact_validation import (
    validate_campaign_evaluation as validate_campaign_evaluation,
)
from experiments.campaigns.artifact_validation import (
    verify_run_artifacts as verify_run_artifacts,
)
from experiments.campaigns.checkpoint_lineage import (
    archive_initial_checkpoints as archive_initial_checkpoints,
)
from experiments.campaigns.checkpoint_lineage import (
    build_initial_checkpoint_map as build_initial_checkpoint_map,
)
from experiments.campaigns.checkpoint_lineage import (
    choose_resume_checkpoint as choose_resume_checkpoint,
)
from experiments.campaigns.checkpoint_lineage import (
    latest_resumable_checkpoint as latest_resumable_checkpoint,
)
from experiments.campaigns.checkpoint_lineage import (
    validate_initial_checkpoints as validate_initial_checkpoints,
)
from experiments.campaigns.checkpoint_lineage import (
    verify_initial_checkpoint_models as verify_initial_checkpoint_models,
)
from experiments.campaigns.constants import DEFAULT_VARIANTS, ROOT
from experiments.campaigns.resources import (
    available_resources as available_resources,
)
from experiments.campaigns.resources import (
    cgroup_memory_file as cgroup_memory_file,
)
from experiments.campaigns.resources import (
    cpu_mask_for_slot as cpu_mask_for_slot,
)
from experiments.campaigns.resources import (
    expand_slurm_hosts as expand_slurm_hosts,
)
from experiments.campaigns.resources import (
    resolve_worker_nodes as resolve_worker_nodes,
)
from experiments.campaigns.resources import (
    sample_resources as sample_resources,
)
from experiments.campaigns.resources import (
    worker_command as worker_command,
)
from experiments.campaigns.runtime_metadata import (
    runtime_metadata as runtime_metadata,
)
from experiments.campaigns.source_snapshot import (
    action_space_matches_width as action_space_matches_width,
)
from experiments.campaigns.source_snapshot import (
    cached_sha256_file as cached_sha256_file,
)
from experiments.campaigns.source_snapshot import (
    campaign_data_inputs as campaign_data_inputs,
)
from experiments.campaigns.source_snapshot import (
    campaign_source_files as campaign_source_files,
)
from experiments.campaigns.source_snapshot import (
    sha256_file as sha256_file,
)
from experiments.campaigns.source_snapshot import (
    verify_campaign_source_snapshot as verify_campaign_source_snapshot,
)
from experiments.campaigns.source_snapshot import (
    write_campaign_source_snapshot as write_campaign_source_snapshot,
)
from experiments.campaigns.state_store import (
    atomic_json as atomic_json,
)
from experiments.campaigns.state_store import (
    read_manifest as read_manifest,
)
from experiments.campaigns.state_store import (
    record as record,
)
from experiments.campaigns.state_store import (
    utc as utc,
)
from experiments.campaigns.state_store import (
    validate_inputs as validate_inputs,
)

TRAIN_ONE_VARIANT_SCRIPT = (
    ROOT / "scripts" / "research" / "frozen-launches" / "train_one_variant.sh"
)
RESOURCE_SAMPLER_SCRIPT = ROOT / "compute" / "slurm" / "resources.py"

__all__ = [
    "DEFAULT_VARIANTS",
    "ROOT",
    "action_space_matches_width",
    "archive_initial_checkpoints",
    "atomic_json",
    "available_resources",
    "build_initial_checkpoint_map",
    "cached_sha256_file",
    "campaign_data_inputs",
    "campaign_source_files",
    "choose_resume_checkpoint",
    "cpu_mask_for_slot",
    "expand_slurm_hosts",
    "hash_run_artifacts",
    "latest_resumable_checkpoint",
    "main",
    "read_manifest",
    "record",
    "resolve_worker_nodes",
    "runtime_metadata",
    "sample_resources",
    "sha256_file",
    "utc",
    "validate_campaign_evaluation",
    "validate_initial_checkpoints",
    "validate_inputs",
    "verify_campaign_source_snapshot",
    "verify_initial_checkpoint_models",
    "verify_run_artifacts",
    "worker_command",
    "write_campaign_source_snapshot",
]

STOP = False
_CACHE_FILE_HASHES: dict[tuple[str, int, int, int], str] = {}


def signal_stop(_sig: int, _frame: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=pathlib.Path)
    parser.add_argument("--variant-file", type=pathlib.Path)
    parser.add_argument("--seeds")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--worker-nodes",
        help="comma-separated nodes inside the current allocation; workers are load-balanced across them",
    )
    parser.add_argument("--worker-cpus", type=int, default=8)
    parser.add_argument("--worker-gpu-type", default="gh200")
    parser.add_argument("--worker-gpus-per-node", type=int, default=1)
    parser.add_argument("--data-source", choices=("frames", "bundle"))
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--initial-checkpoints",
        type=pathlib.Path,
        help="lineage map for checkpoints migrated from a prior campaign",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-fingerprint-override-reason",
        help=(
            "explicitly continue an existing campaign when its current identity fingerprint "
            "differs; the original fingerprint remains the artifact/checkpoint lineage"
        ),
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--fast-reports",
        action="store_true",
        help="Skip per-trade PNGs during training; keep summary plots, metrics, JSON, CSV, and checkpoints",
    )
    parser.add_argument("--test-worker", type=pathlib.Path, help="inject mock worker for CI only")
    args = parser.parse_args()
    import yaml

    evaluation: dict[str, Any] = {"locked_oos_role": "unspecified", "deterministic": True}
    if args.campaign_spec:
        if any(
            value is not None
            for value in (
                args.variant_file,
                args.seeds,
                args.timesteps,
                args.data_source,
                args.campaign_id,
            )
        ):
            raise ValueError("--campaign-spec cannot be combined with campaign identity overrides")
        args.campaign_spec = args.campaign_spec.resolve()
        try:
            args.campaign_spec.relative_to(ROOT)
        except ValueError:
            raise ValueError("campaign spec must be stored inside this repository") from None
        spec = yaml.safe_load(args.campaign_spec.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("campaign spec must be a YAML mapping")
        spec_variants = spec.get("variants")
        if not isinstance(spec_variants, list) or not spec_variants:
            raise ValueError("campaign spec must contain a nonempty variants list")
        spec_variant_file = (
            ROOT / str(spec.get("variant_file", "config/campaigns/variant_list.txt"))
        ).resolve()
        listed = [
            line.strip()
            for line in spec_variant_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if listed != [str(name) for name in spec_variants]:
            raise ValueError("campaign spec variants do not match its authoritative variant file")
        evaluation = validate_campaign_evaluation(spec.get("evaluation", {}))
        default_config = yaml.safe_load(
            (ROOT / "config" / "base" / "default.yaml").read_text(encoding="utf-8")
        )
        if spec.get("split") != default_config["data"]["split"]:
            raise ValueError("campaign split does not match the effective base configuration")
        args.variant_file = spec_variant_file
        args.seeds = " ".join(str(seed) for seed in spec.get("seeds", []))
        args.timesteps = int(spec["timesteps"])
        args.data_source = str(spec.get("data_source", "frames"))
        args.campaign_id = str(spec["campaign_id"])
    else:
        args.variant_file = args.variant_file or DEFAULT_VARIANTS
        args.seeds = args.seeds or "42 43 44 45 46"
        args.timesteps = args.timesteps or 20_000_000
        args.data_source = args.data_source or "frames"
        args.campaign_id = args.campaign_id or "core-18x5"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.campaign_id):
        raise ValueError("unsafe campaign id")
    if (
        args.worker_cpus <= 0
        or args.worker_gpus_per_node <= 0
        or not re.fullmatch(r"[A-Za-z0-9]+", args.worker_gpu_type)
    ):
        raise ValueError("worker CPU/GPU count or GPU type is invalid")
    worker_nodes = resolve_worker_nodes(
        args.worker_nodes, dry_run=args.dry_run or bool(args.test_worker)
    )
    if worker_nodes and len(worker_nodes) > args.workers:
        raise ValueError("worker count must be at least the number of worker nodes")
    jobs = validate_inputs(args)
    initial_checkpoints: dict[str, dict[str, Any]] = {}
    if args.initial_checkpoints is not None:
        args.initial_checkpoints = args.initial_checkpoints.resolve()
        try:
            args.initial_checkpoints.relative_to(ROOT.resolve())
        except ValueError:
            raise ValueError(
                "initial checkpoint map must be stored inside this repository"
            ) from None
        initial_checkpoints = validate_initial_checkpoints(
            args.initial_checkpoints, jobs, args.timesteps
        )
    # Source-of-truth variant lookup without changing training semantics.
    config = yaml.safe_load((ROOT / "config" / "experiments.yaml").read_text())
    known = {str(item["name"]) for item in config["variants"]}
    unknown = {str(j["variant"]) for j in jobs} - known
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    if args.benchmark and (len(jobs) != 1 or args.timesteps > 100_000):
        raise ValueError("benchmark requires one run and <=100000 training steps")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "campaign_id": args.campaign_id,
                    "output_root": str(ROOT / "outputs" / "gpu_campaigns" / args.campaign_id),
                    "variant_file": str(args.variant_file),
                    "seeds": [int(seed) for seed in args.seeds.replace(",", " ").split()],
                    "steps_per_run": args.timesteps,
                    "runs": len(jobs),
                    "total_timesteps": len(jobs) * args.timesteps,
                    "workers": args.workers,
                    "n_envs": args.n_envs,
                    "worker_nodes": worker_nodes,
                    "worker_cpus": args.worker_cpus if worker_nodes else None,
                    "worker_gpu_type": args.worker_gpu_type if worker_nodes else None,
                    "worker_gpus_per_node": (args.worker_gpus_per_node if worker_nodes else None),
                    "data_source": args.data_source,
                    "fast_reports": args.fast_reports,
                    "initial_checkpoints": len(initial_checkpoints),
                    "keys": [j["key"] for j in jobs],
                },
                indent=2,
            )
        )
        return 0
    if not os.getenv("SLURM_JOB_ID") and not args.test_worker:
        raise RuntimeError("must run inside an existing Slurm compute allocation")
    resources = available_resources()
    cpus = resources["cpu_affinity"]
    cpu_ids = cast(list[int], resources["cpu_ids"])
    workers_per_node = math.ceil(args.workers / len(worker_nodes)) if worker_nodes else args.workers
    envs_per_cpu = 12 if args.data_source == "bundle" else 2
    if workers_per_node * args.n_envs > cpus * envs_per_cpu:
        raise RuntimeError(f"requested vector environments exceed {envs_per_cpu} per available CPU")
    if worker_nodes and workers_per_node * args.worker_cpus > cpus:
        raise RuntimeError(
            f"worker CPU request needs {workers_per_node * args.worker_cpus} CPUs per node; "
            f"controller sees {cpus}"
        )
    available_mem = resources["memory_limit_bytes"]
    # Frames estimates are conservative historical PSS values. The bundle
    # n_envs=64 estimate is calibrated from the live three-worker GH200 pilot:
    # each worker peaked at about 23 GB, so reserve 30 GiB per worker before
    # applying the existing 35% margin and 64 GiB controller/OS allowance.
    estimated_per_worker = {4: 12, 8: 19, 16: 35, 32: 67, 64: 132}.get(args.n_envs)
    if args.data_source == "bundle" and args.n_envs == 64:
        estimated_per_worker = 30
    if estimated_per_worker is None and not args.test_worker:
        raise ValueError("n-envs not benchmarked for memory: choose 4,8,16,32,64")
    if available_mem and estimated_per_worker and not args.test_worker:
        recommended_bytes = int((workers_per_node * estimated_per_worker * 1.35 + 64) * 1024**3)
        if recommended_bytes > available_mem:
            raise RuntimeError(
                f"insufficient cgroup RAM: needs >= {recommended_bytes // 1024**3} GiB"
            )
    out = ROOT / "outputs" / "gpu_campaigns" / args.campaign_id
    (out / "logs" / "workers").mkdir(parents=True, exist_ok=True)
    (out / "logs" / "failures").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "markers").mkdir(exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)
    lock = (out / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another controller owns this campaign") from None
    archived_bootstrap = archive_initial_checkpoints(out, initial_checkpoints)
    manifest_path = out / "campaign.json"
    started_time = time.monotonic()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = runtime_metadata()
    source_files = campaign_source_files(ROOT, args.campaign_spec)
    if args.initial_checkpoints is not None:
        source_files.append(args.initial_checkpoints)
    source_files = sorted(set(source_files))
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    raw_config = yaml.safe_load((ROOT / "config" / "base" / "default.yaml").read_text())
    data_config = raw_config["data"]
    data_inputs = campaign_data_inputs(data_config)
    missing_inputs = [path for path, digest in data_inputs.items() if digest == "MISSING"]
    if missing_inputs and not args.test_worker:
        raise RuntimeError(f"configured market-data inputs missing: {missing_inputs}")
    identity = {
        "jobs": [x["key"] for x in jobs],
        "steps": args.timesteps,
        "data_source": args.data_source,
        "workers": args.workers,
        "n_envs": args.n_envs,
        "worker_placement": {
            "mode": "slurm-node-steps" if worker_nodes else "local-subprocesses",
            "node_count": len(worker_nodes) if worker_nodes else 1,
            "gpu_type": args.worker_gpu_type if worker_nodes else "allocation-default",
            "gpus_per_node": args.worker_gpus_per_node if worker_nodes else None,
            "gpu_binding": "least-loaded-map_gpu" if worker_nodes else None,
            "cpus_per_worker": args.worker_cpus if worker_nodes else None,
            "cpu_binding": "unique-mask-per-worker" if worker_nodes else None,
        },
        "fast_reports": args.fast_reports,
        "evaluation": evaluation,
        "runtime_compatibility": {
            "python": runtime["python"],
            "machine": runtime["machine"],
            "packages": runtime["packages"],
        },
        "git_sha": git_sha,
        "source_hashes": source_hashes,
        "data_inputs": data_inputs,
        "initial_checkpoint_lineage": {
            key: {name: value for name, value in checkpoint.items() if name != "checkpoint_path"}
            for key, checkpoint in initial_checkpoints.items()
        },
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if initial_checkpoints:
        verify_initial_checkpoint_models(archived_bootstrap, initial_checkpoints)
    effective_identity = identity
    fingerprint_override: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        if not args.resume:
            raise RuntimeError("campaign exists: use --resume")
        if manifest["fingerprint"] != fingerprint:
            reason = (args.resume_fingerprint_override_reason or "").strip()
            if not reason:
                raise RuntimeError("resume fingerprint mismatch")
            if len(reason) < 12:
                raise ValueError(
                    "resume fingerprint override reason must be at least 12 characters"
                )
            canonical_fingerprint = str(manifest.get("fingerprint", ""))
            canonical_identity: dict[str, Any] | None = None
            for item in manifest.get("jobs", []):
                attempts = int(item.get("attempts", 0))
                for attempt in range(attempts, 0, -1):
                    provenance_path = (
                        out
                        / "runs"
                        / "attempts"
                        / str(item.get("key", ""))
                        / f"attempt{attempt}"
                        / "attempt_provenance.json"
                    )
                    try:
                        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    candidate = provenance.get("identity") if isinstance(provenance, dict) else None
                    if (
                        isinstance(provenance, dict)
                        and provenance.get("campaign_fingerprint") == canonical_fingerprint
                        and isinstance(candidate, dict)
                        and hashlib.sha256(
                            json.dumps(candidate, sort_keys=True).encode()
                        ).hexdigest()
                        == canonical_fingerprint
                    ):
                        canonical_identity = candidate
                        break
                if canonical_identity is not None:
                    break
            if canonical_identity is None:
                raise RuntimeError(
                    "cannot override resume fingerprint: no attempt contains the original "
                    "fingerprint identity"
                )
            if manifest.get("initial_checkpoint_lineage", {}) != canonical_identity.get(
                "initial_checkpoint_lineage", {}
            ):
                raise RuntimeError(
                    "cannot override resume fingerprint: original checkpoint lineage does "
                    "not match the campaign manifest"
                )
            required_matches = {
                "jobs": [job["key"] for job in jobs],
                "steps": args.timesteps,
                "data_source": args.data_source,
                "n_envs": args.n_envs,
                "fast_reports": args.fast_reports,
                "evaluation": evaluation,
            }
            changed_required = [
                name
                for name, value in required_matches.items()
                if canonical_identity.get(name) != value
            ]
            if changed_required:
                raise RuntimeError(
                    "resume fingerprint override rejected; continuation-critical identity "
                    f"changed: {', '.join(changed_required)}"
                )
            effective_identity = identity
            identity = canonical_identity
            fingerprint_override = {
                "reason": reason,
                "canonical_fingerprint": canonical_fingerprint,
                "effective_fingerprint": fingerprint,
                "effective_identity": effective_identity,
                "continuation_critical_fields_verified": sorted(required_matches),
                "canonical_checkpoint_lineage_sha256": hashlib.sha256(
                    json.dumps(
                        canonical_identity.get("initial_checkpoint_lineage", {}), sort_keys=True
                    ).encode()
                ).hexdigest(),
                "effective_checkpoint_lineage_sha256": hashlib.sha256(
                    json.dumps(
                        effective_identity.get("initial_checkpoint_lineage", {}), sort_keys=True
                    ).encode()
                ).hexdigest(),
                "training_evidence_status": "exploratory_after_fingerprint_override",
                "recorded_at": utc(),
            }
            fingerprint = canonical_fingerprint
        if manifest.get("initial_checkpoint_lineage", {}) != identity["initial_checkpoint_lineage"]:
            raise RuntimeError("resume initial checkpoint lineage mismatch")
        snapshot_record = manifest.get("source_snapshot")
        if snapshot_record is not None:
            if not isinstance(snapshot_record, dict) or not isinstance(
                snapshot_record.get("path"), str
            ):
                raise RuntimeError("campaign source snapshot record is invalid")
            snapshot_path = (out / snapshot_record["path"]).resolve()
            if snapshot_path.parent != out.resolve():
                raise RuntimeError("campaign source snapshot path escapes campaign directory")
            snapshot_valid, snapshot_reason = verify_campaign_source_snapshot(
                snapshot_path, snapshot_record
            )
            if not snapshot_valid:
                raise RuntimeError(f"campaign source snapshot invalid: {snapshot_reason}")
        if manifest.get("data_source", "frames") != args.data_source:
            raise RuntimeError("resume data-source mismatch")
        if manifest.get("n_envs") != args.n_envs:
            raise RuntimeError("resume n-envs mismatch")
        if fingerprint_override is not None:
            manifest.setdefault("resume_fingerprint_overrides", []).append(
                {
                    key: value
                    for key, value in fingerprint_override.items()
                    if key != "effective_identity"
                }
            )
            override_path = (
                out
                / "reports"
                / f"resume_fingerprint_override_{fingerprint_override['effective_fingerprint']}.json"
            )
            atomic_json(override_path, fingerprint_override)
            record(
                out / "reports" / "resume_fingerprint_overrides.jsonl",
                {
                    key: value
                    for key, value in fingerprint_override.items()
                    if key != "effective_identity"
                },
            )
        manifest.pop("stopping", None)
        jobs = manifest["jobs"]
        for item in jobs:
            if (
                item["status"] == "succeeded"
                and not (out / "markers" / f"{item['key']}.done").is_file()
            ):
                raise RuntimeError(f"completed marker missing for {item['key']}")
            if item["status"] == "succeeded":
                attempt_out = out / "runs" / "attempts" / item["key"] / f"attempt{item['attempts']}"
                verified, reason = verify_run_artifacts(
                    attempt_out,
                    item["key"],
                    args.timesteps,
                    expected_fingerprint=fingerprint,
                    expected_attempt=int(item["attempts"]),
                )
                if not verified:
                    raise RuntimeError(f"completed artifacts invalid for {item['key']}: {reason}")
                if not args.test_worker and item.get("artifact_hashes") != hash_run_artifacts(
                    attempt_out
                ):
                    raise RuntimeError(f"completed artifacts changed for {item['key']}")
            if item["status"] == "running":
                item.update(
                    status="interrupted",
                    failure_reason="controller ended while the worker was active",
                    interrupted_at=utc(),
                )
            if item["status"] == "interrupted":
                item["resume_from_attempt"] = int(item["attempts"])
                item["status"] = "pending"
    else:
        source_snapshot = write_campaign_source_snapshot(
            ROOT,
            source_files,
            out / "source_snapshot.zip",
            git_sha=git_sha,
            expected_source_hashes=source_hashes,
        )
        manifest = {
            "created": utc(),
            "hostname": socket.gethostname(),
            "job_id": os.getenv("SLURM_JOB_ID", "test"),
            "git_sha": git_sha,
            "source_hashes": source_hashes,
            "source_snapshot": source_snapshot,
            "data_inputs": data_inputs,
            "runtime": runtime,
            "fingerprint": fingerprint,
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "data_source": args.data_source,
            "fast_reports": args.fast_reports,
            "workers": args.workers,
            "worker_nodes": worker_nodes,
            "worker_cpus": args.worker_cpus if worker_nodes else None,
            "worker_gpu_type": args.worker_gpu_type if worker_nodes else None,
            "worker_gpus_per_node": args.worker_gpus_per_node if worker_nodes else None,
            "gpu_binding": "least-loaded-map_gpu" if worker_nodes else None,
            "evaluation": evaluation,
            "initial_checkpoint_lineage": identity["initial_checkpoint_lineage"],
            "jobs": jobs,
            "resources": available_resources(),
        }
        atomic_json(manifest_path, manifest)
    events = out / "queue.jsonl"
    scheduler = out / "logs" / "scheduler.jsonl"
    running: dict[str, tuple[subprocess.Popen[bytes], TextIO, dict[str, Any]]] = {}
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    print(
        f"campaign={args.campaign_id} runs={len(jobs)} workers={args.workers} "
        f"host={socket.gethostname()}",
        flush=True,
    )

    def snapshot() -> None:
        atomic_json(manifest_path, manifest)
        atomic_json(
            out / "reports" / "campaign_status.json",
            {
                "timestamp": utc(),
                "statuses": {
                    state: sum(j["status"] == state for j in jobs)
                    for state in ("pending", "running", "succeeded", "failed", "interrupted")
                },
            },
        )

    resource_samplers: list[tuple[subprocess.Popen[bytes], TextIO]] = []
    if worker_nodes:
        for node in worker_nodes:
            sampler_log = (out / "logs" / f"resource_sampler_{node}.log").open(
                "a", encoding="utf-8"
            )
            sampler_cpu_mask, sampler_cpu_ids = cpu_mask_for_slot(
                cpu_ids, workers_per_node * args.worker_cpus, 1
            )
            sampler_cmd = worker_command(
                [
                    sys.executable,
                    str(RESOURCE_SAMPLER_SCRIPT),
                    "--campaign",
                    str(manifest_path),
                    "--out",
                    str(out / "logs" / f"resources_{node}.csv"),
                ],
                node=node,
                gpu_type=args.worker_gpu_type,
                cpus=1,
                gpu_count=0,
                cpu_mask=sampler_cpu_mask,
                expected_cpu_ids=sampler_cpu_ids,
                step_cpus=cpus,
            )
            sampler_env = dict(os.environ)
            sampler_env.pop("CUDA_VISIBLE_DEVICES", None)
            sampler_proc = subprocess.Popen(
                sampler_cmd,
                cwd=ROOT,
                env=sampler_env,
                stdout=sampler_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            resource_samplers.append((sampler_proc, sampler_log))

    def start(job: dict[str, Any]) -> None:
        key = job["key"]
        job["attempts"] += 1
        attempt = job["attempts"]
        for field in ("artifact_error", "artifact_hashes", "failure_reason"):
            job.pop(field, None)
        logpath = out / "logs" / "workers" / f"{key}__attempt{attempt}.log"
        run_out = out / "runs" / "attempts" / key / f"attempt{attempt}"
        run_out.mkdir(parents=True, exist_ok=True)
        resume_from_attempt = job.pop("resume_from_attempt", None)
        resume_checkpoint, resume_origin = choose_resume_checkpoint(
            out,
            key,
            int(resume_from_attempt) if resume_from_attempt else None,
            fingerprint,
            archived_bootstrap,
        )
        bootstrap_lineage = initial_checkpoints.get(key)
        resume_checkpoint_hash = sha256_file(resume_checkpoint) if resume_checkpoint else None
        worker_node = None
        worker_cpu_mask = None
        worker_cpu_ids: list[int] = []
        if worker_nodes:
            node_load = {
                node: sum(
                    1
                    for running_job in running.values()
                    if running_job[2].get("worker_node") == node
                )
                for node in worker_nodes
            }
            worker_node = min(worker_nodes, key=lambda node: node_load[node])
            worker_cpu_mask, worker_cpu_ids = cpu_mask_for_slot(
                cpu_ids, node_load[worker_node], args.worker_cpus
            )
            job["worker_node"] = worker_node
            job["worker_cpu_mask"] = worker_cpu_mask
            job["worker_cpu_ids"] = worker_cpu_ids
            gpu_load = {
                gpu_id: sum(
                    1
                    for running_job in running.values()
                    if running_job[2].get("worker_node") == worker_node
                    and running_job[2].get("worker_gpu_id") == gpu_id
                )
                for gpu_id in range(args.worker_gpus_per_node)
            }
            worker_gpu_id = min(gpu_load, key=lambda gpu_id: (gpu_load[gpu_id], gpu_id))
            job["worker_gpu_id"] = worker_gpu_id
            job["remote_pid_file"] = str(run_out / "worker_process.json")
        atomic_json(
            run_out / "attempt_provenance.json",
            {
                "campaign_id": args.campaign_id,
                "campaign_fingerprint": fingerprint,
                "run_key": key,
                "variant": job["variant"],
                "seed": job["seed"],
                "attempt": attempt,
                "requested_timesteps": args.timesteps,
                "n_envs": args.n_envs,
                "data_source": args.data_source,
                "fast_reports": args.fast_reports,
                "evaluation": evaluation,
                "oos_selection_allowed": bool(evaluation.get("selection_allowed", False)),
                "identity": identity,
                "started": utc(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "worker_node": worker_node,
                "worker_gpu_id": job.get("worker_gpu_id"),
                "resume_from_attempt": resume_from_attempt,
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                "resume_checkpoint_sha256": resume_checkpoint_hash,
                "resume_origin": resume_origin,
                "initial_checkpoint_lineage": bootstrap_lineage,
                "resume_without_checkpoint_reason": (
                    "no intact checkpoint in the previous interrupted attempt"
                    if resume_from_attempt and resume_origin is None and resume_checkpoint is None
                    else None
                ),
            },
        )
        cmd = (
            [str(args.test_worker), key]
            if args.test_worker
            else ["bash", str(TRAIN_ONE_VARIANT_SCRIPT)]
        )
        if worker_nodes:
            cmd = worker_command(
                cmd,
                node=worker_node,
                gpu_type=args.worker_gpu_type,
                cpus=args.worker_cpus,
                gpu_count=args.worker_gpus_per_node,
                gpu_id=int(job["worker_gpu_id"]),
                cpu_mask=worker_cpu_mask,
                expected_cpu_ids=worker_cpu_ids,
                step_cpus=cpus,
            )
        env = dict(
            os.environ,
            VARIANT=job["variant"],
            SEED=str(job["seed"]),
            STEPS=str(args.timesteps),
            OUT_DIR=str(run_out),
            LOG_DIR=str(out / "logs" / "workers"),
            QUANT_RL_MAX_N_ENVS=str(args.n_envs),
            QUANT_RL_DATA_SOURCE=args.data_source,
            QUANT_RL_FAST_REPORTS="1" if args.fast_reports else "0",
            QUANT_RL_ALLOW_OOS_SELECTION=(
                "1" if evaluation.get("selection_allowed", False) else "0"
            ),
            GPU_ID="0",
            PYTHONUNBUFFERED="1",
        )
        if worker_node is not None:
            # Slurm assigns this step's GPU; it is logical device zero in the worker.
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["CAMPAIGN_WORKER_PID_FILE"] = str(run_out / "worker_process.json")
        env.pop("RESUME_CHECKPOINT", None)
        if resume_checkpoint is not None:
            env["RESUME_CHECKPOINT"] = str(resume_checkpoint)
        handle = logpath.open("a", encoding="utf-8")
        handle.write(
            f"{utc()} START key={key} attempt={attempt} "
            f"host={worker_node or socket.gethostname()} controller={socket.gethostname()} "
            f"slurm={os.getenv('SLURM_JOB_ID')} git={git_sha} "
            f"resume={resume_checkpoint or 'none'} cmd={cmd}\n"
        )
        handle.flush()
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        job.update(status="running", pid=proc.pid, started=utc(), log=str(logpath))
        running[key] = (proc, handle, job)
        record(
            events,
            {
                "event": "start",
                "key": key,
                "attempt": attempt,
                "pid": proc.pid,
                "worker_node": worker_node,
                "worker_cpu_mask": worker_cpu_mask,
                "worker_cpu_ids": worker_cpu_ids,
                "log": str(logpath),
                "attempt_output": str(run_out),
            },
        )
        record(scheduler, {"action": "start", "key": key, "attempt": attempt})
        snapshot()

    last_sample = 0.0
    stop_sent = False
    stop_deadline: float | None = None
    while running or (not STOP and any(j["status"] == "pending" for j in jobs)):
        if STOP and not stop_sent:
            manifest["stopping"] = True
            snapshot()
            stop_deadline = time.monotonic() + 60
            for proc, _, _ in running.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            stop_sent = True
        if STOP and stop_deadline is not None and time.monotonic() >= stop_deadline:
            for proc, _, _ in running.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            stop_deadline = None
        while not STOP and len(running) < args.workers:
            next_job = next((j for j in jobs if j["status"] == "pending"), None)
            if next_job is None:
                break
            start(next_job)
        for key, (proc, handle, job) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            handle.write(f"{utc()} END key={key} exit={ret}\n")
            handle.close()
            attempt_out = out / "runs" / "attempts" / key / f"attempt{job['attempts']}"
            verified, reason = (
                (True, "")
                if args.test_worker
                else verify_run_artifacts(
                    attempt_out,
                    key,
                    args.timesteps,
                    expected_fingerprint=fingerprint,
                    expected_attempt=int(job["attempts"]),
                )
            )
            success = ret == 0 and verified
            gate_failure = not verified and "training risk-survival gate failed" in reason
            interrupted = STOP and not success and not gate_failure
            artifact_hashes = hash_run_artifacts(attempt_out)
            failure_reason = None
            if not success:
                reasons = []
                if ret != 0:
                    reasons.append(f"worker exited with code {ret}")
                if not verified and reason:
                    reasons.append(reason)
                failure_reason = "; ".join(reasons) or "campaign acceptance failed"
            job.update(
                status=("succeeded" if success else "interrupted" if interrupted else "failed"),
                exit_code=ret,
                ended=utc(),
                artifact_error=reason if ret == 0 else "",
                artifact_hashes=artifact_hashes,
            )
            if success:
                (out / "markers" / f"{key}.done").touch()
            else:
                job["failure_reason"] = failure_reason
                record(
                    out / "logs" / "failures" / "summary.jsonl",
                    {
                        "key": key,
                        "attempt": job["attempts"],
                        "exit_code": ret,
                        "reason": failure_reason,
                        "artifact_error": reason,
                        "artifact_hashes": artifact_hashes,
                        "log": job["log"],
                    },
                )
            record(
                events,
                {
                    "event": job["status"],
                    "key": key,
                    "attempt": job["attempts"],
                    "exit_code": ret,
                    "reason": "allocation boundary" if interrupted else failure_reason,
                },
            )
            del running[key]
            snapshot()
        if not worker_nodes and time.monotonic() - last_sample >= 30:
            sample_resources(out / "logs" / "resources.csv")
            last_sample = time.monotonic()
        if running:
            time.sleep(1)
    snapshot()
    for sampler_proc, sampler_log in resource_samplers:
        try:
            sampler_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(sampler_proc.pid, signal.SIGTERM)
            try:
                sampler_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(sampler_proc.pid, signal.SIGKILL)
                sampler_proc.wait()
        sampler_log.close()
    if args.benchmark:
        elapsed = max(time.monotonic() - started_time, 0.001)
        with (out / "reports" / "benchmarks.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    utc(),
                    args.workers,
                    args.n_envs,
                    args.timesteps,
                    elapsed,
                    round(args.timesteps / elapsed, 2),
                    os.getenv("SLURM_JOB_ID", ""),
                ]
            )
    try:
        summary_script = pathlib.Path(__file__).resolve().with_name("reporting.py")
        summary_result = subprocess.run(
            [sys.executable, str(summary_script), "--campaign-dir", str(out)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if summary_result.returncode == 0:
            print(summary_result.stdout.rstrip(), flush=True)
        else:
            print(
                "WARNING: campaign results summary generation failed: "
                f"{summary_result.stderr.strip() or summary_result.stdout.strip()}",
                flush=True,
            )
    except OSError as exc:
        print(f"WARNING: campaign results summary could not start: {exc}", flush=True)
    return 1 if STOP or any(j["status"] != "succeeded" for j in jobs) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
