#!/usr/bin/env python3
"""Durable, allocation-bounded launcher for the fixed Triton variant matrix.

Does not create or cancel Slurm allocations. Runs in an existing compute-node job.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import pathlib
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile
from typing import Any, TextIO, cast

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ROOT / "scripts" / "variant_list.txt"
STOP = False
_CACHE_FILE_HASHES: dict[tuple[str, int, int, int], str] = {}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()  # noqa: UP017


def atomic_json(path: pathlib.Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def record(path: pathlib.Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"timestamp": utc(), **value}, sort_keys=True) + "\n")
        out.flush()


def read_manifest(path: pathlib.Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def validate_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = [
        v.strip()
        for v in args.variant_file.read_text().splitlines()
        if v.strip() and not v.lstrip().startswith("#")
    ]
    for variant in variants:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", variant):
            raise ValueError(f"unsafe variant name: {variant}")
    if len(variants) != len(set(variants)):
        raise ValueError("duplicate variants in variant file")
    if not variants:
        raise ValueError("variant list empty")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds empty or duplicate")
    if args.timesteps <= 0 or args.workers <= 0 or args.n_envs <= 0:
        raise ValueError("timesteps/workers/n-envs must be positive")
    jobs = [
        {"key": f"{v}__seed{s}", "variant": v, "seed": s, "status": "pending", "attempts": 0}
        for v in variants
        for s in seeds
    ]
    if len(jobs) != len({j["key"] for j in jobs}):
        raise ValueError("duplicate run keys")
    return jobs


def signal_stop(_sig: int, _frame: object) -> None:
    global STOP
    STOP = True


def cgroup_memory_file(name: str) -> pathlib.Path | None:
    """Locate a memory-controller file under this process's cgroup."""
    root = pathlib.Path("/sys/fs/cgroup")
    try:
        entries = pathlib.Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        parts = entry.split(":", maxsplit=2)
        if len(parts) != 3:
            continue
        _hierarchy, controllers, relative = parts
        if controllers == "":
            base = root / relative.lstrip("/")
        elif "memory" in controllers.split(","):
            base = root / "memory" / relative.lstrip("/")
        else:
            continue
        candidate = base / name
        try:
            candidate.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if candidate.is_file():
            return candidate
    for fallback in (root / name, root / "memory" / name):
        try:
            fallback.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if fallback.is_file():
            return fallback
    return None


def available_resources() -> dict[str, Any]:
    mem_limit = 0
    memory = cgroup_memory_file("memory.max")
    if memory is not None:
        # Slurm task cgroups often report "max" while a parent job cgroup
        # carries the actual allocation limit. Use the tightest finite limit
        # visible from this task instead of treating that allocation as unbound.
        for parent in (memory, *memory.parents):
            if not parent.is_relative_to(pathlib.Path("/sys/fs/cgroup")):
                break
            candidate = parent / "memory.max"
            try:
                value = candidate.read_text().strip()
            except OSError:
                continue
            if value.isdigit() and 0 < int(value) < 2**60:
                mem_limit = int(value) if not mem_limit else min(mem_limit, int(value))
    if not mem_limit:
        memory = cgroup_memory_file("memory.limit_in_bytes")
        if memory is not None and memory.read_text().strip().isdigit():
            legacy_value = int(memory.read_text().strip())
            if 0 < legacy_value < 2**60:
                mem_limit = legacy_value
    cpu_ids = sorted(os.sched_getaffinity(0))
    return {
        "memory_limit_bytes": mem_limit,
        "cpu_affinity": len(cpu_ids),
        "cpu_ids": cpu_ids,
    }


def cpu_mask_for_slot(cpu_ids: list[int], slot: int, cpus: int) -> tuple[str, list[int]]:
    """Assign a distinct contiguous slice of allocated CPU IDs to one worker."""
    if slot < 0 or cpus <= 0:
        raise ValueError("CPU slot and worker CPU count must be non-negative/positive")
    start = slot * cpus
    selected = cpu_ids[start : start + cpus]
    if len(selected) != cpus:
        raise ValueError(f"CPU slot {slot} needs {cpus} CPUs but only {len(selected)} remain")
    mask = sum(1 << cpu for cpu in selected)
    return f"0x{mask:x}", selected


def expand_slurm_hosts(node_list: str) -> list[str]:
    """Expand a Slurm host expression, preserving the scheduler's host order."""
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise ValueError(f"could not expand Slurm node list {node_list!r}: {result.stderr.strip()}")
    hosts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("Slurm node list expanded to no hosts or duplicate hosts")
    return hosts


def worker_command(
    command: list[str],
    *,
    node: str | None,
    gpu_type: str,
    cpus: int,
    gpu_count: int = 1,
    gpu_id: int = 0,
    cpu_mask: str | None = None,
    expected_cpu_ids: list[int] | None = None,
) -> list[str]:
    """Build an allocation-local or explicitly GPU-pinned Slurm worker command."""
    if node is None:
        return command
    if gpu_count < 0 or (gpu_count and not 0 <= gpu_id < gpu_count):
        raise ValueError("worker GPU count or binding index is invalid")
    if (cpu_mask is None) != (expected_cpu_ids is None):
        raise ValueError("CPU mask and expected CPU IDs must be provided together")
    result = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        f"--cpus-per-task={cpus}",
    ]
    if gpu_count:
        result.extend([f"--gpus={gpu_type}:{gpu_count}", f"--gpu-bind=map_gpu:{gpu_id}"])
    cpu_binding = f"mask_cpu:{cpu_mask}" if cpu_mask is not None else "cores"
    worker_command = command
    if cpu_mask is not None and expected_cpu_ids is not None:
        guard = ROOT / "scripts" / "matrix" / "verify_worker_cpu_affinity.py"
        worker_command = [
            sys.executable,
            str(guard),
            "--expected-cpus",
            ",".join(str(cpu) for cpu in expected_cpu_ids),
            "--",
            *command,
        ]
    result.extend(
        [
            f"--cpu-bind={cpu_binding}",
            "--chdir",
            str(ROOT),
            "--export=ALL",
            *worker_command,
        ]
    )
    return result


def resolve_worker_nodes(raw: str | None, *, dry_run: bool) -> list[str]:
    if not raw:
        return []
    if dry_run:
        requested = [host.strip() for host in raw.split(",") if host.strip()]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("worker node list is empty or contains duplicates")
        return requested
    job_node_list = os.getenv("SLURM_JOB_NODELIST") or os.getenv("SLURM_NODELIST")
    if not job_node_list:
        raise ValueError("--worker-nodes requires an existing multi-node Slurm allocation")
    allocated = set(expand_slurm_hosts(job_node_list))
    requested = expand_slurm_hosts(raw)
    outside = sorted(set(requested) - allocated)
    if outside:
        raise ValueError(f"worker nodes are outside the Slurm allocation: {outside}")
    return requested


def sample_resources(path: pathlib.Path) -> None:
    def read(name: str) -> str:
        p = cgroup_memory_file(name)
        return p.read_text().strip().replace("\n", ";") if p is not None else ""

    gpu_util = gpu_memory = ""
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if gpu.returncode == 0:
            gpu_util, _, gpu_memory = gpu.stdout.strip().partition(", ")
    except (OSError, subprocess.TimeoutExpired):
        pass
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                utc(),
                os.getloadavg()[0],
                read("memory.current"),
                read("memory.max"),
                read("memory.peak"),
                read("memory.events"),
                gpu_util,
                gpu_memory,
            ]
        )
        f.flush()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256_file(path: pathlib.Path) -> str:
    """Hash a shared campaign cache once per unchanged file identity."""
    stat = path.stat()
    identity = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    digest = _CACHE_FILE_HASHES.get(identity)
    if digest is None:
        digest = sha256_file(path)
        _CACHE_FILE_HASHES[identity] = digest
    return digest


def action_space_matches_width(action_space: Any, width: int) -> bool:
    """Check vector and legacy discrete action spaces without conflating them."""
    discrete_width = getattr(action_space, "n", None)
    if isinstance(discrete_width, numbers.Integral):
        return int(discrete_width) == width and tuple(getattr(action_space, "shape", ())) == ()
    return getattr(action_space, "shape", None) == (width,)


def campaign_source_files(
    root: pathlib.Path, campaign_spec: pathlib.Path | None = None
) -> list[pathlib.Path]:
    """Return tracked and untracked source inputs hashed into campaign identity."""
    pathspecs = [
        "quant_rl",
        "config",
        "scripts/train",
        "scripts/matrix",
        "scripts/variant_list.txt",
        "pyproject.toml",
        "uv.lock",
    ]
    tracked = subprocess.check_output(["git", "ls-files", "-z", *pathspecs], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", *pathspecs], cwd=root
    )
    files = {
        (root / item.decode()).resolve()
        for listing in (tracked, untracked)
        for item in listing.split(b"\0")
        if item
    }
    if campaign_spec is not None:
        files.add(campaign_spec.resolve())
    return sorted(path for path in files if path.is_file())


def write_campaign_source_snapshot(
    root: pathlib.Path,
    source_files: list[pathlib.Path],
    destination: pathlib.Path,
    *,
    git_sha: str,
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Archive the exact hashed campaign inputs before any worker is launched."""
    root = root.resolve()
    records: dict[str, str] = {}
    resolved_files: dict[str, pathlib.Path] = {}
    for path in source_files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"campaign source must be inside repository: {resolved}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"campaign source disappeared before snapshot: {resolved}")
        if relative in resolved_files:
            continue
        resolved_files[relative] = resolved
        records[relative] = sha256_file(resolved)

    if not records:
        raise ValueError("campaign source snapshot cannot be empty")
    if records != expected_source_hashes:
        raise RuntimeError("campaign source files changed while creating the source snapshot")
    snapshot_manifest = {
        "schema_version": 1,
        "git_sha": git_sha,
        "files": dict(sorted(records.items())),
    }
    manifest_bytes = (json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        for relative, path in sorted(resolved_files.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, mode="w") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
        manifest_info = zipfile.ZipInfo("source_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(manifest_info, manifest_bytes, compresslevel=6)

    with zipfile.ZipFile(temporary, mode="r") as archive:
        for relative, expected_hash in records.items():
            digest = hashlib.sha256()
            with archive.open(relative, mode="r") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"campaign source snapshot verification failed: {relative}")
        if archive.read("source_manifest.json") != manifest_bytes:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("campaign source snapshot manifest verification failed")
    temporary.replace(destination)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(records),
    }


def verify_campaign_source_snapshot(archive_path: pathlib.Path, record: object) -> tuple[bool, str]:
    """Check an archived campaign source bundle against its campaign manifest."""
    if not isinstance(record, dict):
        return False, "source snapshot record is invalid"
    expected_hash = record.get("sha256")
    if not archive_path.is_file():
        return False, "source snapshot archive is missing"
    if not isinstance(expected_hash, str) or sha256_file(archive_path) != expected_hash:
        return False, "source snapshot archive hash mismatch"
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest_bytes = archive.read("source_manifest.json")
            source_manifest = json.loads(manifest_bytes)
            files = source_manifest["files"]
            if not isinstance(files, dict) or len(files) != record.get("file_count"):
                return False, "source snapshot file manifest is invalid"
            if hashlib.sha256(manifest_bytes).hexdigest() != record.get("manifest_sha256"):
                return False, "source snapshot manifest hash mismatch"
            if set(archive.namelist()) != set(files) | {"source_manifest.json"}:
                return False, "source snapshot contents differ from its file manifest"
            for name, expected_file_hash in files.items():
                digest = hashlib.sha256()
                with archive.open(name, mode="r") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_file_hash:
                    return False, f"source snapshot file hash mismatch: {name}"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"source snapshot is unreadable: {exc}"
    return True, "verified"


def validate_campaign_evaluation(value: object) -> dict[str, Any]:
    """Validate report-only use or explicit selection on exposed historical data."""
    if not isinstance(value, dict) or value.get("deterministic") is not True:
        raise ValueError("campaign evaluation must be deterministic")
    allowed = {"locked_oos_role", "deterministic", "locked_oos_exposure", "selection_allowed"}
    if set(value) - allowed:
        raise ValueError("campaign evaluation contains unknown fields")
    exposure = value.get("locked_oos_exposure")
    if exposure is not None and exposure != "previously_inspected_holdout":
        raise ValueError("unrecognized locked-OOS exposure status")
    selection_allowed = value.get("selection_allowed", False)
    if not isinstance(selection_allowed, bool):
        raise ValueError("selection_allowed must be a boolean")
    if selection_allowed and exposure != "previously_inspected_holdout":
        raise ValueError("selection is allowed only on explicitly exposed historical data")
    expected_role = "selection_acknowledged" if selection_allowed else "final_report_only"
    if value.get("locked_oos_role") != expected_role:
        raise ValueError(f"evaluation role must be {expected_role!r} for this selection setting")
    return dict(value)


def verify_run_artifacts(
    run_out: pathlib.Path,
    key: str,
    expected_steps: int,
    *,
    expected_fingerprint: str,
    expected_attempt: int,
) -> tuple[bool, str]:
    """Require the requested budget, a readable checkpoint, and a valid training record."""
    provenance_path = run_out / "attempt_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "missing or invalid attempt_provenance.json"
    variant, _, seed_text = key.rpartition("__seed")
    if (
        not isinstance(provenance, dict)
        or not seed_text.isdigit()
        or provenance.get("run_key") != key
        or provenance.get("variant") != variant
        or provenance.get("seed") != int(seed_text)
        or provenance.get("requested_timesteps") != expected_steps
        or provenance.get("attempt") != expected_attempt
        or provenance.get("campaign_fingerprint") != expected_fingerprint
    ):
        return False, "attempt provenance does not match the expected campaign run"
    identity = provenance.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("steps") != expected_steps
        or not isinstance(identity.get("jobs"), list)
        or key not in identity["jobs"]
    ):
        return False, "attempt provenance identity does not include the requested run"
    identity_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_fingerprint != expected_fingerprint:
        return False, "attempt provenance identity does not match the campaign fingerprint"
    for path in run_out.glob("**/training_log.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            not isinstance(payload.get("env_flags"), dict)
            or not isinstance(payload.get("observation_space_shapes"), dict)
            or not isinstance(payload.get("action_space_width"), int)
        ):
            continue
        if not seed_text.isdigit() or payload.get("seed") != int(seed_text):
            continue
        try:
            config_manifest = json.loads(
                (path.parent.parent / "resolved_config.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if not isinstance(config_manifest, dict):
            continue
        if (
            config_manifest.get("variant") != variant
            or config_manifest.get("seed") != int(seed_text)
            or config_manifest.get("steps") != expected_steps
        ):
            continue
        required_metrics = (
            "test_sharpe",
            "test_max_dd",
            "test_trades",
            "test_return",
            "test_survived_full_year",
            "train_equity_ok",
            "feature_cache_hash",
            "feature_columns",
            "feature_schema_sha256",
            "effective_config_sha256",
            "action_space_width",
            "observation_space_shapes",
            "env_flags",
        )
        if any(key not in payload for key in required_metrics):
            continue
        numeric_metrics = ("test_sharpe", "test_max_dd", "test_return")
        if any(
            not isinstance(payload[key], (int, float)) or not math.isfinite(payload[key])
            for key in numeric_metrics
        ):
            continue
        if (
            not isinstance(payload["test_trades"], int)
            or not isinstance(payload["test_survived_full_year"], bool)
            or not isinstance(payload["train_equity_ok"], bool)
        ):
            continue
        if not payload["train_equity_ok"]:
            reason = "training risk-survival gate failed"
            gate_path = path.parent / "train_equity_gate.json"
            try:
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                gate = None
            if isinstance(gate, dict) and isinstance(gate.get("reason"), str):
                reason += f": {gate['reason']}"
            return False, reason
        # The cache sidecar is intentionally a 16-hex content key produced by
        # feature_cache_content_hash(); the full parquet SHA-256 is recorded
        # separately in run_manifest.json. Requiring 64 characters here made
        # every otherwise-complete run fail artifact admission.
        cache_hash = payload.get("feature_cache_hash")
        cache_name = payload.get("feature_cache_name")
        manifest_names = payload.get("run_manifests")
        if (
            not isinstance(cache_hash, str)
            or not re.fullmatch(r"[0-9a-f]{16}", cache_hash)
            or not isinstance(cache_name, str)
            or not cache_name
            or not isinstance(manifest_names, list)
            or not manifest_names
            or any(not isinstance(name, str) or not name for name in manifest_names)
        ):
            continue
        cache_manifest_checkpoint_hash: str | None = None
        cache_manifest_valid = True
        for name in manifest_names:
            manifest_path = (path.parent / name).resolve()
            try:
                manifest_path.relative_to(path.parent.resolve())
                run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache_manifest_valid = False
                break
            if not isinstance(run_manifest, dict):
                cache_manifest_valid = False
                break
            feature_cache = run_manifest.get("feature_cache")
            checkpoint_manifest = run_manifest.get("checkpoint")
            config_manifest = run_manifest.get("config")
            if not isinstance(feature_cache, dict):
                cache_manifest_valid = False
                break
            if not isinstance(checkpoint_manifest, dict):
                cache_manifest_valid = False
                break
            if not isinstance(config_manifest, dict):
                cache_manifest_valid = False
                break
            cache_path_text = feature_cache.get("path")
            cache_file_hash = feature_cache.get("file_sha256")
            if (
                feature_cache.get("identity") != cache_name
                or feature_cache.get("sha256") != cache_hash
                or not isinstance(cache_path_text, str)
                or not isinstance(cache_file_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", cache_file_hash)
                or config_manifest.get("sha256") != payload.get("effective_config_sha256")
                or not isinstance(checkpoint_manifest.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_manifest["sha256"])
            ):
                cache_manifest_valid = False
                break
            cache_path = pathlib.Path(cache_path_text)
            sidecar_path = pathlib.Path(str(cache_path) + ".hash")
            try:
                if (
                    not cache_path.is_file()
                    or sidecar_path.read_text(encoding="utf-8").strip() != cache_hash
                    or cached_sha256_file(cache_path) != cache_file_hash
                ):
                    cache_manifest_valid = False
                    break
            except OSError:
                cache_manifest_valid = False
                break
            if cache_manifest_checkpoint_hash not in (None, checkpoint_manifest["sha256"]):
                cache_manifest_valid = False
                break
            cache_manifest_checkpoint_hash = checkpoint_manifest["sha256"]
        if not cache_manifest_valid:
            continue
        feature_columns = payload.get("feature_columns")
        schema_hash = payload.get("feature_schema_sha256")
        if not isinstance(feature_columns, list) or any(
            not isinstance(column, str) for column in feature_columns
        ):
            continue
        actual_schema_hash = hashlib.sha256(
            json.dumps(feature_columns, separators=(",", ":")).encode()
        ).hexdigest()
        if schema_hash != actual_schema_hash:
            continue
        import yaml

        try:
            effective_config = yaml.safe_load(
                (path.parent / "config.yaml").read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(effective_config, dict) or not isinstance(
            effective_config.get("env"), dict
        ):
            continue
        config_hash = payload.get("effective_config_sha256")
        if (
            not isinstance(config_hash, str)
            or sha256_file(path.parent / "config.yaml") != config_hash
        ):
            continue
        if any(
            effective_config["env"].get(name) != value
            for name, value in payload["env_flags"].items()
        ):
            continue
        completed = payload.get("timesteps_completed")
        if not isinstance(completed, int) or completed < expected_steps:
            continue
        if payload.get("timesteps") != expected_steps:
            continue
        model_file = path.parent / "model" / "ppo_final.zip"
        if not model_file.is_file() or model_file.stat().st_size == 0:
            continue
        if cache_manifest_checkpoint_hash != sha256_file(model_file):
            continue
        try:
            with zipfile.ZipFile(model_file) as archive:
                if archive.testzip() is not None:
                    continue
            algo = str(payload.get("algo", "")).lower()
            model_class: Any
            if algo == "ppo":
                from stable_baselines3 import PPO

                model_class = PPO
            elif algo == "sac":
                from stable_baselines3 import SAC

                model_class = SAC
            else:
                continue
            model = model_class.load(model_file, device="cpu")
            if int(model.num_timesteps) < expected_steps:
                continue
            if not action_space_matches_width(model.action_space, payload["action_space_width"]):
                continue
            model_obs = getattr(model.observation_space, "spaces", {})
            model_obs_shapes = {
                str(name): [int(size) for size in space.shape]
                for name, space in model_obs.items()
                if hasattr(space, "shape")
            }
            if model_obs_shapes != payload["observation_space_shapes"]:
                continue
        except Exception:
            continue
        return True, ""
    return False, f"missing valid training record/checkpoint at requested {expected_steps} steps"


def hash_run_artifacts(run_out: pathlib.Path) -> dict[str, str]:
    """Record checksums for completed logs, configs, and model checkpoints."""
    suffixes = {
        "training_log.json",
        "ppo_final.zip",
        "config.yaml",
        "config.json",
        "attempt_provenance.json",
    }
    return {
        str(path.relative_to(run_out)): sha256_file(path)
        for path in run_out.rglob("*")
        if path.is_file() and (path.name in suffixes or path.name == "resolved_config.json")
    }


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
        if attempt < 1 or int(parent_job.get("attempts", 0)) != attempt:
            raise ValueError(f"source attempt mismatch for {key}")
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
            attempt = int(job.get("attempts", 0))
            if attempt < 1:
                continue
            checkpoint = latest_resumable_checkpoint(
                source_dir, key, attempt, fingerprint, max_steps=target_steps
            )
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
    from stable_baselines3 import PPO

    loaded_steps: dict[str, int] = {}
    for key, path in archived.items():
        model = PPO.load(path, device="cpu")
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


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "omegaconf",
        "pyyaml",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "gpu": gpu,
    }


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
            ROOT / str(spec.get("variant_file", "scripts/variant_list.txt"))
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
            (ROOT / "quant_rl" / "config" / "default.yaml").read_text(encoding="utf-8")
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
    raw_config = yaml.safe_load((ROOT / "quant_rl" / "config" / "default.yaml").read_text())
    raw_dir = ROOT / str(raw_config["data"]["raw_dir"])
    data_inputs = {}
    for group in ("m1_files", "tick_files"):
        for filename in raw_config["data"].get(group, {}).values():
            path = raw_dir / str(filename)
            data_inputs[str(path.relative_to(ROOT))] = (
                sha256_file(path) if path.is_file() else "MISSING"
            )
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
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        if not args.resume:
            raise RuntimeError("campaign exists: use --resume")
        if manifest["fingerprint"] != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
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
                    str(ROOT / "scripts" / "triton" / "sample_campaign_resources.py"),
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
            else ["bash", str(ROOT / "scripts" / "train" / "train_one_variant.sh")]
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
        summary_script = pathlib.Path(__file__).resolve().with_name("summarize_campaign_results.py")
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
