"""Focused campaign execution resources services."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

from experiments.campaigns.constants import ROOT
from experiments.campaigns.state_store import utc

WORKER_CPU_AFFINITY_GUARD = ROOT / "scripts" / "compute" / "slurm" / "verify_worker_cpu_affinity.py"


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


def worker_node_cpu_ids(queue_root: pathlib.Path, job_id: str, node: str) -> list[int]:
    """Read the CPU set advertised by the persistent manager on one worker node."""
    ready = queue_root / job_id / node.split(".", maxsplit=1)[0] / "ready"
    try:
        metadata = json.loads(ready.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"worker manager CPU metadata is unavailable for {node}: {exc}") from exc
    node_key = node.split(".", maxsplit=1)[0]
    if metadata.get("node") != node_key or str(metadata.get("job_id")) != job_id:
        raise RuntimeError(f"worker manager CPU metadata identity mismatch for {node}")
    cpu_ids = metadata.get("cpu_ids")
    if (
        not isinstance(cpu_ids, list)
        or not cpu_ids
        or any(not isinstance(cpu, int) or cpu < 0 for cpu in cpu_ids)
        or len(set(cpu_ids)) != len(cpu_ids)
    ):
        raise RuntimeError(f"worker manager advertised an invalid CPU set for {node}")
    return sorted(cpu_ids)


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
    step_cpus: int | None = None,
) -> list[str]:
    """Build an allocation-local or explicitly GPU-pinned Slurm worker command."""
    if node is None:
        return command
    if gpu_count < 0 or (gpu_count and not 0 <= gpu_id < gpu_count):
        raise ValueError("worker GPU count or binding index is invalid")
    if (cpu_mask is None) != (expected_cpu_ids is None):
        raise ValueError("CPU mask and expected CPU IDs must be provided together")
    allocated_cpus = cpus if step_cpus is None else step_cpus
    if allocated_cpus < cpus:
        raise ValueError("Slurm step CPU envelope cannot be smaller than the worker CPU mask")
    result = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        f"--cpus-per-task={allocated_cpus}",
    ]
    if gpu_count:
        result.extend([f"--gpus={gpu_type}:{gpu_count}", f"--gpu-bind=map_gpu:{gpu_id}"])
    cpu_binding = f"mask_cpu:{cpu_mask}" if cpu_mask is not None else "cores"
    worker_command = command
    if cpu_mask is not None and expected_cpu_ids is not None:
        worker_command = [
            sys.executable,
            str(WORKER_CPU_AFFINITY_GUARD),
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
