#!/usr/bin/env python3
"""Append process-tree PSS and GPU samples while a campaign controller runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import socket
import subprocess
import time
from typing import Any


def read_process_table() -> dict[int, int]:
    table: dict[int, int] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        ppid = next(
            (line.split()[1] for line in status.splitlines() if line.startswith("PPid:")), None
        )
        if ppid is not None:
            table[int(entry.name)] = int(ppid)
    return table


def descendant_pids(roots: list[int], parents: dict[int, int]) -> set[int]:
    descendants = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def pss_bytes(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            text = pathlib.Path(f"/proc/{pid}/smaps_rollup").read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("Pss:"):
                total += int(line.split()[1]) * 1024
                break
    return total


def process_io_bytes(pids: set[int]) -> tuple[int, int, int, int]:
    """Aggregate Linux process I/O counters for the campaign process tree."""
    totals = [0, 0, 0, 0]
    keys = ("rchar", "wchar", "read_bytes", "write_bytes")
    for pid in pids:
        try:
            lines = pathlib.Path(f"/proc/{pid}/io").read_text().splitlines()
        except OSError:
            continue
        counters: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                continue
            key = parts[0].rstrip(":")
            if key in keys:
                counters[key] = int(parts[1])
        for index, key in enumerate(keys):
            totals[index] += counters.get(key, 0)
    return tuple(totals)  # type: ignore[return-value]


def _process_stat_cpu_usec(text: str, ticks_per_second: int) -> int:
    """Read user+system CPU time from one Linux ``/proc/<pid>/stat`` record."""
    # comm is parenthesized and may itself contain spaces or ')' characters.
    fields = text[text.rfind(")") + 2 :].split()
    if len(fields) <= 12:
        return 0
    ticks = int(fields[11]) + int(fields[12])
    return int(ticks * 1_000_000 / ticks_per_second)


def process_tree_cpu_usec(pids: set[int]) -> int:
    """Aggregate cumulative user+system CPU time for the tracked process tree."""
    ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    total = 0
    for pid in pids:
        try:
            stat_text = pathlib.Path(f"/proc/{pid}/stat").read_text()
            total += _process_stat_cpu_usec(stat_text, ticks_per_second)
        except (OSError, ValueError):
            continue
    return total


def cgroup_file(
    name: str, *, proc_cgroup: pathlib.Path | None = None, cgroup_root: pathlib.Path | None = None
) -> pathlib.Path | None:
    """Find a cgroup v2 or v1 memory file for this process, safely below root."""
    root = cgroup_root or pathlib.Path("/sys/fs/cgroup")
    proc = proc_cgroup or pathlib.Path("/proc/self/cgroup")
    try:
        entries = proc.read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        parts = entry.split(":", maxsplit=2)
        if len(parts) != 3:
            continue
        _hierarchy, controllers, relative = parts
        if controllers == "":  # cgroup v2
            base = root / relative.lstrip("/")
        elif "memory" in controllers.split(","):  # cgroup v1
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
    # Some containers expose the mounted cgroup directly without a matching
    # process-relative entry. Keep this fallback for those environments.
    for fallback in (root / name, root / "memory" / name):
        try:
            fallback.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if fallback.is_file():
            return fallback
    return None


def allocation_cgroup_file(
    name: str, *, proc_cgroup: pathlib.Path | None = None, cgroup_root: pathlib.Path | None = None
) -> pathlib.Path | None:
    """Resolve a Slurm job-level cgroup file, not the sampler's step cgroup."""
    root = cgroup_root or pathlib.Path("/sys/fs/cgroup")
    path = cgroup_file(name, proc_cgroup=proc_cgroup, cgroup_root=root)
    if path is None:
        return None
    current = path.parent
    while current != root and root in current.parents:
        if current.name.startswith("job_"):
            candidate = current / name
            if candidate.is_file():
                return candidate
        current = current.parent
    return path


def cgroup_value(name: str) -> str:
    path = allocation_cgroup_file(name)
    try:
        return path.read_text().strip() if path is not None else ""
    except OSError:
        return ""


def cgroup_memory_scope() -> str:
    """Label whether memory fields cover the Slurm job or only the sampler step."""
    for name in ("memory.current", "memory.usage_in_bytes"):
        process_path = cgroup_file(name)
        if process_path is None:
            continue
        job_path = allocation_cgroup_file(name)
        return "slurm_job" if job_path != process_path else "process_cgroup"
    return "unavailable"


def gpu_sample() -> tuple[list[int], list[int]]:
    """Return utilization and memory use for each visible GPU, in device order."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return [], []
    utilizations: list[int] = []
    memory_used: list[int] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            utilizations.append(int(fields[0]))
            memory_used.append(int(fields[1]))
        except ValueError:
            continue
    return utilizations, memory_used


def active_roots(path: pathlib.Path) -> tuple[list[int], bool]:
    try:
        campaign: dict[str, Any] = json.loads(path.read_text())
    except (OSError, ValueError):
        return [], False
    if campaign.get("stopping") is True:
        return [], False
    jobs = campaign.get("jobs", [])
    active = [job for job in jobs if job.get("status") in ("running", "pending")]
    local_host = socket.gethostname().split(".", maxsplit=1)[0]
    roots: list[int] = []
    for job in active:
        if job.get("status") != "running":
            continue
        pid_file = job.get("remote_pid_file")
        if pid_file:
            try:
                remote = json.loads(pathlib.Path(pid_file).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                isinstance(remote, dict)
                and remote.get("hostname") == local_host
                and isinstance(remote.get("pid"), int)
            ):
                roots.append(remote["pid"])
            continue
        if job.get("worker_node") in (None, local_host) and job.get("pid"):
            roots.append(int(job["pid"]))
    return roots, bool(active)


def sample(path: pathlib.Path, campaign: pathlib.Path) -> None:
    roots, active = active_roots(campaign)
    parents = read_process_table()
    tree = descendant_pids(roots, parents)
    io_rchar, io_wchar, io_read_bytes, io_write_bytes = process_io_bytes(tree)
    gpu_utils, gpu_memories = gpu_sample()
    gpu_util = round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else ""
    gpu_memory = sum(gpu_memories) if gpu_memories else ""
    row = [
        dt.datetime.now(dt.UTC).isoformat(),
        os.getloadavg()[0],
        len(tree),
        pss_bytes(tree),
        cgroup_value("memory.current"),
        cgroup_value("memory.max") or cgroup_value("memory.limit_in_bytes"),
        cgroup_value("memory.peak"),
        cgroup_value("memory.events"),
        io_rchar,
        io_wchar,
        io_read_bytes,
        io_write_bytes,
        gpu_util,
        gpu_memory,
        json.dumps(roots),
        int(active),
        process_tree_cpu_usec(tree),
        json.dumps(gpu_utils),
        json.dumps(gpu_memories),
        cgroup_memory_scope(),
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "timestamp_utc",
                    "load1",
                    "process_tree_count",
                    "process_tree_pss_bytes",
                    "cgroup_memory_current",
                    "cgroup_memory_max",
                    "cgroup_memory_peak",
                    "cgroup_memory_events",
                    "process_tree_rchar_bytes",
                    "process_tree_wchar_bytes",
                    "process_tree_storage_read_bytes",
                    "process_tree_storage_write_bytes",
                    "gpu_utilization_pct",
                    "gpu_memory_used_mib",
                    "root_pids",
                    "campaign_active",
                    "process_tree_cpu_time_usec",
                    "gpu_utilization_pct_by_device",
                    "gpu_memory_used_mib_by_device",
                    "cgroup_memory_scope",
                ]
            )
    while True:
        _, active = active_roots(args.campaign)
        if not active:
            break
        sample(args.out, args.campaign)
        time.sleep(args.interval)
    sample(args.out, args.campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
