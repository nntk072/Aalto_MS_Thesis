#!/usr/bin/env python3
"""Read-only per-node Slurm inventory for GPU allocation planning.

Run this on a Triton login node before choosing a GPU allocation:

    python3 scripts/compute/slurm/node_inventory.py
    python3 scripts/compute/slurm/node_inventory.py --partition gpu-grace-h200-141g
    python3 scripts/compute/slurm/node_inventory.py --node gpuarm1

This reports scheduler inventory only. It does not submit, cancel, or modify jobs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime


def run(command: Sequence[str]) -> str:
    """Run a read-only Slurm query and fail with a useful message."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit {exc.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}") from exc
    return result.stdout


def fields(record: str) -> dict[str, str]:
    """Parse the space-delimited key=value fields from scontrol -o output."""
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", record))


def tres_counts(value: str) -> dict[str, int]:
    """Return GPU GRES counts from a CfgTRES or AllocTRES field."""
    counts: dict[str, int] = {}
    for item in value.split(","):
        match = re.fullmatch(r"gres/gpu(?::([^=]+))?=(\d+)", item)
        if match:
            gpu_type = match.group(1) or "all"
            counts[gpu_type] = int(match.group(2))
    return counts


def gres_counts(value: str) -> dict[str, int]:
    """Fallback GPU inventory parser for Slurm Gres/GresUsed fields."""
    counts: dict[str, int] = {}
    for match in re.finditer(r"gpu:([^:,() ]+):(\d+)", value):
        gpu_type, count = match.groups()
        counts[gpu_type] = counts.get(gpu_type, 0) + int(count)
    return counts


def number(value: str, default: int = 0) -> int:
    """Parse Slurm numeric memory fields, which may have a K/M/G/T suffix."""
    match = re.fullmatch(r"(\d+)([KMGTP]?)", value, flags=re.IGNORECASE)
    if not match:
        return default
    amount = int(match.group(1))
    suffix = match.group(2).upper()
    # Slurm RealMemory/AllocMem/FreeMem are normally MiB. Convert suffixed
    # values to MiB so all RAM columns share the same unit.
    factors = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "P": 1024**3}
    return round(amount * factors[suffix])


def gibibytes(mebibytes: int) -> str:
    return f"{mebibytes / 1024:.1f} GiB"


def parse_node_records(raw: str) -> list[dict[str, str]]:
    return [fields(line) for line in raw.splitlines() if line.strip()]


def node_gpu_summary(record: dict[str, str]) -> tuple[int, int, str]:
    """Return configured, allocated, and readable per-type GPU counts."""
    configured_types = tres_counts(record.get("CfgTRES", ""))
    allocated_types = tres_counts(record.get("AllocTRES", ""))
    configured_total = configured_types.get("all", sum(configured_types.values()))
    allocated_total = allocated_types.get("all", sum(allocated_types.values()))

    # Some Slurm configurations expose only the generic total in TRES. Use
    # Gres/GresUsed as a typed fallback; never invent a per-type split when
    # an aggregate allocation spans multiple configured GPU types.
    if not configured_types or set(configured_types) == {"all"}:
        configured_types = gres_counts(record.get("Gres", ""))
        configured_total = sum(configured_types.values())
    if not allocated_types or set(allocated_types) == {"all"}:
        used_types = gres_counts(record.get("GresUsed", ""))
        if used_types:
            allocated_types = used_types
        elif len(configured_types) == 1:
            only_type = next(iter(configured_types))
            allocated_types = {only_type: allocated_total}

    type_labels: list[str] = []
    unknown_allocation = False
    for gpu_type, total in sorted(configured_types.items()):
        if gpu_type == "all":
            continue
        if gpu_type in allocated_types:
            used = allocated_types[gpu_type]
            type_labels.append(f"{gpu_type} {max(0, total - used)}/{total} free")
        elif allocated_total == 0:
            type_labels.append(f"{gpu_type} {total}/{total} free")
        else:
            unknown_allocation = True

    if unknown_allocation:
        type_labels.append(
            f"type split unknown; aggregate {max(0, configured_total - allocated_total)}/{configured_total} free"
        )
    if not type_labels:
        type_labels.append(
            f"GPU {max(0, configured_total - allocated_total)}/{configured_total} free"
        )
    return configured_total, allocated_total, ", ".join(type_labels)


def gpu_tres(value: str) -> dict[str, int]:
    """Read typed or generic GPU allocations from a TRES string."""
    typed: dict[str, int] = {}
    generic = 0
    for item in value.split(","):
        match = re.fullmatch(r"gres/gpu(?::([^=]+))?=(\d+)", item)
        if match:
            gpu_type, count = match.groups()
            if gpu_type:
                typed[gpu_type] = int(count)
            else:
                generic = int(count)
            continue
        # squeue JSON uses the TresPerNode spelling `gres/gpu:1` or
        # `gres/gpu:h100:1`, rather than the key=value AllocTRES spelling.
        if item.startswith("gres/gpu:"):
            parts = item.removeprefix("gres/gpu:").split(":")
            if len(parts) == 1 and parts[0].isdigit():
                generic = int(parts[0])
            elif len(parts) == 2 and parts[1].isdigit():
                typed[parts[0]] = int(parts[1])
    return typed or ({"all": generic} if generic else {})


def expand_nodes(node_list: str, known_nodes: set[str], cache: dict[str, list[str]]) -> list[str]:
    """Expand a Slurm hostlist only when it is not already one node name."""
    if not node_list:
        return []
    if node_list in cache:
        return cache[node_list]
    if node_list in known_nodes:
        cache[node_list] = [node_list]
    elif "[" not in node_list and "]" not in node_list:
        cache[node_list] = [part for part in node_list.split(",") if part]
    else:
        try:
            expanded = run(["scontrol", "show", "hostnames", node_list]).split()
        except RuntimeError:
            expanded = [part for part in node_list.split(",") if part in known_nodes]
        cache[node_list] = expanded
    return cache[node_list]


def running_gpu_jobs(nodes: set[str]) -> dict[str, list[dict[str, object]]]:
    """Map currently running GPU jobs to nodes and their scheduled end times."""
    try:
        payload = json.loads(run(["squeue", "--json", "-t", "RUNNING"]))
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"WARNING: running-job end-time estimates unavailable: {exc}", file=sys.stderr)
        return {}

    by_node: dict[str, list[dict[str, object]]] = {}
    hostlist_cache: dict[str, list[str]] = {}
    for job in payload.get("jobs", []):
        state = job.get("job_state", [])
        if isinstance(state, str):
            state = [state]
        if "RUNNING" not in state:
            continue

        allocated = gpu_tres(str(job.get("tres_alloc_str", "")))
        if not allocated:
            continue

        job_nodes = expand_nodes(str(job.get("nodes", "")), nodes, hostlist_cache)
        matching_nodes = [node for node in job_nodes if node in nodes]
        if not matching_nodes:
            continue

        per_node = gpu_tres(str(job.get("tres_per_node", "")))
        if per_node:
            gpu_allocation = per_node
        elif len(matching_nodes) == 1:
            gpu_allocation = allocated
        else:
            divisor = len(job_nodes)
            gpu_allocation = {
                gpu_type: count // divisor
                for gpu_type, count in allocated.items()
                if count // divisor > 0
            }

        end_time = job.get("end_time", {})
        end_epoch = (
            int(end_time["number"])
            if isinstance(end_time, dict) and end_time.get("set") and not end_time.get("infinite")
            else None
        )
        info: dict[str, object] = {
            "job_id": str(job.get("job_id", "?")),
            "user": str(job.get("user_name", "?")),
            "partition": str(job.get("partition", "?")),
            "end_epoch": end_epoch,
            "gpu_allocation": gpu_allocation,
        }
        for node in matching_nodes:
            by_node.setdefault(node, []).append(info)
    return by_node


def next_release_text(
    state: str,
    gpu_total: int,
    gpu_allocated: int,
    jobs: list[dict[str, object]],
) -> str:
    """Describe current free GPUs or the earliest running GPU job end."""
    flags = {flag.upper() for flag in state.split("+")}
    blockers = sorted(
        flags & {"DRAIN", "DRAINING", "DOWN", "FAIL", "MAINT", "REBOOT_REQUESTED", "RESERVED"}
    )
    planned = "planned jobs may claim it" if "PLANNED" in flags else ""

    if gpu_total > gpu_allocated:
        free_now = gpu_total - gpu_allocated
        text = f"NOW: {free_now} GPU free"
        notes = blockers + ([planned] if planned else [])
        return text + (f" ({'; '.join(notes)})" if notes else "")

    releases = [
        (end_epoch, job) for job in jobs if isinstance((end_epoch := job.get("end_epoch")), int)
    ]
    if not releases:
        return "unknown: no visible GPU job end time"
    earliest = min(end_epoch for end_epoch, _ in releases)
    first_jobs = [job for end_epoch, job in releases if end_epoch == earliest]
    end = datetime.fromtimestamp(earliest).astimezone()
    remaining_seconds = max(0, earliest - int(datetime.now().timestamp()))
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes = remainder // 60
    job_ids = ",".join(str(job["job_id"]) for job in first_jobs[:4])
    if len(first_jobs) > 4:
        job_ids += f",+{len(first_jobs) - 4}"
    text = f"~{end:%Y-%m-%d %H:%M %Z} (in {hours}h {minutes}m; job {job_ids})"
    notes = blockers + ([planned] if planned else [])
    if notes:
        text += f" ({'; '.join(notes)})"
    return text


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    return "\n".join(
        ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
        + ["  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows]
    )


def print_pending_gpu_jobs(partitions: set[str], *, show_examples: bool) -> None:
    payload = json.loads(run(["squeue", "--json", "-t", "PENDING"]))
    grouped: dict[str, list[list[str]]] = {}
    for job in payload.get("jobs", []):
        partition = str(job.get("partition", ""))
        requested = ""
        for key in ("tres_per_job", "tres_per_node", "tres_per_task"):
            requested = str(job.get(key, ""))
            if gpu_tres(requested):
                break
        if not requested or not gpu_tres(requested):
            requested = str(job.get("tres_req_str", ""))
        if partition not in partitions or "gres/gpu" not in requested.lower():
            continue
        cpus_data = job.get("cpus_per_task", {})
        cpus = str(cpus_data.get("number", "?")) if isinstance(cpus_data, dict) else "?"
        memory_match = re.search(r"(?:^|,)mem=([^,]+)", str(job.get("tres_req_str", "")))
        memory = memory_match.group(1) if memory_match else "?"
        grouped.setdefault(partition, []).append(
            [
                str(job.get("job_id", "?")),
                str(job.get("user_name", "?")),
                str(job.get("name", "?")),
                requested,
                cpus,
                memory,
                str(job.get("state_reason", "")) or "-",
            ]
        )

    print("\nPending GPU requests in the listed GPU partitions:")
    if not grouped:
        print("  none reported")
        return
    for partition, jobs in sorted(grouped.items()):
        if not show_examples:
            print(f"  {partition}: {len(jobs)} pending GPU requests")
            continue
        print(f"  {partition}: {len(jobs)} pending")
        for job in jobs[:8]:
            print("    job={} user={} name={} gpu={} cpu={} mem={} reason={}".format(*job))
        if len(jobs) > 8:
            print(f"    ... plus {len(jobs) - 8} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", help="show only nodes belonging to this partition")
    parser.add_argument("--node", help="show only this node")
    args = parser.parse_args()

    try:
        sinfo_raw = run(["sinfo", "-N", "-h", "-o", "%N|%P|%T|%C|%m|%e|%G"])
        scontrol_raw = run(["scontrol", "show", "nodes", "-o"])
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # sinfo lists each node's partition membership; scontrol supplies detailed
    # per-node allocation counters. A node appears more than once when it is
    # exposed through multiple partitions.
    sinfo_partitions: dict[str, set[str]] = {}
    for line in sinfo_raw.splitlines():
        pieces = line.split("|", 2)
        if len(pieces) == 3:
            node_name, partition, _state = pieces
            sinfo_partitions.setdefault(node_name, set()).add(partition)
    if not sinfo_partitions:
        print("WARNING: sinfo returned no node rows.", file=sys.stderr)

    records = parse_node_records(scontrol_raw)
    selected_records: list[dict[str, str]] = []
    gpu_partitions: set[str] = set()
    gpu_nodes: list[str] = []
    for record in records:
        name = record.get("NodeName", "")
        if not name:
            continue
        cfg = tres_counts(record.get("CfgTRES", ""))
        gres = gres_counts(record.get("Gres", ""))
        if not cfg and not gres:
            continue

        partitions = sorted(
            sinfo_partitions.get(name) or {p for p in record.get("Partitions", "").split(",") if p}
        )
        if args.partition and args.partition not in partitions:
            continue
        if args.node and args.node != name:
            continue

        selected_records.append(record)
        gpu_partitions.update(partitions)
        gpu_nodes.append(name)

    rows: list[list[str]] = []
    gpu_job_map = running_gpu_jobs(set(gpu_nodes))
    for record in selected_records:
        name = record.get("NodeName", "")
        partitions = sorted(
            sinfo_partitions.get(name) or {p for p in record.get("Partitions", "").split(",") if p}
        )
        state = record.get("State", "unknown")
        cpu_total = number(record.get("CPUTot", "0"))
        cpu_alloc = number(record.get("CPUAlloc", "0"))
        real_mem = number(record.get("RealMemory", "0"))
        alloc_mem = number(record.get("AllocMem", "0"))
        free_mem = number(record.get("FreeMem", "0"))
        gpu_total, gpu_alloc, gpu_text = node_gpu_summary(record)
        release = next_release_text(
            state,
            gpu_total,
            gpu_alloc,
            gpu_job_map.get(name, []),
        )
        rows.append(
            [
                name,
                ",".join(partitions) or "-",
                state,
                gpu_text,
                f"{cpu_total - cpu_alloc}/{cpu_total}",
                f"{gibibytes(max(0, real_mem - alloc_mem))}/{gibibytes(real_mem)}",
                gibibytes(free_mem),
                release,
            ]
        )

    rows.sort(key=lambda row: row[0])
    print(f"Slurm node inventory — {run(['date', '+%Y-%m-%d %H:%M:%S %Z']).strip()}")
    print(f"Host: {run(['hostname', '-f']).strip()} | user: {run(['id', '-un']).strip()}")
    print("Read-only: this script does not allocate, cancel, or modify jobs.\n")
    if args.node and not rows:
        print(f"No GPU node matched --node {args.node!r}.")
        return 1
    if not rows:
        print("No GPU nodes matched the selected filter.")
        return 0

    print(
        format_table(
            [
                "NODE",
                "PARTITIONS",
                "STATE",
                "GPU FREE / TOTAL BY TYPE",
                "CPU FREE / TOTAL",
                "SLURM RAM FREE / TOTAL",
                "OS FreeMem",
                "NEXT GPU FREE (estimate)",
            ],
            rows,
        )
    )

    fully_allocated_releases: list[tuple[int, str, str]] = []
    free_now = 0
    for record in selected_records:
        name = record.get("NodeName", "")
        gpu_total, gpu_alloc, _ = node_gpu_summary(record)
        free_now += max(0, gpu_total - gpu_alloc)
        if gpu_total <= gpu_alloc:
            for job in gpu_job_map.get(name, []):
                end_epoch = job.get("end_epoch")
                if isinstance(end_epoch, int):
                    fully_allocated_releases.append((end_epoch, name, str(job.get("job_id", "?"))))
    print(f"\nGPU devices currently free by Slurm accounting: {free_now}")
    if fully_allocated_releases:
        earliest, node, job_id = min(fully_allocated_releases)
        end = datetime.fromtimestamp(earliest).astimezone()
        remaining_seconds = max(0, earliest - int(datetime.now().timestamp()))
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes = remainder // 60
        print(
            "Next scheduled release among fully allocated nodes: "
            f"{node} at {end:%Y-%m-%d %H:%M %Z} (in {hours}h {minutes}m; job {job_id})."
        )
    else:
        print("No scheduled GPU release time could be derived from running GPU jobs.")

    try:
        user_jobs = run(
            [
                "squeue",
                "-u",
                run(["id", "-un"]).strip(),
                "-h",
                "-o",
                "%i|%P|%j|%T|%N|%b|%C|%m|%M|%R",
            ]
        )
        print("\nYour Slurm jobs (including the current allocation/check):")
        job_rows = [line.split("|", 9) for line in user_jobs.splitlines()]
        if job_rows:
            print(
                format_table(
                    [
                        "JOBID",
                        "PARTITION",
                        "NAME",
                        "STATE",
                        "NODE",
                        "GPU REQUEST",
                        "CPUS",
                        "MEMORY",
                        "ELAPSED",
                        "REASON",
                    ],
                    job_rows,
                )
            )
        else:
            print("  none")
        if any(len(row) > 3 and row[3] == "RUNNING" for row in job_rows):
            print(
                "  RUNNING means Slurm assigned these resources; the job name and queue do not "
                "show live GPU utilization or the child command inside an interactive shell."
            )

        if args.node:
            assigned = run(
                [
                    "squeue",
                    "-h",
                    "-w",
                    args.node,
                    "-o",
                    "%i|%u|%P|%T|%N|%b|%C|%m",
                ]
            )
            print(f"\nJobs currently assigned to {args.node}:")
            print("\n".join(f"  {line}" for line in assigned.splitlines()) or "  none")
        else:
            assigned_partitions = run(["squeue", "-h", "-w", ",".join(gpu_nodes), "-o", "%P"])
            active_counts = Counter(
                partition for partition in assigned_partitions.splitlines() if partition
            )
            print("\nActive jobs on listed GPU nodes by partition:")
            print(
                "\n".join(
                    f"  {partition}: {count} running jobs"
                    for partition, count in sorted(active_counts.items())
                )
                or "  none"
            )
        print_pending_gpu_jobs(
            gpu_partitions,
            show_examples=bool(args.node or args.partition),
        )
    except RuntimeError as exc:
        print(f"\nWARNING: could not read complete queue details: {exc}", file=sys.stderr)

    print(
        "\nInterpretation: next-release times are running-job time-limit estimates, "
        "not guarantees; jobs may finish early, be extended, or have GPUs claimed "
        "by pending jobs. GPU free counts come from Slurm CfgTRES/AllocTRES "
        "(or Gres/GresUsed fallback); CPU free is CPUTot-CPUAlloc; Slurm RAM "
        "free is RealMemory-AllocMem; OS FreeMem is a separate live reading. "
        "A free GPU count is not an allocation guarantee. Check node state, "
        "partition/account/QOS limits, reservations, pending jobs, architecture, "
        "and your workload's CPU/RAM needs before requesting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
