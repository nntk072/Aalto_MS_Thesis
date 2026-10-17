#!/usr/bin/env python3
"""Check whether a partition currently has the requested GH200 node capacity."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def _field(output: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", output)
    return match.group(1) if match else None


def _tres_value(tres: str, name: str) -> int:
    match = re.search(rf"(?:^|,){re.escape(name)}=(\d+)(?:,|$)", tres)
    return int(match.group(1)) if match else 0


def node_has_capacity(
    output: str,
    *,
    gpu_type: str,
    gpus: int,
    cpus: int,
    memory_mb: int,
) -> bool:
    """Return whether one Slurm node has the requested free resources."""
    state = _field(output, "State")
    if state is None:
        return False
    state_flags = set(state.upper().split("+"))
    unavailable_states = ("DOWN", "DRAIN", "FAIL", "NO_RESP", "FUTURE", "POWERED_DOWN")
    if any(flag.startswith(unavailable_states) for flag in state_flags):
        return False

    try:
        cpu_total = int(_field(output, "CPUTot") or "0")
        cpu_used = int(_field(output, "CPUAlloc") or "0")
        memory_total = int(_field(output, "RealMemory") or "0")
        memory_used = int(_field(output, "AllocMem") or "0")
        free_memory = int(_field(output, "FreeMem") or "0")
    except ValueError:
        return False

    configured_tres = _field(output, "CfgTRES") or ""
    allocated_tres = _field(output, "AllocTRES") or ""
    configured_gpus = _tres_value(configured_tres, f"gres/gpu:{gpu_type}")
    allocated_gpus = _tres_value(allocated_tres, f"gres/gpu:{gpu_type}")
    return (
        configured_gpus - allocated_gpus >= gpus
        and cpu_total - cpu_used >= cpus
        and memory_total - memory_used >= memory_mb
        and free_memory >= memory_mb
    )


def available_nodes(
    partition: str,
    *,
    gpu_type: str,
    gpus_per_node: int,
    cpus_per_node: int,
    memory_mb: int,
) -> list[str]:
    """List distinct nodes with the requested free resources; Slurm errors raise."""
    listed = subprocess.run(
        ["sinfo", "-N", "-h", "-p", partition, "-o", "%N"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    candidates = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    available: list[str] = []
    for node in dict.fromkeys(candidates):
        detail = subprocess.run(
            ["scontrol", "show", "node", node],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        if node_has_capacity(
            detail.stdout,
            gpu_type=gpu_type,
            gpus=gpus_per_node,
            cpus=cpus_per_node,
            memory_mb=memory_mb,
        ):
            available.append(node)
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpu-type", default="gh200")
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=48)
    parser.add_argument("--memory-mb", type=int, default=512 * 1024)
    args = parser.parse_args()
    try:
        nodes = available_nodes(
            args.partition,
            gpu_type=args.gpu_type,
            gpus_per_node=args.gpus_per_node,
            cpus_per_node=args.cpus_per_node,
            memory_mb=args.memory_mb,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Slurm capacity query failed: {exc}", file=sys.stderr)
        return 2

    if len(nodes) < args.nodes:
        print(
            f"waiting: {len(nodes)}/{args.nodes} nodes have {args.gpu_type} GPU, "
            f"{args.cpus_per_node} CPUs, and {args.memory_mb} MB free"
        )
        return 1
    print(f"available: {', '.join(nodes[: args.nodes])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
