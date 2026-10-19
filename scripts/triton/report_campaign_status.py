#!/usr/bin/env python3
"""Print a live campaign status report as Markdown (login-node safe)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNG_ORDER = {name: index for index, name in enumerate(("a0", "a1", "a2", "a3", "a4a", "a4b"))}


def command(args: list[str], *, timeout: int = 8) -> str:
    """Run a bounded, read-only status command and return stdout."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def fields(text: str) -> dict[str, str]:
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", text))


def scheduler_jobs() -> list[dict[str, str]]:
    raw = command(
        [
            "squeue",
            "-u",
            os.environ.get("USER", ""),
            "-h",
            "-o",
            "%i|%T|%j|%M|%l|%N|%R",
        ]
    )
    result = []
    for line in raw.splitlines():
        parts = line.split("|", maxsplit=6)
        if len(parts) == 7:
            result.append(
                dict(zip(("id", "state", "name", "elapsed", "limit", "nodes", "reason"), parts))
            )
    return result


def expand_nodes(node_list: str) -> set[str]:
    """Expand common Slurm node-list ranges such as gpuarm[1-2]."""
    match = re.fullmatch(r"([^,\[]*)\[([^\]]+)\]", node_list)
    if not match:
        return set(node_list.split(",")) if node_list else set()
    prefix, ranges = match.groups()
    expanded: set[str] = set()
    for item in ranges.split(","):
        bounds = item.split("-", maxsplit=1)
        if len(bounds) == 1:
            expanded.add(prefix + bounds[0])
        elif bounds[0].isdigit() and bounds[1].isdigit():
            width = max(len(bounds[0]), len(bounds[1]))
            expanded.update(
                prefix + f"{n:0{width}d}" for n in range(int(bounds[0]), int(bounds[1]) + 1)
            )
    return expanded


def find_job_id(
    explicit: str | None, manifest: dict[str, Any], nodes: list[str]
) -> tuple[str | None, str]:
    if explicit:
        return explicit, "explicit argument"
    candidates = [
        item
        for item in scheduler_jobs()
        if item["state"] == "RUNNING" and set(nodes).issubset(expand_nodes(item["nodes"]))
    ]
    if len(candidates) == 1:
        return candidates[0]["id"], "live squeue match on campaign nodes"
    if len(candidates) > 1:
        return None, "multiple live allocations match; pass --job-id"
    return str(manifest.get("job_id")) if manifest.get(
        "job_id"
    ) else None, "campaign manifest fallback"


def slurm_job(job_id: str | None) -> tuple[dict[str, str], str]:
    if not job_id:
        return {}, "not identified"
    raw = command(["scontrol", "show", "job", job_id])
    return fields(raw), ("live scontrol" if raw else "unavailable from scontrol")


def slurm_node(node: str) -> dict[str, str]:
    return fields(command(["scontrol", "show", "node", node]))


def parse_gres_count(value: str, family: str | None = None) -> int | None:
    if "gres/gpu" in value:
        pattern = (
            rf"gres/gpu(?::{re.escape(family)})?=(\d+)" if family else r"gres/gpu(?::[^,]+)?=(\d+)"
        )
    else:
        pattern = rf"gpu:{re.escape(family)}:(\d+)" if family else r"gpu(?::[^:,]+)?:(\d+)"
    match = re.search(pattern, value)
    return int(match.group(1)) if match else None


def latest_resource_sample(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("timestamp_utc")]
    except (OSError, csv.Error):
        return {}
    return rows[-1] if rows else {}


def manager_ready(campaign_dir: pathlib.Path, job_id: str | None, node: str) -> bool | None:
    if not job_id:
        return None
    logs = sorted((campaign_dir / "logs").glob(f"worker_manager_*_{job_id}.log"))
    for path in logs:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(rf"CAMPAIGN_NODE_MANAGER ready host={re.escape(node)}\b", content):
            return True
    return False


def variant_sort_key(job: dict[str, Any]) -> tuple[int, int, str]:
    name = str(job.get("variant", job.get("key", "unknown")))
    match = re.search(r"(?:^|_)((?:a[0-3])|(?:a4[ab]))(?:_|$)", name)
    rung = match.group(1) if match else "unknown"
    if "po3" in name or "ifvg" in name:
        family_order = 1
    elif "dist" in name or "distribution" in name:
        family_order = 2
    else:
        family_order = 0
    return RUNG_ORDER.get(rung, 99), family_order, name


def logged_steps(log_path: pathlib.Path) -> int | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Stable-Baselines table field; prefer the latest total over progress rows.
    matches = re.findall(r"^\|\s*total_timesteps\s*\|\s*([\d,]+)\s*\|", text, re.MULTILINE)
    if not matches:
        matches = re.findall(r"^\|\s*timesteps\s*\|\s*([\d,]+)\s*\|", text, re.MULTILINE)
    return int(matches[-1].replace(",", "")) if matches else None


def recent_fps(log_path: pathlib.Path, sample_count: int = 5) -> tuple[float, int] | None:
    """Return the median of recent Stable-Baselines FPS samples and their count."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r"^\|\s*fps\s*\|\s*([\d,]+)\s*\|", text, re.MULTILINE)
    values = [int(value.replace(",", "")) for value in matches[-sample_count:]]
    if not values:
        return None
    return float(statistics.median(values)), len(values)


def estimate_remaining_seconds(steps: int | None, target: int, fps: float | None) -> int | None:
    """Project remaining training time from the latest FPS window, if available."""
    if steps is None or target <= steps or fps is None or fps <= 0:
        return None
    return math.ceil((target - steps) / fps)


def latest_checkpoint(campaign_dir: pathlib.Path, key: str, now: dt.datetime) -> str:
    files = [
        path
        for path in (campaign_dir / "runs" / "attempts" / key).glob("**/*.zip")
        if path.is_file() and "model" in path.parts
    ]
    if not files:
        return "none found"
    path = max(files, key=lambda item: item.stat().st_mtime)
    saved = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=now.tzinfo)
    return f"{path.name} ({saved:%Y-%m-%d %H:%M})"


def job_phase(job: dict[str, Any], log_path: pathlib.Path) -> str:
    status = str(job.get("status", "unknown"))
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
    except OSError:
        text = ""
    if status == "running" and "Evaluating trained model" in text:
        return "evaluation"
    return {
        "running": "training",
        "pending": "pending",
        "failed": "failed",
        "succeeded": "completed",
    }.get(status, status)


def failure_detail(job: dict[str, Any], log_path: pathlib.Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in reversed(lines):
        marker = "campaign worker dispatch error:"
        if marker in line:
            detail = line.split(marker, maxsplit=1)[1]
            detail = re.split(r"\s+\d{4}-\d{2}-\d{2}T", detail, maxsplit=1)[0]
            return detail.strip() or str(job.get("failure_reason", "failed; details unavailable"))
    return str(job.get("failure_reason", "failed; details unavailable"))


def fmt_time(seconds: str | None) -> str:
    if not seconds:
        return "unavailable"
    try:
        value = int(seconds)
    except ValueError:
        return seconds
    return str(dt.timedelta(seconds=value))


def get_tmux_supervisor(explicit: str | None) -> str:
    if explicit:
        return explicit
    raw = command(["tmux", "list-sessions", "-F", "#{session_name}"])
    return f"not identified; tmux sessions: {raw.replace(chr(10), ', ')}" if raw else "not detected"


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def elapsed_for_job(
    started: object,
    ended: object,
    status: str,
    now: dt.datetime,
    timezone: dt.tzinfo,
) -> str:
    """Calculate the current attempt's elapsed time, ignoring stale end times."""
    if not isinstance(started, str):
        return "unavailable"
    try:
        began = dt.datetime.fromisoformat(started).astimezone(timezone)
        end_time = (
            dt.datetime.fromisoformat(ended).astimezone(timezone)
            if status != "running" and isinstance(ended, str)
            else now
        )
    except ValueError:
        return "unavailable"
    seconds = max(0, int((end_time - began).total_seconds()))
    return str(dt.timedelta(seconds=seconds))


def render(campaign_path: pathlib.Path, job_id_arg: str | None, supervisor: str | None) -> str:
    campaign_dir = campaign_path.parent
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    timezone = ZoneInfo("Europe/Helsinki")
    now = dt.datetime.now(timezone)
    nodes = [str(node) for node in campaign.get("worker_nodes", [])]
    job_id, id_source = find_job_id(job_id_arg, campaign, nodes)
    allocation, allocation_source = slurm_job(job_id)
    active_job_id = allocation.get("JobId", job_id)
    node_list = allocation.get("NodeList", ",".join(nodes))
    start = allocation.get("StartTime", "unavailable")
    end = allocation.get("EndTime", "unavailable")
    elapsed = fmt_time(allocation.get("RunTime"))
    limit = fmt_time(allocation.get("TimeLimit"))
    remaining = "unavailable"
    try:
        end_time = dt.datetime.fromisoformat(end).replace(tzinfo=timezone)
        remaining_seconds = int((end_time - now).total_seconds())
        remaining = str(dt.timedelta(seconds=max(0, remaining_seconds)))
    except (ValueError, TypeError):
        pass

    counts = {name: 0 for name in ("pending", "running", "failed", "succeeded", "interrupted")}
    for item in campaign.get("jobs", []):
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    lines = [
        f"## Campaign status — {now:%Y-%m-%d %H:%M:%S %Z}",
        "",
        "### Allocation",
        "",
        "| Campaign | Fingerprint | Slurm job | State | Nodes | Requested / granted resources | Elapsed / limit | Remaining / end | Supervisor |",
        "|---|---|---|---|---|---|---|---|---|",
        "| "
        + " | ".join(
            cell(value)
            for value in (
                campaign_dir.name,
                campaign.get("fingerprint", "unavailable"),
                f"{active_job_id or 'unavailable'} ({id_source})",
                allocation.get("JobState", "not live / unavailable"),
                node_list,
                f"requested {allocation.get('ReqTRES', 'unavailable')}; granted {allocation.get('AllocTRES', 'unavailable')}",
                f"{elapsed} / {limit}",
                f"{remaining} / {end}",
                get_tmux_supervisor(supervisor),
            )
        )
        + " |",
        "",
        f"Scheduler status: job {active_job_id or 'unavailable'} is {allocation.get('JobState', 'not live / unavailable')} on {node_list}.",
        f"Timing: started {start}; elapsed {elapsed} of {limit}; estimated remaining {remaining}; scheduled end {end}.",
        f"Campaign metadata records job_id {campaign.get('job_id', 'unavailable')} (allocation identified by {id_source}).",
        f"Variant totals: {counts.get('running', 0)} running, {counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed, {counts.get('succeeded', 0)} completed, {counts.get('interrupted', 0)} interrupted.",
        "",
        "### Node resources",
        "",
        "| Node | CPU allocated / total (free; load) | Host RAM allocated / total (Slurm unallocated) | GPU configured / allocated | Latest GPU utilization / memory sample | Manager | Active workers |",
        "|---|---|---|---|---|---|---|",
    ]
    for node in nodes:
        data = slurm_node(node)
        cpu_total = data.get("CPUTot", "?")
        cpu_alloc = data.get("CPUAlloc", "?")
        try:
            cpu_free = str(int(cpu_total) - int(cpu_alloc))
        except ValueError:
            cpu_free = "?"
        mem_total = data.get("RealMemory", "?")
        mem_alloc = data.get("AllocMem", "?")
        try:
            mem_free = str(int(mem_total) - int(mem_alloc))
        except ValueError:
            mem_free = "?"
        gres = data.get("Gres", "unavailable")
        alloc_tres = data.get("AllocTRES", "")
        gpu_type_match = re.search(r"gpu:([^:,]+)", gres)
        gpu_type = gpu_type_match.group(1) if gpu_type_match else "GPU"
        gpu_total = parse_gres_count(gres)
        gpu_alloc = parse_gres_count(alloc_tres, gpu_type) or parse_gres_count(alloc_tres)
        sample = latest_resource_sample(campaign_dir / "logs" / f"resources_{node}.csv")
        if sample:
            sampled_at = dt.datetime.fromisoformat(
                sample["timestamp_utc"].replace("Z", "+00:00")
            ).astimezone(timezone)
            age = max(0, int((now - sampled_at).total_seconds()))
            freshness = "current" if age <= 120 else "STALE"
            gpu_status = (
                f"{freshness}: {sample.get('gpu_utilization_pct', '?')}% / "
                f"{sample.get('gpu_memory_used_mib', '?')} MiB used, total unavailable; "
                f"{sampled_at:%H:%M:%S %Z}, {age}s old"
            )
        else:
            gpu_status = "no sample"
        ready = manager_ready(campaign_dir, active_job_id, node)
        readiness = "ready" if ready is True else "not ready" if ready is False else "unknown"
        active_workers = sum(
            item.get("worker_node") == node and item.get("status") == "running"
            for item in campaign.get("jobs", [])
        )
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    node,
                    f"{cpu_alloc}/{cpu_total} ({cpu_free}; load {data.get('CPULoad', 'unavailable')})",
                    f"{mem_alloc}/{mem_total} MiB ({mem_free} MiB)",
                    f"{gpu_type} {gpu_total if gpu_total is not None else '?'} / {gpu_alloc if gpu_alloc is not None else '?'}",
                    gpu_status,
                    readiness,
                    active_workers,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Variant progress (ladder order)",
            "",
            "| Variant | State / phase | Node / GPU / CPU set | Attempt | Elapsed | Latest logged steps / target | Recent FPS (median of up to 5 samples) | Estimated remaining at recent FPS | Latest checkpoint file (mtime) | Failure / recovery detail |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    target = int(campaign.get("timesteps", 0))
    for item in sorted(campaign.get("jobs", []), key=variant_sort_key):
        status = str(item.get("status", "unknown"))
        log_path = pathlib.Path(str(item.get("log", "")))
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        steps = logged_steps(log_path)
        progress = f"{steps:,} / {target:,}" if steps is not None and target else "not logged"
        fps_sample = recent_fps(log_path)
        fps = f"{fps_sample[0]:g} (n={fps_sample[1]})" if fps_sample else "not logged"
        estimate_seconds = estimate_remaining_seconds(
            steps, target, fps_sample[0] if fps_sample else None
        )
        estimate = (
            fmt_time(str(estimate_seconds)) if estimate_seconds is not None else "unavailable"
        )
        elapsed_variant = elapsed_for_job(
            item.get("started"), item.get("ended"), status, now, timezone
        )
        node = item.get("worker_node") or "unassigned"
        gpu = item.get("worker_gpu_id")
        cpus = item.get("worker_cpu_ids")
        assignment = f"{node} / GPU {gpu if gpu is not None else '?'} / CPU {','.join(map(str, cpus)) if cpus else '?'}"
        detail = failure_detail(item, log_path) if status == "failed" else ""
        if not detail and status == "running" and job_phase(item, log_path) == "evaluation":
            detail = "training target reached; evaluation in progress"
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    item.get("variant", item.get("key", "unknown")),
                    f"{status} / {job_phase(item, log_path)}",
                    assignment,
                    item.get("attempts", "?"),
                    elapsed_variant,
                    progress,
                    fps,
                    estimate,
                    latest_checkpoint(campaign_dir, str(item.get("key", "")), now),
                    detail or "—",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Estimated remaining time uses the median of up to five recent FPS samples; it is indicative and excludes post-training evaluation.",
            "",
            "Checkpoint files are reported by filename and modification time only; this report does not claim that a checkpoint passed integrity checks or evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=pathlib.Path, help="campaign.json or campaign directory")
    parser.add_argument("--job-id", help="Slurm allocation ID when automatic matching is ambiguous")
    parser.add_argument("--supervisor", help="persistent tmux supervisor/session name")
    args = parser.parse_args()
    path = args.campaign / "campaign.json" if args.campaign.is_dir() else args.campaign
    try:
        print(render(path.resolve(), args.job_id, args.supervisor))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"campaign status report failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
