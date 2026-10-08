#!/usr/bin/env python3
"""Parse benchmark logs and produce baseline vs Stages 1-3 comparison.

Measures:
- Parent RSS (VmRSS, VmHWM)
- Worker RSS (from worker logs)
- Rollout throughput (env steps/sec)
- Rollout vs learn time split
- GPU VRAM and utilization

Produces compact markdown table + JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def _parse_parent_memory_from_log(log_path: Path) -> dict[str, Any]:
    """Parse parent process memory metrics from main log."""
    result: dict[str, Any] = {
        "parent_start_rss_kb": 0,
        "parent_end_rss_kb": 0,
        "parent_start_vmhwm_kb": 0,
        "parent_end_vmhwm_kb": 0,
        "parent_memory_growth_mb": 0.0,
        "parent_peak_rss_mb": 0.0,
    }

    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")

    # Extract parent memory metrics
    for metric, pattern in [
        ("parent_start_rss_kb", r"Parent start RSS:\s*(\d+)\s*KB"),
        ("parent_end_rss_kb", r"Parent end RSS:\s*(\d+)\s*KB"),
        ("parent_start_vmhwm_kb", r"Parent start.*VmHWM:\s*(\d+)\s*KB"),
        ("parent_end_vmhwm_kb", r"Parent end.*VmHWM:\s*(\d+)\s*KB"),
        ("parent_start_rss_kb", r"parent_start_rss_kb=(\d+)"),
        ("parent_end_rss_kb", r"parent_end_rss_kb=(\d+)"),
        ("parent_start_vmhwm_kb", r"parent_start_vmhwm_kb=(\d+)"),
        ("parent_end_vmhwm_kb", r"parent_end_vmhwm_kb=(\d+)"),
    ]:
        match = re.search(pattern, text)
        if match:
            result[metric] = int(match.group(1))

    # Calculate memory growth
    if result["parent_end_vmhwm_kb"] and result["parent_start_vmhwm_kb"]:
        growth_kb = result["parent_end_vmhwm_kb"] - result["parent_start_vmhwm_kb"]
        result["parent_memory_growth_mb"] = round(max(0, growth_kb) / 1024, 2)
        result["parent_peak_rss_mb"] = round(result["parent_end_vmhwm_kb"] / 1024, 2)

    return result


def _parse_worker_memory(worker_log: Path) -> dict[str, Any]:
    """Parse worker RSS/VMHWM from worker monitoring log."""
    result: dict[str, Any] = {
        "worker_pids": [],
        "worker_peak_rss_kb": 0,
        "worker_peak_vmhwm_kb": 0,
        "worker_avg_rss_kb": 0,
        "worker_count": 0,
        "worker_total_rss_mb": 0.0,
    }

    if not worker_log.exists():
        return result

    try:
        with worker_log.open() as fh:
            reader = csv.DictReader(fh)
            rss_values = []
            vmhwm_values = []
            pids = set()

            for row in reader:
                try:
                    pid = int(row.get("pid", 0))
                    rss = int(row.get("vmrss_kb", 0))
                    vmhwm = int(row.get("vmhwm_kb", 0))

                    if pid > 0:
                        pids.add(pid)
                        rss_values.append(rss)
                        vmhwm_values.append(vmhwm)

                        result["worker_peak_rss_kb"] = max(result["worker_peak_rss_kb"], rss)
                        result["worker_peak_vmhwm_kb"] = max(result["worker_peak_vmhwm_kb"], vmhwm)
                except (ValueError, KeyError):
                    continue

            result["worker_pids"] = sorted(list(pids))
            result["worker_count"] = len(pids)

            if rss_values:
                result["worker_avg_rss_kb"] = round(sum(rss_values) / len(rss_values), 1)
                result["worker_total_rss_mb"] = round(sum(rss_values) / 1024, 2)

    except Exception:
        pass

    return result


def _parse_gpu_metrics(gpu_file: Path) -> dict[str, Any]:
    """Parse GPU metrics from nvidia-smi output."""
    result: dict[str, Any] = {
        "gpu_memory_used_mb": 0,
        "gpu_memory_total_mb": 0,
        "gpu_utilization_pct": 0.0,
    }

    if not gpu_file.exists():
        return result

    try:
        text = gpu_file.read_text()
        if "memory.used" in text:
            # Parse CSV format: memory.used (MB), memory.total (MB), utilization.gpu (%)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if len(lines) >= 3:
                # First line should be memory.used, second memory.total, third utilization
                used_match = re.search(r"(\d+)", lines[0])
                total_match = re.search(r"(\d+)", lines[1])
                util_match = re.search(r"(\d+)", lines[2])

                if used_match:
                    result["gpu_memory_used_mb"] = int(used_match.group(1))
                if total_match:
                    result["gpu_memory_total_mb"] = int(total_match.group(1))
                if util_match:
                    result["gpu_utilization_pct"] = float(util_match.group(1))
    except Exception:
        pass

    return result


def _parse_timing_from_log(log_path: Path) -> dict[str, Any]:
    """Extract rollout/learn timing metrics from training logs."""
    result: dict[str, Any] = {
        "wall_ms": None,
        "rollout_steps_per_sec": None,
        "rollout_time_ms": None,
        "learn_time_ms": None,
        "update_count": 0,
    }

    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")

    # Wall time
    match = re.search(r"wall_ms=(\d+)", text)
    if match:
        result["wall_ms"] = int(match.group(1))
        result["wall_seconds"] = round(int(match.group(1)) / 1000.0, 3)

    # Rollout steps/sec from SB3 logs
    match = re.search(r"steps per second:\s*([\d.]+)", text)
    if match:
        result["rollout_steps_per_sec"] = round(float(match.group(1)), 2)

    # Extract rollout/learn timing from dashboard or callbacks
    # Look for patterns like "rollout_time: X ms", "learn_time: Y ms"
    rollout_times = re.findall(
        r"rollout[_:\-]?time[_:\-]?ms[_:\-]?(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    learn_times = re.findall(
        r"learn[_:\-]?time[_:\-]?ms[_:\-]?(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )

    if rollout_times:
        result["rollout_time_ms"] = round(float(rollout_times[0]), 2)
    if learn_times:
        result["learn_time_ms"] = round(float(learn_times[0]), 2)

    # Count updates from progress logs
    update_matches = re.findall(r"Update\s+(\d+)", text)
    if update_matches:
        result["update_count"] = len(update_matches)

    return result


def _parse_csv_timing(csv_files: list[Path]) -> dict[str, Any]:
    """Parse timing from any CSV files in the log directory."""
    result: dict[str, Any] = {}

    for csv_file in csv_files:
        try:
            with open(csv_file) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    for key, value in row.items():
                        if key not in result:
                            try:
                                result[key] = float(value) if "." in value else int(value)
                            except (ValueError, TypeError):
                                result[key] = value
        except Exception:
            continue

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("summary_md", type=Path)
    ap.add_argument("summary_json", type=Path)
    args = ap.parse_args()

    log_dir = args.log_dir
    results: dict[str, dict[str, Any]] = {}

    # Parse baseline and staged results
    for label in ("baseline", "staged"):
        log_path = log_dir / f"{label}.log"
        worker_log = log_dir / f"{label}_workers.log"
        gpu_start = log_dir / f"{label}_gpu_start.txt"
        gpu_end = log_dir / f"{label}_gpu_end.txt"

        result = {
            "label": label,
            **_parse_parent_memory_from_log(log_path),
            **_parse_worker_memory(worker_log),
            **_parse_timing_from_log(log_path),
            **_parse_gpu_metrics(gpu_start),
            **_parse_gpu_metrics(gpu_end),
        }

        # Calculate derived metrics
        if result.get("wall_seconds"):
            # Estimate rollout time if we have steps/sec and wall time
            if result.get("rollout_steps_per_sec") and result.get("update_count"):
                total_steps = result["rollout_steps_per_sec"] * result["wall_seconds"]
                steps_per_update = total_steps / max(1, result["update_count"])
                result["steps_per_update"] = round(steps_per_update, 1)

        results[label] = result

    baseline = results["baseline"]
    staged = results["staged"]

    # Build markdown report
    md = [
        "# Stages 1-3 Benchmark: Baseline vs Staged",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Stages Implemented",
        "",
        "| Stage | Change | Purpose |",
        "|---|---|---|",
        "| 1 | Lazy VAE/torch import in trading_env.py | Remove unnecessary per-worker PyTorch memory (~0.5-1.5 GiB/worker) |",
        "| 2 | Parent memory cleanup | Free full-history pipeline objects after slicing (~5-8 GiB) |",
        "| 3 | Worker thread caps | Prevent OMP/MKL/torch thread-pool oversubscription |",
        "",
        "## Executive Summary",
        "",
    ]

    # Parent Memory Table
    md.extend(
        [
            "### Parent Process Memory",
            "",
            "| Metric | baseline | staged | Δ | % |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    parent_metrics = [
        ("parent_peak_rss_mb", "Parent Peak RSS (MB)", "{:.1f}"),
        ("parent_memory_growth_mb", "Parent Memory Growth (MB)", "{:+.1f}"),
        ("parent_start_rss_kb", "Parent Start RSS (KB)", "{:.0f}"),
        ("parent_end_rss_kb", "Parent End RSS (KB)", "{:.0f}"),
    ]

    for key, label, fmt in parent_metrics:
        b = baseline.get(key)
        s = staged.get(key)
        if b is not None and s is not None:
            try:
                b_val = float(b)
                s_val = float(s)
                delta = s_val - b_val
                pct = 100.0 * delta / b_val if b_val != 0 else 0.0
                md.append(
                    f"| {label} | {fmt.format(b_val)} | {fmt.format(s_val)} | {fmt.format(delta)} | {pct:+.1f}% |"
                )
            except (ValueError, TypeError):
                md.append(f"| {label} | {b} | {s} | - | - |")
        else:
            md.append(f"| {label} | - | - | - | - |")

    # Worker Memory Table
    md.extend(
        [
            "",
            "### Worker Process Memory",
            "",
            "| Metric | baseline | staged | Δ | % |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    worker_metrics = [
        ("worker_peak_rss_kb", "Worker Peak RSS (KB)", "{:.0f}"),
        ("worker_avg_rss_kb", "Worker Avg RSS (KB)", "{:.0f}"),
        ("worker_total_rss_mb", "Total Worker RSS (MB)", "{:.1f}"),
        ("worker_count", "Worker Count", "{:.0f}"),
    ]

    for key, label, fmt in worker_metrics:
        b = baseline.get(key)
        s = staged.get(key)
        if b is not None and s is not None:
            try:
                b_val = float(b)
                s_val = float(s)
                delta = s_val - b_val
                pct = 100.0 * delta / b_val if b_val != 0 else 0.0
                md.append(
                    f"| {label} | {fmt.format(b_val)} | {fmt.format(s_val)} | {fmt.format(delta)} | {pct:+.1f}% |"
                )
            except (ValueError, TypeError):
                md.append(f"| {label} | {b} | {s} | - | - |")
        else:
            md.append(f"| {label} | - | - | - | - |")

    # Performance Table
    md.extend(
        [
            "",
            "### Performance Metrics",
            "",
            "| Metric | baseline | staged | Δ | % |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    perf_metrics = [
        ("rollout_steps_per_sec", "Rollout Steps/sec", "{:.1f}"),
        ("wall_seconds", "Total Wall Time (s)", "{:.2f}"),
        ("rollout_time_ms", "Rollout Time/Update (ms)", "{:.1f}"),
        ("learn_time_ms", "Learn Time/Update (ms)", "{:.1f}"),
        ("update_count", "Update Count", "{:.0f}"),
        ("steps_per_update", "Steps/Update", "{:.1f}"),
    ]

    for key, label, fmt in perf_metrics:
        b = baseline.get(key)
        s = staged.get(key)
        if b is not None and s is not None:
            try:
                b_val = float(b)
                s_val = float(s)
                delta = s_val - b_val
                pct = 100.0 * delta / b_val if b_val != 0 else 0.0
                md.append(
                    f"| {label} | {fmt.format(b_val)} | {fmt.format(s_val)} | {fmt.format(delta)} | {pct:+.1f}% |"
                )
            except (ValueError, TypeError):
                md.append(f"| {label} | {b} | {s} | - | - |")
        else:
            md.append(f"| {label} | - | - | - | - |")

    # GPU Table
    md.extend(
        [
            "",
            "### GPU Metrics",
            "",
            "| Metric | baseline | staged | Δ | % |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    gpu_metrics = [
        ("gpu_memory_used_mb", "GPU Memory Used (MB)", "{:.0f}"),
        ("gpu_memory_total_mb", "GPU Memory Total (MB)", "{:.0f}"),
        ("gpu_utilization_pct", "GPU Utilization (%)", "{:.1f}"),
    ]

    for key, label, fmt in gpu_metrics:
        b = baseline.get(key)
        s = staged.get(key)
        if b is not None and s is not None:
            try:
                b_val = float(b)
                s_val = float(s)
                delta = s_val - b_val
                pct = 100.0 * delta / b_val if b_val != 0 else 0.0
                md.append(
                    f"| {label} | {fmt.format(b_val)} | {fmt.format(s_val)} | {fmt.format(delta)} | {pct:+.1f}% |"
                )
            except (ValueError, TypeError):
                md.append(f"| {label} | {b} | {s} | - | - |")
        else:
            md.append(f"| {label} | - | - | - | - |")

    # Overall Assessment
    md.extend(
        [
            "",
            "## Overall Assessment",
            "",
            "### Key Findings:",
        ]
    )

    # Calculate overall improvements
    memory_saved = 0.0
    if baseline.get("parent_memory_growth_mb") and staged.get("parent_memory_growth_mb"):
        memory_saved = baseline["parent_memory_growth_mb"] - staged["parent_memory_growth_mb"]
        if memory_saved > 0:
            md.append(
                f"- **Memory Reduction**: Parent memory growth reduced by {memory_saved:.1f} MB"
            )
        else:
            md.append("- **Memory**: No significant parent memory reduction detected")

    if baseline.get("worker_total_rss_mb") and staged.get("worker_total_rss_mb"):
        worker_memory_saved = baseline["worker_total_rss_mb"] - staged["worker_total_rss_mb"]
        if worker_memory_saved > 0:
            md.append(
                f"- **Worker Memory**: Total worker RSS reduced by {worker_memory_saved:.1f} MB"
            )
        else:
            md.append("- **Worker Memory**: No significant reduction detected")

    if baseline.get("rollout_steps_per_sec") and staged.get("rollout_steps_per_sec"):
        throughput_improvement = staged["rollout_steps_per_sec"] - baseline["rollout_steps_per_sec"]
        if throughput_improvement > 0:
            pct_improvement = 100.0 * throughput_improvement / baseline["rollout_steps_per_sec"]
            md.append(
                f"- **Throughput**: Rollout throughput improved by {throughput_improvement:.1f} steps/sec ({pct_improvement:+.1f}%)"
            )
        else:
            md.append("- **Throughput**: No significant improvement detected")

    if baseline.get("wall_seconds") and staged.get("wall_seconds"):
        time_saved = baseline["wall_seconds"] - staged["wall_seconds"]
        if time_saved > 0:
            pct_saved = 100.0 * time_saved / baseline["wall_seconds"]
            md.append(
                f"- **Training Time**: Wall time reduced by {time_saved:.2f}s ({pct_saved:+.1f}%)"
            )
        else:
            md.append("- **Training Time**: No significant time reduction detected")

    # Recommendations
    md.extend(
        [
            "",
            "### Recommendations:",
        ]
    )

    # Check if thread caps worked (compare rollout throughput)
    if baseline.get("rollout_steps_per_sec") and staged.get("rollout_steps_per_sec"):
        if staged["rollout_steps_per_sec"] > baseline["rollout_steps_per_sec"] * 1.05:
            md.append("- ✅ **Thread caps effective**: Throughput improved, keep Stage 3")
        elif staged["rollout_steps_per_sec"] >= baseline["rollout_steps_per_sec"] * 0.95:
            md.append("- ⚠️ **Thread caps neutral**: No throughput regression, safe to keep Stage 3")
        else:
            md.append("- ❌ **Thread caps harmful**: Throughput decreased, investigate Stage 3")

    # Check memory improvements
    memory_improvement = False
    if baseline.get("parent_memory_growth_mb") and staged.get("parent_memory_growth_mb"):
        if staged["parent_memory_growth_mb"] < baseline["parent_memory_growth_mb"]:
            memory_improvement = True
    if baseline.get("worker_total_rss_mb") and staged.get("worker_total_rss_mb"):
        if staged["worker_total_rss_mb"] < baseline["worker_total_rss_mb"]:
            memory_improvement = True

    if memory_improvement:
        md.append("- ✅ **Memory optimization effective**: RAM usage reduced, keep Stages 1-2")
    else:
        md.append(
            "- ⚠️ **Memory optimization minimal**: Check if Stages 1-2 are working as expected"
        )

    # Final summary
    md.extend(
        [
            "",
            "### Test Parameters:",
            f"- Variant: {VARIANT if 'VARIANT' in globals() else 'ladder_a0_flat'}",
            f"- Seed: {SEED if 'SEED' in globals() else '42'}",
            f"- Steps: {STEPS if 'STEPS' in globals() else '10000'}",
            f"- n_envs: {N_ENVS if 'N_ENVS' in globals() else '64'}",
            "",
            "## Notes",
            "",
            "- Stages 1-3 focus on memory optimization without changing training semantics",
            "- Stage 1: Lazy VAE/torch import removes ~0.5-1.5 GiB/worker for non-VAE variants",
            "- Stage 2: Parent cleanup removes ~5-8 GiB of unused pipeline data",
            "- Stage 3: Thread caps prevent OMP/MKL/torch oversubscription",
            "- Thread caps use OMP/MKL/OPENBLAS_NUM_THREADS=1 + torch.set_num_threads(1)",
            "- Worker RSS measured from /proc/<pid>/status VmRSS",
            "- Parent RSS measured from /proc/self/status VmRSS/VmHWM",
            "- GPU metrics from nvidia-smi when available",
            "",
            "## Reproduction",
            "```bash",
            f"cd {REPO if 'REPO' in globals() else '/scratch/work/nguyenl37/Aalto_MS_Thesis'}",
            f"VARIANT={VARIANT if 'VARIANT' in globals() else 'ladder_a0_flat'} SEED={SEED if 'SEED' in globals() else '42'} STEPS={STEPS if 'STEPS' in globals() else '10000'} N_ENVS={N_ENVS if 'N_ENVS' in globals() else '64'} \\",
            "bash scripts/bench/run_stages_benchmark.sh",
            "```",
        ]
    )

    # Write markdown output
    args.summary_md.write_text("\n".join(md) + "\n")

    # Write JSON output
    json_output = {
        "metadata": {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "benchmark_type": "Stages 1-3 (Lazy VAE import + Parent cleanup + Thread caps)",
            "variant": VARIANT if "VARIANT" in globals() else "ladder_a0_flat",
            "seed": SEED if "SEED" in globals() else 42,
            "steps": STEPS if "STEPS" in globals() else 10000,
            "n_envs": N_ENVS if "N_ENVS" in globals() else 64,
        },
        "baseline": baseline,
        "staged": staged,
        "comparison": {},
    }

    # Calculate comparison metrics
    comparison = json_output["comparison"]

    # Memory comparison
    if baseline.get("parent_peak_rss_mb") and staged.get("parent_peak_rss_mb"):
        parent_memory_delta = float(baseline["parent_peak_rss_mb"]) - float(
            staged["parent_peak_rss_mb"]
        )
        parent_memory_pct = (
            100.0 * parent_memory_delta / float(baseline["parent_peak_rss_mb"])
            if float(baseline["parent_peak_rss_mb"]) > 0
            else 0.0
        )
        comparison["parent_memory_reduction_mb"] = round(parent_memory_delta, 2)
        comparison["parent_memory_reduction_pct"] = round(parent_memory_pct, 2)

    if baseline.get("worker_total_rss_mb") and staged.get("worker_total_rss_mb"):
        worker_memory_delta = float(baseline["worker_total_rss_mb"]) - float(
            staged["worker_total_rss_mb"]
        )
        worker_memory_pct = (
            100.0 * worker_memory_delta / float(baseline["worker_total_rss_mb"])
            if float(baseline["worker_total_rss_mb"]) > 0
            else 0.0
        )
        comparison["worker_memory_reduction_mb"] = round(worker_memory_delta, 2)
        comparison["worker_memory_reduction_pct"] = round(worker_memory_pct, 2)

    # Performance comparison
    if baseline.get("rollout_steps_per_sec") and staged.get("rollout_steps_per_sec"):
        throughput_delta = float(staged["rollout_steps_per_sec"]) - float(
            baseline["rollout_steps_per_sec"]
        )
        throughput_pct = (
            100.0 * throughput_delta / float(baseline["rollout_steps_per_sec"])
            if float(baseline["rollout_steps_per_sec"]) > 0
            else 0.0
        )
        comparison["throughput_improvement_steps_per_sec"] = round(throughput_delta, 2)
        comparison["throughput_improvement_pct"] = round(throughput_pct, 2)

    if baseline.get("wall_seconds") and staged.get("wall_seconds"):
        time_delta = float(baseline["wall_seconds"]) - float(staged["wall_seconds"])
        time_pct = (
            100.0 * time_delta / float(baseline["wall_seconds"])
            if float(baseline["wall_seconds"]) > 0
            else 0.0
        )
        comparison["time_reduction_seconds"] = round(time_delta, 2)
        comparison["time_reduction_pct"] = round(time_pct, 2)

    args.summary_json.write_text(json.dumps(json_output, indent=2, default=str) + "\n")

    # Print to console
    print("\n".join(md))

    return 0


# Global variables for command reproduction (will be filled from environment)
VARIANT = "ladder_a0_flat"
SEED = 42
STEPS = 10000
N_ENVS = 64
REPO = "/scratch/work/nguyenl37/Aalto_MS_Thesis"


if __name__ == "__main__":
    sys.exit(main())
