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
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, TextIO, cast

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ROOT / "scripts" / "variant_list.txt"
STOP = False


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


def available_resources() -> dict[str, int]:
    memory = pathlib.Path("/sys/fs/cgroup/memory.max")
    mem_limit = (
        int(memory.read_text().strip())
        if memory.exists() and memory.read_text().strip().isdigit()
        else 0
    )
    cpu_cores = len(os.sched_getaffinity(0))
    return {"memory_limit_bytes": mem_limit, "cpu_affinity": cpu_cores}


def sample_resources(path: pathlib.Path) -> None:
    def read(name: str) -> str:
        p = pathlib.Path("/sys/fs/cgroup") / name
        return p.read_text().strip().replace("\n", ";") if p.exists() else ""

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
                read("memory.peak"),
                read("memory.events"),
                gpu_util,
                gpu_memory,
            ]
        )
        f.flush()


def verify_run_artifacts(run_out: pathlib.Path, key: str) -> tuple[bool, str]:
    """Require a completed model and training record."""
    for path in (run_out / key).glob("**/training_log.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("timesteps_completed")
            and (path.parent / "model").is_dir()
        ):
            return True, ""
    return False, "missing completed training_log.json or model directory"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-file", type=pathlib.Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", default="42 43 44 45 46")
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--data-source", choices=("frames", "bundle"), default="frames")
    parser.add_argument("--campaign-id", default="core-18x5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--test-worker", type=pathlib.Path, help="inject mock worker for CI only")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.campaign_id):
        raise ValueError("unsafe campaign id")
    jobs = validate_inputs(args)
    # Source-of-truth variant lookup without changing training semantics.
    import yaml

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
                    "runs": len(jobs),
                    "total_timesteps": len(jobs) * args.timesteps,
                    "workers": args.workers,
                    "n_envs": args.n_envs,
                    "data_source": args.data_source,
                    "keys": [j["key"] for j in jobs],
                },
                indent=2,
            )
        )
        return 0
    if not os.getenv("SLURM_JOB_ID") and not args.test_worker:
        raise RuntimeError("must run inside an existing Slurm compute allocation")
    if args.workers * args.n_envs > available_resources()["cpu_affinity"] * 2:
        raise RuntimeError("requested vector environments exceed twice available CPUs")
    available_mem = available_resources()["memory_limit_bytes"]
    # Conservative frame-path estimates from Triton PSS measurements.
    estimated_per_worker = {8: 19, 16: 35, 32: 67, 64: 132}.get(args.n_envs)
    if estimated_per_worker is None and not args.test_worker:
        raise ValueError("n-envs not benchmarked for memory: choose 8,16,32,64")
    if available_mem and estimated_per_worker and not args.test_worker:
        recommended_bytes = int((args.workers * estimated_per_worker * 1.35 + 64) * 1024**3)
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
    manifest_path = out / "campaign.json"
    started_time = time.monotonic()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {"jobs": [x["key"] for x in jobs], "steps": args.timesteps}
    # Preserve the fingerprint of campaigns created before bundle opt-in.
    if args.data_source != "frames":
        identity["data_source"] = args.data_source
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        if not args.resume:
            raise RuntimeError("campaign exists: use --resume")
        if manifest["fingerprint"] != fingerprint:
            raise RuntimeError("resume fingerprint mismatch")
        if manifest.get("data_source", "frames") != args.data_source:
            raise RuntimeError("resume data-source mismatch")
        if manifest.get("n_envs") != args.n_envs:
            raise RuntimeError("resume n-envs mismatch")
        jobs = manifest["jobs"]
        for item in jobs:
            if (
                item["status"] == "succeeded"
                and not (out / "markers" / f"{item['key']}.done").is_file()
            ):
                raise RuntimeError(f"completed marker missing for {item['key']}")
            if item["status"] in ("running", "interrupted", "failed"):
                item["status"] = "pending"
    else:
        manifest = {
            "created": utc(),
            "hostname": socket.gethostname(),
            "job_id": os.getenv("SLURM_JOB_ID", "test"),
            "git_sha": git_sha,
            "fingerprint": fingerprint,
            "timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "data_source": args.data_source,
            "workers": args.workers,
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
        f"campaign={args.campaign_id} runs={len(jobs)} workers={args.workers} host={socket.gethostname()}",
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

    def start(job: dict[str, Any]) -> None:
        key = job["key"]
        job["attempts"] += 1
        attempt = job["attempts"]
        logpath = out / "logs" / "workers" / f"{key}__attempt{attempt}.log"
        run_out = out / "runs"
        cmd = (
            [str(args.test_worker), key]
            if args.test_worker
            else ["bash", str(ROOT / "scripts" / "train" / "train_one_variant.sh")]
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
            PYTHONUNBUFFERED="1",
        )
        handle = logpath.open("a", encoding="utf-8")
        handle.write(
            f"{utc()} START key={key} attempt={attempt} host={socket.gethostname()} slurm={os.getenv('SLURM_JOB_ID')} git={git_sha} cmd={cmd}\n"
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
                "log": str(logpath),
            },
        )
        record(scheduler, {"action": "start", "key": key, "attempt": attempt})
        snapshot()

    last_sample = 0.0
    while running or (not STOP and any(j["status"] == "pending" for j in jobs)):
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
            verified, reason = (
                (True, "") if args.test_worker else verify_run_artifacts(out / "runs", key)
            )
            success = ret == 0 and verified
            job.update(
                status="succeeded" if success else "failed",
                exit_code=ret,
                ended=utc(),
                artifact_error=reason if ret == 0 else "",
            )
            if success:
                (out / "markers" / f"{key}.done").touch()
            else:
                record(
                    out / "logs" / "failures" / "summary.jsonl",
                    {"key": key, "attempt": job["attempts"], "exit_code": ret, "log": job["log"]},
                )
            record(
                events,
                {"event": job["status"], "key": key, "attempt": job["attempts"], "exit_code": ret},
            )
            del running[key]
            snapshot()
        if time.monotonic() - last_sample >= 30:
            sample_resources(out / "logs" / "resources.csv")
            last_sample = time.monotonic()
        if running:
            time.sleep(1)
    if STOP and running:
        for proc, _, _ in running.values():
            os.killpg(proc.pid, signal.SIGTERM)
        for proc, handle, job in running.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            job.update(status="interrupted", exit_code=proc.returncode, ended=utc())
            handle.close()
            record(
                events, {"event": "interrupted", "key": job["key"], "exit_code": proc.returncode}
            )
    snapshot()
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
    return 1 if STOP or any(j["status"] != "succeeded" for j in jobs) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
