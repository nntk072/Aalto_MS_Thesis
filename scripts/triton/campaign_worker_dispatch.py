#!/usr/bin/env python3
"""Dispatch campaign worker commands through one persistent process per node."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ENV_KEYS = (
    "VARIANT",
    "SEED",
    "SEEDS",
    "STEPS",
    "GPU_ID",
    "OUT_DIR",
    "LOG_DIR",
    "VAE_PATH",
    "DRY_RUN",
    "QUANT_RL_MAX_N_ENVS",
    "QUANT_RL_DATA_SOURCE",
    "QUANT_RL_FAST_REPORTS",
    "QUANT_RL_ALLOW_OOS_SELECTION",
    "CAMPAIGN_WORKER_PID_FILE",
    "RESUME_CHECKPOINT",
    "PYTHONUNBUFFERED",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "TRITON_REPO",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def _host_key(value: str) -> str:
    return value.split(".", maxsplit=1)[0]


def _parse_srun(argv: list[str]) -> dict[str, Any]:
    """Parse only the fixed worker_command() options used by the campaign."""
    node: str | None = None
    cwd: str | None = None
    gpu_count = 0
    cpu_mask: int | None = None
    node_count: int | None = None
    task_count: int | None = None
    step_cpus: int | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if not arg.startswith("-"):
            break
        if arg == "--overlap":
            pass
        elif arg.startswith("--nodelist="):
            node = arg.split("=", 1)[1]
        elif arg.startswith("--gpus="):
            gpu_count = int(arg.rsplit(":", 1)[1])
        elif arg.startswith("--gpu-bind="):
            if arg != "--gpu-bind=map_gpu:0":
                raise ValueError(f"unsupported campaign GPU binding: {arg}")
        elif arg.startswith("--cpu-bind=mask_cpu:"):
            cpu_mask = int(arg.split(":", 1)[1], 0)
        elif arg.startswith("--cpu-bind="):
            raise ValueError(f"unsupported campaign CPU binding: {arg}")
        elif arg.startswith("--cpus-per-task="):
            step_cpus = int(arg.split("=", 1)[1])
        elif arg.startswith("--nodes="):
            node_count = int(arg.split("=", 1)[1])
        elif arg.startswith("--ntasks="):
            task_count = int(arg.split("=", 1)[1])
        elif arg == "--chdir":
            index += 1
            if index >= len(argv):
                raise ValueError("--chdir is missing its path")
            cwd = argv[index]
        elif arg.startswith("--export="):
            pass
        else:
            raise ValueError(f"unsupported campaign srun option: {arg}")
        index += 1

    command = argv[index:]
    if (
        not node
        or not cwd
        or not command
        or cpu_mask is None
        or node_count != 1
        or task_count != 1
        or step_cpus is None
    ):
        raise ValueError("campaign worker command is missing node, cwd, CPU mask, or argv")
    cpus = [cpu for cpu in range(cpu_mask.bit_length()) if cpu_mask & (1 << cpu)]
    if not cpus:
        raise ValueError("campaign worker CPU mask is empty")
    if step_cpus < len(cpus):
        raise ValueError("campaign worker CPU mask exceeds its Slurm CPU envelope")
    if gpu_count not in (0, 1):
        raise ValueError(f"campaign node allocation exposes one GPU per node, got {gpu_count}")
    return {
        "node": _host_key(node),
        "cwd": cwd,
        "argv": command,
        "cpu_ids": cpus,
        "gpu_count": gpu_count,
    }


def _paths(queue: Path, node: str) -> tuple[Path, Path, Path, Path]:
    root = queue / str(os.environ.get("SLURM_JOB_ID", "unknown"))
    host = root / _host_key(node)
    return host / "pending", host / "claimed", root / "results", host / "ready"


def _request() -> int:
    queue_value = os.environ.get("CAMPAIGN_WORKER_QUEUE")
    if not queue_value:
        raise RuntimeError("CAMPAIGN_WORKER_QUEUE is not set")
    arguments = sys.argv[1:]
    if arguments[:1] == ["--srun-shim"]:
        arguments = arguments[1:]
    request = _parse_srun(arguments)
    queue = Path(queue_value)
    pending, _claimed, results, ready = _paths(queue, request["node"])
    deadline = time.monotonic() + 60
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"persistent worker manager did not become ready on {request['node']}"
            )
        time.sleep(0.2)

    request_id = uuid.uuid4().hex
    cancel_dir = pending.parent / "cancelled"
    cancel_path = cancel_dir / f"{request_id}.cancel"

    def cancel_request(signum: int, _frame: object) -> None:
        cancel_dir.mkdir(parents=True, exist_ok=True)
        cancel_path.touch()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancel_request)
    signal.signal(signal.SIGINT, cancel_request)
    request_dir = pending
    request_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"{request_id}.json"
    payload = {
        **request,
        "request_id": request_id,
        "env": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "stdout_path": os.readlink("/proc/self/fd/1"),
    }
    temporary = request_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, request_path)

    result_path = results / f"{request_id}.json"
    while not result_path.is_file():
        time.sleep(1)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return int(result["returncode"])


def _log_path(env: dict[str, str], cwd: str, argv: list[str]) -> Path:
    out_dir = env.get("OUT_DIR")
    log_dir = env.get("LOG_DIR")
    if out_dir and log_dir and Path(out_dir).name.startswith("attempt"):
        attempt_dir = Path(out_dir)
        return Path(log_dir) / f"{attempt_dir.parent.name}__{attempt_dir.name}.log"
    for index, arg in enumerate(argv[:-1]):
        if arg == "--out":
            return Path(argv[index + 1])
    return Path(cwd) / "campaign_worker_dispatch.log"


def _write_result(results: Path, request_id: str, returncode: int) -> None:
    result_path = results / f"{request_id}.json"
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"returncode": returncode}), encoding="utf-8")
    os.replace(temporary, result_path)


def _launch(payload: dict[str, Any], results: Path) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.pop("RESUME_CHECKPOINT", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(payload["env"])
    env["SLURM_JOB_ID"] = str(payload.get("job_id", ""))
    if payload["gpu_count"]:
        allocated_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not allocated_devices:
            raise RuntimeError("GPU worker manager has no Slurm CUDA_VISIBLE_DEVICES binding")
        env["CUDA_VISIBLE_DEVICES"] = allocated_devices
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    cpu_ids = {int(value) for value in payload["cpu_ids"]}

    def apply_affinity() -> None:
        os.sched_setaffinity(0, cpu_ids)

    stdout_path = str(payload.get("stdout_path", ""))
    if stdout_path.startswith("/"):
        log_path = Path(stdout_path)
    else:
        log_path = _log_path(env, payload["cwd"], payload["argv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            payload["argv"],
            cwd=payload["cwd"],
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=apply_affinity,
        )
    finally:
        log_stream.close()
    process._campaign_log_path = str(log_path)  # type: ignore[attr-defined]
    process._campaign_request_id = str(payload["request_id"])  # type: ignore[attr-defined]
    return process


def _manager() -> int:
    queue_value = os.environ.get("CAMPAIGN_WORKER_QUEUE")
    node_value = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not queue_value or not job_id:
        raise RuntimeError("worker manager requires CAMPAIGN_WORKER_QUEUE and SLURM_JOB_ID")
    queue = Path(queue_value)
    pending, claimed, results, ready = _paths(queue, node_value)
    pending.mkdir(parents=True, exist_ok=True)
    claimed.mkdir(parents=True, exist_ok=True)
    completed = claimed.parent / "completed"
    completed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.touch()
    print(
        "CAMPAIGN_NODE_MANAGER ready"
        f" host={_host_key(node_value)} job={job_id}"
        f" cpu_affinity={sorted(os.sched_getaffinity(0))}"
        f" cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )

    campaign_out = Path(os.environ["CAMPAIGN_OUT_DIR"])
    done_marker = campaign_out / "allocation_controller.exit"
    cancelled = claimed.parent / "cancelled"
    cancelled.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen[bytes], float | None]] = {}
    while True:
        for request_path in sorted(pending.glob("*.json")):
            try:
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                if _host_key(str(payload["node"])) != _host_key(node_value):
                    continue
                claimed_path = claimed / request_path.name
                os.replace(request_path, claimed_path)
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            request_id = str(payload["request_id"])
            print(
                f"CAMPAIGN_NODE_DISPATCH request={request_id}"
                f" cpus={payload['cpu_ids']} gpu={bool(payload['gpu_count'])}"
                f" argv={shlex.join(payload['argv'])}",
                flush=True,
            )
            try:
                process = _launch(payload, results)
            except (OSError, RuntimeError, ValueError) as exc:
                print(
                    f"CAMPAIGN_NODE_LAUNCH_FAILED request={request_id} error={exc}",
                    file=sys.stderr,
                    flush=True,
                )
                _write_result(results, request_id, 127)
                os.replace(claimed_path, completed / claimed_path.name)
                continue
            running[request_id] = (process, None)

        for request_id, (process, cancel_sent_at) in list(running.items()):
            cancel_path = cancelled / f"{request_id}.cancel"
            if cancel_path.exists() and cancel_sent_at is None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                cancel_sent_at = time.monotonic()
                running[request_id] = (process, cancel_sent_at)
            elif (
                cancel_sent_at is not None
                and time.monotonic() - cancel_sent_at >= 60
                and process.poll() is None
            ):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            returncode = process.poll()
            if returncode is None:
                continue
            _write_result(results, request_id, returncode)
            claimed_path = claimed / f"{request_id}.json"
            os.replace(claimed_path, completed / claimed_path.name)
            print(
                f"CAMPAIGN_NODE_COMPLETE request={request_id} exit={returncode}",
                flush=True,
            )
            del running[request_id]

        if done_marker.exists() and not running and not list(pending.glob("*.json")):
            return 0
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manager", action="store_true")
    group.add_argument("--srun-shim", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.manager:
        return _manager()
    sys.argv = [sys.argv[0], *remaining]
    return _request()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"campaign worker dispatch error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(127) from exc
