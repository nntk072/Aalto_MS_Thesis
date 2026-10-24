"""Tests for the allocation-local worker command dispatcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from compute.slurm.worker_dispatch import _gpu_device_for_request, _host_key, _parse_srun

DISPATCHER = Path(__file__).resolve().parents[1] / "compute" / "slurm" / "worker_dispatch.py"


def test_parse_campaign_worker_command() -> None:
    parsed = _parse_srun(
        [
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--nodelist=gpuarm2.int.triton.aalto.fi",
            "--cpus-per-task=48",
            "--gpus=gh200:1",
            "--gpu-bind=map_gpu:0",
            "--cpu-bind=mask_cpu:0x60",
            "--chdir",
            "/scratch/work/nguyenl37/Aalto_MS_Thesis",
            "--export=ALL",
            "/venv/bin/python",
            "/repo/scripts/compute/slurm/verify_worker_cpu_affinity.py",
            "--expected-cpus",
            "5,6",
            "--",
            "bash",
            "/repo/scripts/research/frozen-launches/train_one_variant.sh",
        ]
    )

    assert parsed == {
        "node": "gpuarm2",
        "cwd": "/scratch/work/nguyenl37/Aalto_MS_Thesis",
        "argv": [
            "/venv/bin/python",
            "/repo/scripts/compute/slurm/verify_worker_cpu_affinity.py",
            "--expected-cpus",
            "5,6",
            "--",
            "bash",
            "/repo/scripts/research/frozen-launches/train_one_variant.sh",
        ],
        "cpu_ids": [5, 6],
        "gpu_count": 1,
        "gpu_id": 0,
    }


def test_gpu_worker_binding_selects_the_requested_visible_device() -> None:
    request = {"gpu_count": 1, "gpu_id": 1}
    assert _gpu_device_for_request(request, ["2", "4"]) == "4"
    with pytest.raises(RuntimeError, match="allocation exposes 1 visible GPU"):
        _gpu_device_for_request(request, ["2"])


def test_parse_campaign_resource_sampler_without_gpu() -> None:
    parsed = _parse_srun(
        [
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            "--nodelist=gpuarm1",
            "--cpus-per-task=48",
            "--cpu-bind=mask_cpu:0x20",
            "--chdir",
            "/repo",
            "--export=ALL",
            "/venv/bin/python",
            "/repo/compute/slurm/resources.py",
            "--campaign",
            "/repo/campaign.json",
            "--out",
            "/repo/resources.csv",
        ]
    )

    assert parsed["node"] == "gpuarm1"
    assert parsed["cpu_ids"] == [5]
    assert parsed["gpu_count"] == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--nodelist=gpuarm1", "--chdir", "/repo", "python", "worker.py"],
        [
            "--nodelist=gpuarm1",
            "--cpu-bind=cores",
            "--chdir",
            "/repo",
            "python",
            "worker.py",
        ],
    ],
)
def test_parse_rejects_incomplete_or_unsupported_srun(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        _parse_srun(argv)


def test_host_key_strips_domain_suffix() -> None:
    assert _host_key("gpuarm1.int.triton.aalto.fi") == "gpuarm1"


def test_persistent_manager_dispatches_worker_with_affinity_and_gpu_binding(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue"
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    node = "gpuarm-test"
    job_id = "dispatch-test-job"
    manager_env = dict(
        os.environ,
        CAMPAIGN_WORKER_QUEUE=str(queue),
        CAMPAIGN_OUT_DIR=str(campaign),
        SLURM_JOB_ID=job_id,
        SLURMD_NODENAME=node,
        CUDA_VISIBLE_DEVICES="2,4",
    )
    manager_log_path = tmp_path / "manager.log"
    with manager_log_path.open("wb") as manager_log:
        manager = subprocess.Popen(
            [sys.executable, str(DISPATCHER), "--manager"],
            env=manager_env,
            stdout=manager_log,
            stderr=subprocess.STDOUT,
        )
        try:
            ready = queue / job_id / node / "ready"
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists()
            ready_metadata = json.loads(ready.read_text())
            assert ready_metadata == {
                "node": node,
                "job_id": job_id,
                "cpu_ids": sorted(os.sched_getaffinity(0)),
            }

            cpu_id = min(os.sched_getaffinity(0))
            worker_log = tmp_path / "worker.log"
            with worker_log.open("wb") as output:
                shim = subprocess.run(
                    [
                        sys.executable,
                        str(DISPATCHER),
                        "--srun-shim",
                        "--nodes=1",
                        "--ntasks=1",
                        f"--nodelist={node}",
                        "--cpus-per-task=1",
                        "--gpus=gh200:1",
                        "--gpu-bind=map_gpu:1",
                        f"--cpu-bind=mask_cpu:0x{1 << cpu_id:x}",
                        "--chdir",
                        str(tmp_path),
                        "--export=ALL",
                        sys.executable,
                        "-c",
                        "import os; print(sorted(os.sched_getaffinity(0))); "
                        "print(os.environ['CUDA_VISIBLE_DEVICES']); "
                        "print(os.environ['GPU_ID'])",
                    ],
                    env=manager_env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=5,
                )
            assert shim.returncode == 0
            assert worker_log.read_text().splitlines() == [f"[{cpu_id}]", "4", "4"]
        finally:
            (campaign / "allocation_controller.exit").touch()
            assert manager.wait(timeout=5) == 0
    assert "CAMPAIGN_NODE_MANAGER ready" in manager_log_path.read_text()


def test_cancelled_dispatch_terminates_the_node_child(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    node = "gpuarm-cancel-test"
    job_id = "dispatch-cancel-test-job"
    manager_env = dict(
        os.environ,
        CAMPAIGN_WORKER_QUEUE=str(queue),
        CAMPAIGN_OUT_DIR=str(campaign),
        SLURM_JOB_ID=job_id,
        SLURMD_NODENAME=node,
        CUDA_VISIBLE_DEVICES="2",
    )
    manager_log_path = tmp_path / "manager.log"
    worker_marker = tmp_path / "worker-finished"
    cpu_id = min(os.sched_getaffinity(0))
    command = [
        sys.executable,
        str(DISPATCHER),
        "--srun-shim",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={node}",
        "--cpus-per-task=1",
        "--gpus=gh200:1",
        "--gpu-bind=map_gpu:0",
        f"--cpu-bind=mask_cpu:0x{1 << cpu_id:x}",
        "--chdir",
        str(tmp_path),
        "--export=ALL",
        sys.executable,
        "-c",
        f"import time; time.sleep(10); open({str(worker_marker)!r}, 'w').close()",
    ]
    with manager_log_path.open("wb") as manager_log:
        manager = subprocess.Popen(
            [sys.executable, str(DISPATCHER), "--manager"],
            env=manager_env,
            stdout=manager_log,
            stderr=subprocess.STDOUT,
        )
        with (tmp_path / "worker.log").open("wb") as worker_log:
            shim = subprocess.Popen(
                command,
                env=manager_env,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
            )
            try:
                claimed = queue / job_id / node / "claimed"
                deadline = time.monotonic() + 5
                while not list(claimed.glob("*.json")) and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert list(claimed.glob("*.json"))
                shim.terminate()
                assert shim.wait(timeout=3) == 143
                deadline = time.monotonic() + 5
                results = queue / job_id / "results"
                while not list(results.glob("*.json")) and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert list(results.glob("*.json"))
            finally:
                (campaign / "allocation_controller.exit").touch()
                assert manager.wait(timeout=5) == 0
    assert not worker_marker.exists()
