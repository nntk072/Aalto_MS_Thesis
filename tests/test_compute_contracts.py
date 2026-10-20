from __future__ import annotations

import pytest

from compute.contracts import ExecutionBackend, JobSpec, ResourceProfile
from compute.environment.capabilities import discover_device_capabilities


class _LocalBackend:
    def submit(self, job: JobSpec) -> str:
        return f"local:{job.command[0]}:{job.resources.cpu_count}"


def test_execution_backend_accepts_model_free_semantic_job_spec() -> None:
    backend: ExecutionBackend = _LocalBackend()
    job = JobSpec(
        command=("python3", "train.py"),
        resources=ResourceProfile(
            cpu_count=4,
            host_memory_bytes=8 * 1024**3,
            walltime_seconds=1800,
        ),
        working_directory="/work/project",
        environment={"OMP_NUM_THREADS": "4"},
    )

    assert backend.submit(job) == "local:python3:4"


def test_resource_profile_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ResourceProfile(cpu_count=0, host_memory_bytes=1, walltime_seconds=1)
    with pytest.raises(ValueError, match="accelerator kind"):
        ResourceProfile(
            cpu_count=1,
            host_memory_bytes=1,
            walltime_seconds=1,
            accelerator_count=1,
        )


def test_device_capabilities_are_backend_neutral() -> None:
    capabilities = discover_device_capabilities("cpu")

    assert capabilities.accelerator_kind is None
    assert capabilities.accelerator_count == 0
    assert capabilities.accelerator_memory_bytes is None
    assert capabilities.architecture


def test_device_capabilities_identify_rocm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    class _Properties:
        name = "AMD Instinct MI210"
        total_memory = 64 * 1024**3
        major = 9
        minor = 0

    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: _Properties())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    capabilities = discover_device_capabilities("cuda:0")

    assert capabilities.accelerator_kind == "rocm"
    assert capabilities.accelerator_backend == "rocm"
    assert capabilities.device_name == "AMD Instinct MI210"
    assert capabilities.compute_capability == (9, 0)
    assert capabilities.supports_bfloat16
