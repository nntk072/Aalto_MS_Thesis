"""Backend-neutral compute capabilities and workload requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DeviceCapabilities:
    architecture: str
    accelerator_kind: str | None
    accelerator_count: int
    accelerator_memory_bytes: int | None
    supports_bfloat16: bool
    accelerator_backend: str | None = None
    device_name: str | None = None
    compute_capability: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    cpu_count: int
    host_memory_bytes: int
    walltime_seconds: int
    accelerator_kind: str | None = None
    accelerator_count: int = 0

    def __post_init__(self) -> None:
        if self.cpu_count <= 0 or self.host_memory_bytes <= 0 or self.walltime_seconds <= 0:
            raise ValueError("resource profile CPU, memory, and walltime values must be positive")
        if self.accelerator_count < 0:
            raise ValueError("accelerator count must be nonnegative")
        if self.accelerator_count and not self.accelerator_kind:
            raise ValueError("accelerator kind is required when accelerator count is nonzero")


@dataclass(frozen=True, slots=True)
class JobSpec:
    command: tuple[str, ...]
    resources: ResourceProfile
    working_directory: str
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.command or not self.working_directory:
            raise ValueError("job command and working directory are required")


class ExecutionBackend(Protocol):
    """Submit a semantic workload through a local, Slurm, or cloud adapter."""

    def submit(self, job: JobSpec) -> str: ...


__all__ = ["DeviceCapabilities", "ExecutionBackend", "JobSpec", "ResourceProfile"]
