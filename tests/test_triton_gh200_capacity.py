from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "compute/slurm/capacity.py"
SPEC = importlib.util.spec_from_file_location("check_gh200_capacity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
capacity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capacity)


def _node(
    *, state: str = "MIXED", cpu_alloc: int = 48, memory_alloc: int = 524288, gpu_alloc: int = 1
) -> str:
    return (
        f"NodeName=gpuarm1 State={state} CPUAlloc={cpu_alloc} CPUTot=144 "
        f"RealMemory=1100000 AllocMem={memory_alloc} FreeMem=1100000 "
        "CfgTRES=cpu=144,mem=1100000M,gres/gpu=2,gres/gpu:gh200=2 "
        f"AllocTRES=cpu={cpu_alloc},mem={memory_alloc}M,gres/gpu={gpu_alloc},"
        f"gres/gpu:gh200={gpu_alloc}"
    )


def test_node_capacity_accepts_one_free_gh200_with_requested_cpu_and_ram() -> None:
    assert capacity.node_has_capacity(_node(), gpu_type="gh200", gpus=1, cpus=48, memory_mb=524288)


def test_node_capacity_rejects_busy_gpu_or_insufficient_cpu_or_ram() -> None:
    for output in (
        _node(gpu_alloc=2),
        _node(cpu_alloc=100),
        _node(memory_alloc=700000),
        _node().replace("FreeMem=1100000", "FreeMem=128000"),
        _node(state="DRAIN"),
    ):
        assert not capacity.node_has_capacity(
            output, gpu_type="gh200", gpus=1, cpus=48, memory_mb=524288
        )
