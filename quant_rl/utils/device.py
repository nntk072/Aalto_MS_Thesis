"""Shared PyTorch device resolution and CUDA throughput knobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_GiB = 1024**3
# SubprocVecEnv workers import torch and copy the full feature/environment
# state. Steady-state 32-worker RSS was ~2.4 GiB/worker, but forkserver pickle
# spawn peaks higher: 64 workers @ 256G Slurm OOM'd mid-unpickle on the PO3
# full feature matrix. Plan at 4 GiB/worker so 256G caps at 32 and 512G fits 64.
_WORKER_RAM_BYTES = 4 * _GiB
_RAM_HEADROOM_BYTES = int(1.5 * _GiB)
# Main process holds the policy, Adam states, and the source env (~3 GiB observed).
_PARENT_RAM_BYTES = 3 * _GiB


def get_device(config_device: str | None = None) -> torch.device:
    """Resolve the torch device to use.

    Parameters
    ----------
    config_device:
        Optional device hint from config. Accepted values:

        * ``None`` or ``"auto"`` — pick CUDA if available, else CPU.
        * ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ... — use as-is.
        * Any other string raises ``ValueError``.

    Returns
    -------
    torch.device
    """
    if not config_device or config_device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        return torch.device(config_device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device value {config_device!r}") from exc


def configure_cuda() -> None:
    """Enable TF32 Tensor Cores and cuDNN autotune. No-op when CUDA is absent."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def suggest_n_envs(
    *,
    requested: int,
    total_ram_bytes: int,
    available_ram_bytes: int,
    cpu_count: int,
    vram_bytes: int | None,
) -> int:
    """Pick a SubprocVecEnv width that fits RAM (power of two).

    ``requested <= 1`` is left as DummyVecEnv. Otherwise the value is scaled
    up toward the GPU target, then capped by the effective allocation's free
    RAM after headroom.
    ``cpu_count`` is retained for API compatibility, but CPU allocation is not
    used as a hard worker cap because environment workers can safely share the
    allocated CPUs while the memory cap remains the OOM safeguard.

    Parameters
    ----------
    requested:
        ``cfg.env.n_envs`` before scaling.
    total_ram_bytes:
        Host MemTotal.
    available_ram_bytes:
        Host MemAvailable (leaves other processes alone).
    cpu_count:
        Logical CPUs.
    vram_bytes:
        GPU VRAM, or ``None`` on CPU (do not scale up).

    Returns
    -------
    int
    """
    if requested <= 1:
        return 1

    # Required allocation ~= parent + headroom + (workers * worker estimate).
    # The power-of-two result is the largest rollout width that satisfies it.
    max_from_avail = max(
        1,
        (available_ram_bytes - _RAM_HEADROOM_BYTES - _PARENT_RAM_BYTES) // _WORKER_RAM_BYTES,
    )
    gpu_target = requested if vram_bytes is None else _gpu_n_envs_target(vram_bytes)
    cap = min(max_from_avail, gpu_target)
    target = min(max(requested, gpu_target), cap)
    return _floor_power_of_two(max(1, target))


def suggest_n_steps(*, current: int, vram_bytes: int | None) -> int:
    """Grow the PPO rollout so A100-class cards get larger update buffers."""
    if vram_bytes is None:
        return current
    if vram_bytes >= 35 * _GiB:
        return max(current, 8192)
    if vram_bytes >= 16 * _GiB:
        return max(current, 4096)
    return current


def suggest_batch_size(
    *,
    current: int,
    n_steps: int,
    n_envs: int,
    vram_bytes: int | None,
    arch: str,
) -> int:
    """Raise minibatch size to fill VRAM; result always divides the rollout.

    Parameters
    ----------
    current:
        Configured ``batch_size`` (lower bound when it still divides).
    n_steps:
        PPO ``n_steps`` *before* dividing by ``n_envs``.
    n_envs:
        Parallel env count after RAM capping.
    vram_bytes:
        GPU VRAM, or ``None`` to keep ``current`` (CPU).
    arch:
        Encoder name; ``transformer`` is treated as the heavy model.

    Returns
    -------
    int
    """
    rollout = n_steps if n_envs <= 1 else max(1, n_steps // n_envs) * n_envs
    if vram_bytes is None or rollout <= 0:
        return min(current, max(1, rollout or current))

    heavy = arch == "transformer"
    if vram_bytes >= 70 * _GiB:
        want = 8192 if heavy else 4096
    elif vram_bytes >= 35 * _GiB:
        want = 4096 if heavy else 2048
    elif vram_bytes >= 16 * _GiB:
        want = 2048 if heavy else 1024
    elif vram_bytes >= 8 * _GiB:
        want = 1024 if heavy else 512
    else:
        # 6 GB laptop: 12-layer transformer + Adam barely fits.
        want = 512 if heavy else 1024
    want = min(max(current, want), rollout)
    return _largest_divisor_pow2(rollout, want)


def suggest_n_epochs(*, current: int, has_cuda: bool) -> int:
    """More PPO epochs per rollout keeps the GPU busy while envs catch up."""
    if not has_cuda:
        return current
    return max(current, 15)


def scale_training_cfg(cfg: Any, device: torch.device, arch: str) -> None:
    """Mutate ``cfg`` in place with RAM/VRAM-aware PPO/SAC throughput settings."""
    configure_cuda()
    total_ram, avail_ram = _read_meminfo()
    cgroup_memory = _read_cgroup_memory_limit()
    if cgroup_memory is not None:
        limit, current = cgroup_memory
        total_ram = min(total_ram, limit)
        avail_ram = min(avail_ram, max(0, limit - current))
    slurm_memory = _read_slurm_memory_limit()
    if slurm_memory is not None:
        total_ram = min(total_ram, slurm_memory)
        avail_ram = min(avail_ram, slurm_memory)
    cpu_count = _read_cgroup_cpu_count() or _read_slurm_cpu_limit() or os.cpu_count() or 1
    vram: int | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        vram = int(torch.cuda.get_device_properties(idx).total_memory)

    requested = int(getattr(cfg.get("env", {}), "n_envs", 1))
    n_envs = suggest_n_envs(
        requested=requested,
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        cpu_count=cpu_count,
        vram_bytes=vram,
    )
    max_envs_env = os.environ.get("QUANT_RL_MAX_N_ENVS")
    if max_envs_env is not None:
        n_envs = min(n_envs, max(1, int(max_envs_env)))
        n_envs = _floor_power_of_two(max(1, n_envs))
    cfg.env.n_envs = n_envs

    n_steps = suggest_n_steps(current=int(cfg.ppo.n_steps), vram_bytes=vram)
    cfg.ppo.n_steps = n_steps
    cfg.ppo.n_epochs = suggest_n_epochs(
        current=int(cfg.ppo.n_epochs), has_cuda=device.type == "cuda"
    )
    cfg.ppo.batch_size = suggest_batch_size(
        current=int(cfg.ppo.batch_size),
        n_steps=n_steps,
        n_envs=n_envs,
        vram_bytes=vram,
        arch=arch,
    )
    sac_cfg = cfg.get("sac")
    if sac_cfg is not None:
        sac_cfg.batch_size = suggest_batch_size(
            current=int(sac_cfg.get("batch_size", cfg.ppo.batch_size)),
            n_steps=n_steps,
            n_envs=n_envs,
            vram_bytes=vram,
            arch=arch,
        )


def enable_extractor_autocast(extractor: nn.Module) -> None:
    """Run the encoder under bf16 autocast; cast back to fp32 for SB3 heads."""
    from quant_rl.models.encoder import TransformerEncoder

    # Transformer attention in bf16 can NaN under PPO's large rollout batches.
    if isinstance(extractor, TransformerEncoder):
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return
    if getattr(extractor, "_amp_wrapped", False):
        return
    orig = extractor.forward

    def forward_amp(observations: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = orig(observations)
        return out.float()  # type: ignore[no-any-return]

    extractor.forward = forward_amp
    setattr(extractor, "_amp_wrapped", True)


def _gpu_n_envs_target(vram_bytes: int) -> int:
    if vram_bytes >= 70 * _GiB:
        return 64
    if vram_bytes >= 35 * _GiB:
        return 16
    if vram_bytes >= 8 * _GiB:
        return 8
    return 8


def _floor_power_of_two(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _largest_divisor_pow2(rollout: int, cap: int) -> int:
    best = 1
    cand = 1
    while cand <= cap:
        if rollout % cand == 0:
            best = cand
        cand *= 2
    return best


def _read_meminfo() -> tuple[int, int]:
    total = 0
    available = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    return total, available


def _read_cgroup_memory_limit() -> tuple[int, int] | None:
    """Return the current cgroup memory limit and usage, when constrained.

    Slurm jobs are commonly constrained by cgroup v2 while ``/proc/meminfo``
    still exposes the whole node.  Prefer the cgroup limit so worker scaling
    cannot exceed the job allocation.
    """
    candidates = [
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ]
    cgroup_dir = _current_cgroup_dir()
    if cgroup_dir is not None:
        candidates.insert(
            0,
            (cgroup_dir / "memory.max", cgroup_dir / "memory.current"),
        )
        candidates.append(
            (
                cgroup_dir / "memory.limit_in_bytes",
                cgroup_dir / "memory.usage_in_bytes",
            )
        )
    for limit_path, current_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            current_text = current_path.read_text(encoding="utf-8").strip()
            if not limit_text or limit_text == "max":
                continue
            limit = int(limit_text)
            current = max(0, int(current_text))
        except (OSError, ValueError):
            continue
        if limit > 0:
            return limit, current
    return None


def _read_cgroup_cpu_count() -> int | None:
    """Return the CPU quota as a worker cap, when a quota is configured."""
    paths = [Path("/sys/fs/cgroup/cpu.max")]
    cgroup_dir = _current_cgroup_dir()
    if cgroup_dir is not None:
        paths.insert(0, cgroup_dir / "cpu.max")
    for path in paths:
        try:
            quota, period = path.read_text(encoding="utf-8").split()
        except (OSError, ValueError):
            continue
        if quota == "max":
            continue
        try:
            return max(1, int(int(quota) / int(period)))
        except (ValueError, ZeroDivisionError):
            continue
    return None


def _current_cgroup_dir() -> Path | None:
    """Resolve this process's cgroup v2 directory, if available."""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, _, relative = line.partition("::")
            if not hierarchy and relative:
                return Path("/sys/fs/cgroup") / relative.lstrip("/")
    except OSError:
        pass
    return None


def _read_slurm_memory_limit() -> int | None:
    """Return a Slurm memory allocation in bytes, when exported by Slurm."""
    value = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    if not value:
        return None
    try:
        suffix = value[-1].upper()
        if suffix in "KMGT":
            multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[suffix]
            number = float(value[:-1])
        else:
            # Slurm reports SLURM_MEM_PER_NODE/CPU in megabytes by default.
            multiplier = 1024**2
            number = float(value)
        return max(1, int(number * multiplier))
    except ValueError:
        return None


def _read_slurm_cpu_limit() -> int | None:
    """Return Slurm's allocated CPU count, when exported by Slurm."""
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return max(1, int(value)) if value else None
    except ValueError:
        return None
