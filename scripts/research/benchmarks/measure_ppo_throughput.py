"""Measure PPO training throughput (FPS) on the read-only env bundle.

This probe resolves the requested variant/config and applies the same hardware
scaling function as production training. It reports both configured and
effective SB3 values so a probe is not mistaken for a production match when
the requested profile differs.

``total_timesteps`` is configurable and defaults to a longer (not short-probe)
amount of work. It is not a training-quality run -- it only measures throughput.

This is a PPO-core benchmark. It resolves the same PPO configuration and uses
the production bundle environment factory, but does not install production
dashboard, checkpoint, or evaluation callbacks. Callback and checkpoint timing
remain unmeasured, so this is not an end-to-end campaign benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from learning.rl.agent import build_agent
from scripts.research.benchmarks.measure_env_memory import (
    _cgroup_memory_bytes,
    _load_train_slice,
    _tree_pss_mib,
)

from compute.environment.device import scale_training_cfg
from compute.environment.precision import configure_cuda_for_training
from compute.slurm.resources import (
    descendant_pids,
    process_tree_cpu_usec,
    read_process_table,
)
from simulation.environment.bundles import build_env_bundle
from simulation.environment.spec import EnvSpec
from simulation.environment.worker import make_bundle_env
from training.configuration.cli import load_merged_config
from training.runner.seed import _setup_rngs
from training.setup.environments import strategy_from_cfg


class _StepTimingWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Collect per-environment step duration for a diagnostic probe."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self._step_seconds = 0.0
        self._step_count = 0

    def step(self, action: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.env.step(action)
        finally:
            self._step_seconds += time.perf_counter() - started
            self._step_count += 1

    def get_step_timing(self) -> dict[str, float | int]:
        return {"seconds": self._step_seconds, "steps": self._step_count}


def _timed_env_from_factory(env_factory: Any) -> _StepTimingWrapper:
    return _StepTimingWrapper(env_factory())


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar metadata to JSON-native Python values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _process_tree_cpu_seconds() -> float:
    """Return cumulative CPU time for this trainer and its live descendants."""
    process_ids = descendant_pids([os.getpid()], read_process_table())
    return process_tree_cpu_usec(process_ids) / 1_000_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override the resolved/scaled environment count for a separate profile.",
    )
    parser.add_argument(
        "--rollout-steps-per-env",
        type=int,
        default=None,
        help="Override rollout length per environment; this changes PPO update geometry.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the effective PPO minibatch size after hardware scaling.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default=None,
        help="Compile the PPO policy for an isolated learner benchmark.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=None,
        help="Override CUDA float32 matmul precision after production CUDA setup.",
    )
    parser.add_argument(
        "--measure-env-steps",
        action="store_true",
        help="Time TradingEnv.step calls and report per-shard environment time.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total env steps (default: resolved config ppo.total_timesteps).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Production base config.")
    parser.add_argument("--seed", type=int, default=42, help="Training RNG seed (default: 42).")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="key=value overrides, applied after strategy config merge.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to train on; CUDA is required for the production GPU estimate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        default=None,
        help=(
            "Measure an experiments.yaml variant: its base config, overrides, algo and "
            "arch are taken from training.configuration.variant_resolution. Requires --bundle-dir."
        ),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Reuse an already-built bundle at this path instead of rebuilding one. "
            "Rebuilding loads the full train-split frames and needs ~512 GiB."
        ),
    )
    parser.add_argument(
        "--update-semaphore-dir",
        type=Path,
        default=None,
        help="Optional shared lock directory for an isolated multi-trainer update-scheduling probe.",
    )
    parser.add_argument(
        "--update-semaphore-slots",
        type=int,
        default=0,
        help="Maximum concurrent PPO train() sections when using --update-semaphore-dir.",
    )
    return parser.parse_args()


def _acquire_update_slot(directory: Path, slots: int) -> Any:
    """Acquire one cross-process update slot; caller closes the returned handle."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slots):
            handle = (directory / f"slot-{index}.lock").open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        time.sleep(0.01)


def _prepare_bundle(
    bundle_dir: Path | None,
    variant: str | None = None,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
) -> tuple[Any, Path, str, str]:
    """Return resolved config and bundle identity for the production backend.

    Building a bundle from scratch loads the whole train-split frame set (which
    peaked around 473 GiB and OOMs below 256 GiB), so a throughput probe that
    only needs the training loop can point at an already-built bundle instead.
    """
    strategy = "baseline"
    algo, arch = "ppo", "tcn"
    resolved_overrides = list(overrides or [])
    if variant is not None:
        from training.configuration.variant_resolution import resolve_variant

        resolved = resolve_variant(variant)
        if bundle_dir is None:
            raise SystemExit(
                "--variant needs --bundle-dir: rebuilding a bundle is not variant-aware"
            )
        config_path = Path(resolved["base_config"])
        strategy = str(resolved["strategy"])
        algo, arch = str(resolved["algo"]), str(resolved["arch"])
        resolved_overrides = [*resolved["overrides"], *resolved_overrides]

    if config_path is None and bundle_dir is None and not resolved_overrides:
        cfg, bars, features, tickbook = _load_train_slice(None)
        strategy_obj = strategy_from_cfg(cfg)[0]
        bundle_path = build_env_bundle(
            cfg.data.cache_dir,
            bars,
            features,
            cfg,
            raw_columns=list(strategy_obj.raw_columns),
            feature_config_hash="phase4-share-ppo",
            tickbook=tickbook,
            pre_ny_by_date=None,
        )
        del bars, features, tickbook
        gc.collect()
        return (
            cfg,
            bundle_path,
            algo,
            arch,
        )

    if bundle_dir is None:
        raise SystemExit("custom configs/overrides require --bundle-dir")
    config_name = str(config_path or "config/base/default.yaml")
    # This is the same merge routine used by the production CLI: base config,
    # strategy overlay, then explicit overrides.
    cfg = load_merged_config(
        argparse.Namespace(
            config=config_name,
            strategy=strategy,
            variant=None,
            overrides=resolved_overrides,
        )
    )
    if algo != "ppo":
        raise SystemExit(f"PPO throughput probe does not support algo={algo!r}")
    if variant is None:
        arch = str(cfg.agent.get("arch", arch))
    return (
        cfg,
        bundle_dir,
        algo,
        arch,
    )


def _apply_profile_overrides(
    cfg: Any,
    *,
    n_envs: int | None,
    rollout_steps_per_env: int | None,
    batch_size: int | None = None,
) -> None:
    """Apply explicit benchmark overrides after production hardware scaling."""
    if n_envs is not None:
        cfg.env.n_envs = n_envs
    if rollout_steps_per_env is not None:
        cfg.ppo.n_steps = int(cfg.env.n_envs) * rollout_steps_per_env
    if batch_size is not None:
        cfg.ppo.batch_size = batch_size


def _train_probe(
    n_envs: int | None,
    rollout_steps_per_env: int | None,
    total_timesteps: int | None,
    device: str,
    output: Path,
    bundle_dir: Path | None = None,
    variant: str | None = None,
    config_path: Path | None = None,
    overrides: list[str] | None = None,
    seed: int = 42,
    batch_size: int | None = None,
    torch_compile_mode: str | None = None,
    matmul_precision: str | None = None,
    measure_env_steps: bool = False,
    update_semaphore_dir: Path | None = None,
    update_semaphore_slots: int = 0,
) -> dict[str, Any]:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    cfg, bundle_path, algo, arch = _prepare_bundle(bundle_dir, variant, config_path, overrides)
    # This probe always instantiates the shared-bundle backend; make the
    # hardware scaler use the same worker memory estimate as that backend.
    cfg.env.data_source = "bundle"
    configure_cuda_for_training()
    scale_training_cfg(cfg, torch.device(device), arch)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    _apply_profile_overrides(
        cfg,
        n_envs=n_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        batch_size=batch_size,
    )
    n_envs = int(cfg.env.n_envs)
    total_timesteps = int(
        total_timesteps if total_timesteps is not None else cfg.ppo.total_timesteps
    )
    spec = EnvSpec.from_cfg(
        cfg,
        algo=algo,
        reward=str(cfg.env.get("reward_mode", "dsr")),
        arch=arch,
        bundle_dir=str(bundle_path),
    )
    cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    config_sha256 = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _setup_rngs(seed)

    gpu_before: dict[str, int | str] = {}
    if device == "cuda":
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu_before = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_total_bytes": total_bytes,
            "gpu_free_before_bytes": free_bytes,
        }
        torch.cuda.reset_peak_memory_stats()

    start_bytes = _cgroup_memory_bytes()
    parent_env = make_bundle_env(spec)
    if not isinstance(parent_env.observation_space, spaces.Dict):
        raise TypeError("PPO throughput probe requires a Dict observation space")
    env_fn: Callable[[], gym.Env[Any, Any]] = partial(make_bundle_env, spec)
    if measure_env_steps:
        env_fn = partial(_timed_env_from_factory, env_fn)
    model = build_agent(parent_env, cfg, arch=arch, algo=algo, device=device, env_fn=env_fn)
    vec_env = model.get_env()
    compile_warmup_seconds = 0.0
    if torch_compile_mode is not None:
        if device != "cuda":
            raise ValueError("torch.compile mode probes require --device cuda")
        from stable_baselines3.common.utils import obs_as_tensor

        py_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all()
        compile_started = time.monotonic()
        model.policy = torch.compile(model.policy, mode=torch_compile_mode)
        inference_obs = {
            key: np.broadcast_to(space.sample(), (int(model.n_envs), *(space.shape or ()))).copy()
            for key, space in parent_env.observation_space.spaces.items()
        }
        training_obs = {
            key: np.broadcast_to(
                space.sample(), (int(model.batch_size), *(space.shape or ()))
            ).copy()
            for key, space in parent_env.observation_space.spaces.items()
        }
        inference_obs_tensor = obs_as_tensor(inference_obs, model.device)
        training_obs_tensor = obs_as_tensor(training_obs, model.device)
        sample_count = int(model.batch_size)
        if isinstance(model.action_space, spaces.Discrete):
            sample_actions = torch.zeros(sample_count, dtype=torch.long, device=model.device)
        else:
            action_shape = tuple(model.action_space.shape)
            sample_actions = torch.zeros(
                (sample_count, *action_shape), dtype=torch.float32, device=model.device
            )
        with torch.no_grad():
            model.policy(inference_obs_tensor)
        model.policy.evaluate_actions(training_obs_tensor, sample_actions)
        torch.cuda.synchronize()
        compile_warmup_seconds = time.monotonic() - compile_started
        # Compilation warmup must not advance the benchmark's seeded policy RNG.
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_state)
    vector_backend = type(vec_env).__name__
    remotes = getattr(vec_env, "remotes", None)
    physical_worker_count = len(remotes) if remotes is not None else 1
    logical_envs_per_worker = getattr(vec_env, "logical_envs_per_shard", None)
    vecenv_bytes = _cgroup_memory_bytes()

    # n_steps is split by n_envs inside build_agent (n_steps // n_envs per env).
    rollout_steps_per_env = int(model.n_steps)
    nominal_env_steps = int(model.n_envs * model.n_steps)

    # Time PPO's gradient phase separately from the env rollout phase, so the
    # reported FPS can be split into "how fast can the envs step" and "how much
    # of the budget the n_epochs=15 update loop costs". SB3 calls ``self.train()``
    # from inside ``learn()``, so shadowing it on the instance times that phase.
    update_timing = {
        "update_seconds": 0.0,
        "update_compute_seconds": 0.0,
        "update_semaphore_wait_seconds": 0.0,
        "update_calls": 0,
    }
    optimizer_timing = {"optimizer_steps": 0}
    original_train = model.train
    original_optimizer_step = model.policy.optimizer.step

    def _counted_optimizer_step(*args: Any, **kwargs: Any) -> Any:
        optimizer_timing["optimizer_steps"] += 1
        return original_optimizer_step(*args, **kwargs)

    model.policy.optimizer.step = _counted_optimizer_step

    def _timed_train(*args: Any, **kwargs: Any) -> Any:
        update_started = time.monotonic()
        slot_handle = (
            _acquire_update_slot(update_semaphore_dir, update_semaphore_slots)
            if update_semaphore_dir is not None and update_semaphore_slots > 0
            else None
        )
        compute_started = time.monotonic()
        try:
            return original_train(*args, **kwargs)
        finally:
            if device == "cuda":
                torch.cuda.synchronize()
            compute_elapsed = time.monotonic() - compute_started
            if slot_handle is not None:
                import fcntl

                fcntl.flock(slot_handle.fileno(), fcntl.LOCK_UN)
                slot_handle.close()
            elapsed = time.monotonic() - update_started
            update_timing["update_seconds"] += elapsed
            update_timing["update_compute_seconds"] += compute_elapsed
            update_timing["update_semaphore_wait_seconds"] += max(elapsed - compute_elapsed, 0.0)
            update_timing["update_calls"] += 1

    model.train = _timed_train

    cpu_seconds_before = _process_tree_cpu_seconds()
    started = time.monotonic()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    cpu_seconds_after = _process_tree_cpu_seconds()
    model.train = original_train
    model.policy.optimizer.step = original_optimizer_step

    env_step_stats: list[dict[str, float | int]] = []
    env_step_shard_seconds: list[float] = []
    if measure_env_steps:
        env_step_stats = vec_env.env_method("get_step_timing")
        widths = getattr(vec_env, "logical_envs_per_shard", [1] * len(env_step_stats))
        cursor = 0
        for width in widths:
            env_step_shard_seconds.append(
                sum(float(item["seconds"]) for item in env_step_stats[cursor : cursor + width])
            )
            cursor += width

    # SB3 counts every per-env step into ``num_timesteps``, so this is the real
    # number of env steps the rollout produced (>= total_timesteps because the
    # last rollout overshoots).
    actual_env_steps = int(model.num_timesteps)
    update_seconds = float(update_timing["update_seconds"])
    rollout_and_framework_seconds = max(training_seconds - update_seconds, 0.0)
    process_tree_cpu_seconds = max(cpu_seconds_after - cpu_seconds_before, 0.0)
    average_cpu_cores = process_tree_cpu_seconds / max(training_seconds, 1e-9)
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None
    head_cfg = cfg.agent.get("head_arch", {})

    gpu_after: dict[str, int] = {}
    if device == "cuda":
        torch.cuda.synchronize()
        free_bytes, _ = torch.cuda.mem_get_info()
        gpu_after = {
            "gpu_free_after_bytes": free_bytes,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    report = {
        "profile_scope": "ppo_core_without_production_callbacks",
        "production_callbacks_included": False,
        "callback_evaluation_seconds": None,
        "checkpoint_seconds": None,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "working_tree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        ),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variant": variant,
        "config_path": str(config_path) if config_path is not None else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "config_sha256": config_sha256,
        "resolved_config_yaml": cfg_text,
        "bundle_manifest_sha256": (
            hashlib.sha256((bundle_path / "manifest.json").read_bytes()).hexdigest()
            if (bundle_path / "manifest.json").is_file()
            else None
        ),
        "arch": arch,
        "algo": algo,
        "seed": seed,
        "n_envs": int(model.n_envs),
        "vector_backend": vector_backend,
        "rollout_buffer_class": type(model.rollout_buffer).__name__,
        "rollout_buffer_on_device": bool(cfg.ppo.get("rollout_buffer_on_device", False)),
        "defer_metric_sync": bool(cfg.ppo.get("defer_metric_sync", False)),
        "ppo_class": type(model).__name__,
        "torch_compile_mode": torch_compile_mode,
        "torch_compile_warmup_seconds": compile_warmup_seconds,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "env_step_timing_enabled": measure_env_steps,
        "env_step_calls": sum(int(item["steps"]) for item in env_step_stats),
        "env_step_cpu_seconds_total": sum(float(item["seconds"]) for item in env_step_stats),
        "env_step_wall_lower_bound_seconds": max(env_step_shard_seconds, default=0.0),
        "env_step_mean_ms": (
            1000.0
            * sum(float(item["seconds"]) for item in env_step_stats)
            / max(sum(int(item["steps"]) for item in env_step_stats), 1)
        ),
        "configured_vector_backend": str(cfg.env.get("vector_backend", "subproc")),
        "physical_worker_count": physical_worker_count,
        "logical_envs_per_worker": logical_envs_per_worker
        if logical_envs_per_worker is not None
        else [1] * physical_worker_count,
        "total_timesteps": total_timesteps,
        "rollout_steps_per_env": rollout_steps_per_env,
        "aggregate_rollout_size": nominal_env_steps,
        "n_epochs": int(model.n_epochs),
        "batch_size": int(model.batch_size),
        "minibatches_per_epoch": int(
            (nominal_env_steps + int(model.batch_size) - 1) // int(model.batch_size)
        ),
        "optimizer_steps": optimizer_timing["optimizer_steps"],
        "ppo_epochs_completed": int(getattr(model, "_n_updates", 0)),
        "configured_ppo_n_steps": int(cfg.ppo.n_steps),
        "effective_ppo_n_steps_per_env": int(model.n_steps),
        "learning_rate": float(cfg.ppo.learning_rate),
        "gamma": float(cfg.ppo.gamma),
        "gae_lambda": float(cfg.ppo.gae_lambda),
        "clip_range": float(cfg.ppo.clip_range),
        "ent_coef": float(cfg.ppo.ent_coef),
        "effective_ent_coef": float(model.ent_coef),
        "parameter_count": sum(p.numel() for p in model.policy.parameters()),
        "feature_extractor_class": type(model.policy.features_extractor).__name__,
        "encoder_spec": {
            "channels": getattr(model.policy.features_extractor, "channels", None),
            "dilations": list(model.policy.features_extractor.dilations)
            if hasattr(model.policy.features_extractor, "dilations")
            else None,
            "kernel_size": getattr(model.policy.features_extractor, "kernel_size", None),
            "latent_dim": int(getattr(model.policy.features_extractor, "latent_dim", 128)),
            "parameters": {
                key: getattr(model.policy.features_extractor, key)
                for key in ("hidden_size", "num_layers", "d_model", "nhead", "dim_feedforward")
                if hasattr(model.policy.features_extractor, key)
            },
            "receptive_field": getattr(model.policy.features_extractor, "receptive_field", None),
        },
        "policy_head_architecture": [int(width) for width in head_cfg.get(arch, [256, 128])],
        "observation_shapes": {
            key: list(space.shape or ())
            for key, space in parent_env.observation_space.spaces.items()
        },
        "action_space": {
            "class": type(parent_env.action_space).__name__,
            "shape": list(getattr(parent_env.action_space, "shape", ()) or ()),
            "n": getattr(parent_env.action_space, "n", None),
        },
        "action_schema": (
            parent_env.action_schema.to_dict()
            if hasattr(parent_env, "action_schema") and hasattr(parent_env.action_schema, "to_dict")
            else None
        ),
        "device": device,
        "bundle_size_bytes": sum(
            path.stat().st_size for path in bundle_path.iterdir() if path.is_file()
        ),
        "cgroup_start_bytes": start_bytes,
        "cgroup_vecenv_ready_bytes": vecenv_bytes,
        "cgroup_after_training_bytes": _cgroup_memory_bytes(),
        "trainer_tree_pss_mib": _tree_pss_mib(os.getpid()),
        "process_tree_cpu_seconds": process_tree_cpu_seconds,
        "average_cpu_cores_used": average_cpu_cores,
        "allocated_cpus": allocated_cpus,
        "allocated_cpu_fraction": (
            average_cpu_cores / allocated_cpus if allocated_cpus is not None else None
        ),
        "training_seconds": training_seconds,
        "total_wallclock_seconds": training_seconds,
        "env_steps_per_second": actual_env_steps / max(training_seconds, 1e-9),
        "effective_total_timesteps": total_timesteps,
        "actual_env_steps": actual_env_steps,
        "nominal_env_steps_per_rollout_window": nominal_env_steps,
        "update_seconds": update_seconds,
        "update_compute_seconds": float(update_timing["update_compute_seconds"]),
        "update_semaphore_wait_seconds": float(update_timing["update_semaphore_wait_seconds"]),
        "update_semaphore_slots": update_semaphore_slots or None,
        "update_calls": update_timing["update_calls"],
        "rollout_and_framework_overhead_seconds": rollout_and_framework_seconds,
        "rollout_steps_per_second": actual_env_steps / max(rollout_and_framework_seconds, 1e-9),
        "update_share_of_wallclock": update_seconds / max(training_seconds, 1e-9),
        **gpu_before,
        **gpu_after,
    }
    model.get_env().close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    return report


def main() -> None:
    args = _parse_args()
    if (
        (args.n_envs is not None and args.n_envs < 1)
        or (args.rollout_steps_per_env is not None and args.rollout_steps_per_env < 1)
        or (args.batch_size is not None and args.batch_size < 1)
        or (args.total_timesteps is not None and args.total_timesteps < 1)
        or args.update_semaphore_slots < 0
    ):
        raise SystemExit(
            "n_envs, rollout steps, batch size, and total_timesteps must be positive; "
            "update semaphore slots cannot be negative"
        )
    if (args.update_semaphore_slots > 0) != (args.update_semaphore_dir is not None):
        raise SystemExit(
            "set both --update-semaphore-slots and --update-semaphore-dir for semaphore probes"
        )
    report = _train_probe(
        args.n_envs,
        args.rollout_steps_per_env,
        args.total_timesteps,
        args.device,
        args.output,
        args.bundle_dir,
        args.variant,
        args.config,
        args.overrides,
        args.seed,
        args.batch_size,
        args.torch_compile_mode,
        args.matmul_precision,
        args.measure_env_steps,
        args.update_semaphore_dir,
        args.update_semaphore_slots,
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
