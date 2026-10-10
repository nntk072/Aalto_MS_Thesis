"""Measure PPO training throughput (FPS) on the read-only env bundle.

This probe follows the production training configuration so the measured
environment-steps-per-second is representative of a real run:

    env.n_envs       -> 64 (production)
    ppo.n_steps      -> 2048  (production, = n_envs * rollout_steps)
    ppo.batch_size   -> 512   (production)
    ppo.n_epochs     -> 15    (production CUDA default)
    ppo.gamma        -> 0.99
    ppo.learning_rate-> 3.0e-4

``total_timesteps`` is configurable and defaults to a longer (not short-probe)
amount of work. It is not a training-quality run -- it only measures throughput.

The report separates the two halves of PPO wall-clock: ``rollout_seconds`` (the
VecEnv stepping phase, which is what a raw ``env.step()`` microbenchmark
measures) and ``update_seconds`` (the ``n_epochs`` minibatch gradient loop,
which the short probe skipped by running ``n_epochs=1``). ``env_steps_per_second``
divides the steps SB3 actually collected by total wall-clock.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from scripts.bench.measure_env_memory import (
    _cgroup_memory_bytes,
    _load_train_slice,
    _tree_pss_mib,
)

from quant_rl.config import load_config
from quant_rl.envs.env_bundle_builder import build_env_bundle
from quant_rl.envs.env_spec import EnvSpec
from quant_rl.envs.worker import make_bundle_env
from quant_rl.models.agent import build_agent
from quant_rl.train.train_rl import _strategy_from_cfg

# Production PPO defaults (quant_rl/config/default.yaml), CUDA path.
_PROD_N_STEPS = 2048
_PROD_BATCH_SIZE = 512
_PROD_N_EPOCHS = 15
_PROD_GAMMA = 0.99
_PROD_LR = 3.0e-4
_PROD_ENT_COEF = 0.01
_PROD_GAE_LAMBDA = 0.95
_PROD_CLIP_RANGE = 0.2
_PROD_EPOCHS_CPU = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--rollout-steps-per-env", type=int, default=32)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total env steps to train for (default: 200k, production full run is 20M).",
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Multiplicity of the rollout window."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to train on; CUDA is required for the production GPU estimate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "Reuse an already-built bundle at this path instead of rebuilding one. "
            "Rebuilding loads the full train-split frames and needs ~512 GiB."
        ),
    )
    return parser.parse_args()


def _prepare_bundle(bundle_dir: Path | None) -> tuple[Any, EnvSpec, Path]:
    """Return cfg + EnvSpec, reusing an existing bundle when *bundle_dir* is given.

    Building a bundle from scratch loads the whole train-split frame set (which
    peaked around 473 GiB and OOMs below 256 GiB), so a throughput probe that
    only needs the training loop can point at an already-built bundle instead.
    """
    if bundle_dir is not None:
        cfg = load_config()
        return (
            cfg,
            EnvSpec.from_cfg(
                cfg,
                algo="ppo",
                reward="dsr",
                arch="tcn",
                bundle_dir=str(bundle_dir),
            ),
            bundle_dir,
        )

    cfg, bars, features, tickbook = _load_train_slice(None)
    strategy = _strategy_from_cfg(cfg)[0]
    bundle_path = build_env_bundle(
        cfg.data.cache_dir,
        bars,
        features,
        cfg,
        raw_columns=list(strategy.raw_columns),
        feature_config_hash="phase4-share-ppo",
        tickbook=tickbook,
        pre_ny_by_date=None,
    )
    del bars, features, tickbook
    gc.collect()
    return (
        cfg,
        EnvSpec.from_cfg(
            cfg,
            algo="ppo",
            reward="dsr",
            arch="tcn",
            bundle_dir=str(bundle_path),
        ),
        bundle_path,
    )


def _apply_production_cfg(cfg: Any, n_envs: int, device: str) -> None:
    """Apply the production PPO configuration to the probe."""
    cfg.env.n_envs = n_envs
    cfg.env.data_source = "bundle"
    cfg.ppo.n_steps = _PROD_N_STEPS
    cfg.ppo.batch_size = _PROD_BATCH_SIZE
    cfg.ppo.n_epochs = _PROD_N_EPOCHS if device == "cuda" else _PROD_EPOCHS_CPU
    cfg.ppo.gamma = _PROD_GAMMA
    cfg.ppo.learning_rate = _PROD_LR
    cfg.ppo.ent_coef = _PROD_ENT_COEF
    cfg.ppo.gae_lambda = _PROD_GAE_LAMBDA
    cfg.ppo.clip_range = _PROD_CLIP_RANGE
    cfg.ppo.total_timesteps = cfg.ppo.get("total_timesteps", 20_000_000)


def _train_probe(
    n_envs: int,
    total_timesteps: int,
    iterations: int,
    device: str,
    output: Path,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    cfg, spec, bundle_path = _prepare_bundle(bundle_dir)
    _apply_production_cfg(cfg, n_envs, device)

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

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
    env_fn = partial(make_bundle_env, spec)
    model = build_agent(parent_env, cfg, arch="tcn", algo="ppo", device=device, env_fn=env_fn)
    vecenv_bytes = _cgroup_memory_bytes()

    # n_steps is split by n_envs inside build_agent (n_steps // n_envs per env).
    rollout_steps_per_env = _PROD_N_STEPS // n_envs
    nominal_env_steps = n_envs * rollout_steps_per_env * iterations

    # Time PPO's gradient phase separately from the env rollout phase, so the
    # reported FPS can be split into "how fast can the envs step" and "how much
    # of the budget the n_epochs=15 update loop costs". SB3 calls ``self.train()``
    # from inside ``learn()``, so shadowing it on the instance times that phase.
    update_timing = {"update_seconds": 0.0, "update_calls": 0}
    original_train = model.train

    def _timed_train(*args: Any, **kwargs: Any) -> Any:
        train_started = time.monotonic()
        try:
            return original_train(*args, **kwargs)
        finally:
            if device == "cuda":
                torch.cuda.synchronize()
            update_timing["update_seconds"] += time.monotonic() - train_started
            update_timing["update_calls"] += 1

    model.train = _timed_train

    started = time.monotonic()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    model.train = original_train

    # SB3 counts every per-env step into ``num_timesteps``, so this is the real
    # number of env steps the rollout produced (>= total_timesteps because the
    # last rollout overshoots).
    actual_env_steps = int(model.num_timesteps)
    update_seconds = float(update_timing["update_seconds"])
    rollout_seconds = max(training_seconds - update_seconds, 0.0)

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
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "rollout_steps_per_env": rollout_steps_per_env,
        "n_epochs": cfg.ppo.n_epochs,
        "batch_size": cfg.ppo.batch_size,
        "n_steps": cfg.ppo.n_steps,
        "iterations": iterations,
        "device": device,
        "bundle_size_bytes": sum(
            path.stat().st_size for path in bundle_path.iterdir() if path.is_file()
        ),
        "cgroup_start_bytes": start_bytes,
        "cgroup_vecenv_ready_bytes": vecenv_bytes,
        "cgroup_after_training_bytes": _cgroup_memory_bytes(),
        "trainer_tree_pss_mib": _tree_pss_mib(os.getpid()),
        "training_seconds": training_seconds,
        "env_steps_per_second": actual_env_steps / max(training_seconds, 1e-9),
        "effective_total_timesteps": total_timesteps,
        "actual_env_steps": actual_env_steps,
        "nominal_env_steps_per_rollout_window": nominal_env_steps,
        "update_seconds": update_seconds,
        "update_calls": update_timing["update_calls"],
        "rollout_seconds": rollout_seconds,
        "rollout_steps_per_second": actual_env_steps / max(rollout_seconds, 1e-9),
        "update_share_of_wallclock": update_seconds / max(training_seconds, 1e-9),
        **gpu_before,
        **gpu_after,
    }
    model.get_env().close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = _parse_args()
    if (
        args.n_envs < 1
        or args.rollout_steps_per_env < 1
        or args.iterations < 1
        or args.total_timesteps < 1
    ):
        raise SystemExit("n_envs, rollout steps, iterations, and total_timesteps must be positive")
    report = _train_probe(
        args.n_envs,
        args.total_timesteps,
        args.iterations,
        args.device,
        args.output,
        args.bundle_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
