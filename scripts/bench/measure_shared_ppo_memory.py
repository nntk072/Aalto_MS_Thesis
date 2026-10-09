"""Measure concurrent CPU PPO processes using the same read-only env bundle."""

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

from quant_rl.envs.env_bundle_builder import build_env_bundle
from quant_rl.envs.env_spec import EnvSpec
from quant_rl.envs.worker import make_bundle_env
from quant_rl.models.agent import build_agent
from quant_rl.train.train_rl import _strategy_from_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--rollout-steps-per-env", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prepare_bundle() -> tuple[Any, EnvSpec, Path]:
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


def _train_probe(n_envs: int, rollout_steps: int, output: Path) -> dict[str, Any]:
    import torch

    cfg, spec, bundle_path = _prepare_bundle()
    cfg.env.n_envs = n_envs
    cfg.env.data_source = "bundle"
    cfg.ppo.n_steps = n_envs * rollout_steps
    cfg.ppo.batch_size = min(256, n_envs * rollout_steps)
    cfg.ppo.n_epochs = 1
    torch.set_num_threads(1)

    start_bytes = _cgroup_memory_bytes()
    parent_env = make_bundle_env(spec)
    env_fn = partial(make_bundle_env, spec)
    model = build_agent(parent_env, cfg, arch="tcn", algo="ppo", device="cpu", env_fn=env_fn)
    vecenv_bytes = _cgroup_memory_bytes()
    training_steps = n_envs * rollout_steps
    started = time.monotonic()
    model.learn(total_timesteps=training_steps, progress_bar=False)
    training_seconds = time.monotonic() - started
    report = {
        "n_envs": n_envs,
        "training_steps": training_steps,
        "rollout_steps_per_env": rollout_steps,
        "bundle_size_bytes": sum(
            path.stat().st_size for path in bundle_path.iterdir() if path.is_file()
        ),
        "cgroup_start_bytes": start_bytes,
        "cgroup_vecenv_ready_bytes": vecenv_bytes,
        "cgroup_after_training_bytes": _cgroup_memory_bytes(),
        "trainer_tree_pss_mib": _tree_pss_mib(os.getpid()),
        "training_seconds": training_seconds,
    }
    model.get_env().close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = _parse_args()
    if args.n_envs < 1 or args.rollout_steps_per_env < 1:
        raise SystemExit("environment count and rollout steps must be positive")
    report = _train_probe(args.n_envs, args.rollout_steps_per_env, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
