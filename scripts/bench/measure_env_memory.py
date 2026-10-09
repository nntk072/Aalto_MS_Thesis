"""Measure baseline SubprocVecEnv memory and throughput on a fixed train slice.

The probe uses the canonical training pipeline and ``make_env`` factory. It
keeps the slice short enough for local runs while retaining the same source
frames and worker payload shape as training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv

from quant_rl.config import load_config
from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.split import get_split_config, make_train_mask, split_train_test
from quant_rl.data.ticks import TickBook, ticks_covering
from quant_rl.features.build import build_features, feature_cache_path
from quant_rl.train.train_rl import _publish_obs_memmap, _strategy_from_cfg, make_env


def _read_pss_kib(pid: int) -> int:
    """Read a process PSS value from Linux's smaps rollup."""
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    return 0


def _descendant_pids(pid: int) -> set[int]:
    """Return *pid* and every process in its descendant tree."""
    found = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for child in (int(value) for value in children.split()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _tree_pss_mib(pid: int) -> float:
    """Sum PSS for a process and all live descendants, in MiB."""
    return sum(_read_pss_kib(child) for child in _descendant_pids(pid)) / 1024.0


def _cgroup_memory_bytes() -> int | None:
    """Read cgroup v2/v1 memory usage when available."""
    root = Path("/sys/fs/cgroup")
    paths: list[Path] = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", maxsplit=2)
            if hierarchy == "0" and not controllers:
                paths.append(root / relative.lstrip("/") / "memory.current")
            elif "memory" in controllers.split(","):
                paths.append(root / "memory" / relative.lstrip("/") / "memory.usage_in_bytes")
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    paths.extend((root / "memory.current", root / "memory/memory.usage_in_bytes"))
    for path in paths:
        try:
            return int(path.read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return None


def _load_train_slice(days: int | None) -> tuple[Any, pd.DataFrame, pd.DataFrame, TickBook | None]:
    """Load the canonical train split, optionally taking its last *days* of bars."""
    cfg = load_config()
    data = run_pipeline(cfg)
    primary = str(cfg.data.primary)
    secondary = str(cfg.data.secondary)
    primary_bars = data[primary]["M1"]
    secondary_bars = data.get(secondary, {}).get("M1")
    train_end, test_start = get_split_config(cfg)
    train_mask = make_train_mask(cast(pd.DatetimeIndex, primary_bars.index), train_end)
    cache_path = feature_cache_path(
        Path(cfg.data.cache_dir), primary, cfg, primary_bars, train_mask=train_mask
    )
    features = build_features(
        primary_bars,
        secondary=secondary_bars,
        cfg=cfg,
        train_mask=train_mask,
        cache_path=cache_path,
    )
    train_bars, _, train_features, _ = split_train_test(
        primary_bars, features, train_end, test_start
    )
    if days is not None:
        slice_start = pd.Timestamp(train_bars.index[-1]) - pd.Timedelta(days=days)
        train_bars = train_bars.loc[train_bars.index >= slice_start]
        train_features = train_features.loc[train_bars.index]
    tick_books = build_tick_books(cfg)
    tickbook = ticks_covering(tick_books.get(primary), train_bars)
    del data, primary_bars, secondary_bars, features, tick_books
    return cfg, train_bars, train_features, tickbook


def _memory_report(bars: pd.DataFrame, features: pd.DataFrame, env_fn: Any) -> dict[str, Any]:
    """Return source-frame and serialized-payload sizes."""
    tickbook = getattr(env_fn, "keywords", {}).get("tickbook")
    tick_bytes = 0
    if tickbook is not None:
        tick_bytes = sum(int(getattr(tickbook, name).nbytes) for name in ("_ts", "_bid", "_ask"))
    return {
        "bars_rows": len(bars),
        "bars_deep_bytes": int(bars.memory_usage(deep=True).sum()),
        "features_rows": len(features),
        "features_deep_bytes": int(features.memory_usage(deep=True).sum()),
        "tickbook_array_bytes": tick_bytes,
        "env_payload_pickle_bytes": len(pickle.dumps(env_fn)),
    }


def _env_attribute_sizes(env: Any) -> dict[str, int]:
    """Report pandas and numpy storage held directly by one TradingEnv."""
    sizes: dict[str, int] = {}
    for name, value in vars(env).items():
        if isinstance(value, pd.DataFrame):
            sizes[name] = int(value.memory_usage(deep=True).sum())
        elif isinstance(value, np.ndarray):
            sizes[name] = int(value.nbytes)
        elif name == "_tickbook" and value is not None:
            sizes[name] = sum(
                int(getattr(value, array_name).nbytes) for array_name in ("_ts", "_bid", "_ask")
            )
        elif name == "pre_ny_by_date" and value:
            sizes[name] = sum(int(array.nbytes) for array in value.values())
    return sizes


def _measure_env_count(
    cfg: Any,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    tickbook: TickBook | None,
    n_envs: int,
    steps: int,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Start one vector env, measure PSS, and time fixed vector steps."""
    import torch

    torch.set_num_threads(1)
    cfg.env.n_envs = n_envs
    with tempfile.TemporaryDirectory(prefix="rl-mem-probe-") as temp_dir:
        if bundle_path is None:
            mmap_path = _publish_obs_memmap(
                features, bars, cfg, Path(temp_dir) / "obs_features.npy"
            )
            env_fn = partial(
                make_env,
                bars,
                features,
                cfg,
                algo="ppo",
                reward="dsr",
                tickbook=tickbook,
                obs_features_mmap=mmap_path,
            )
        else:
            from quant_rl.envs.env_spec import EnvSpec
            from quant_rl.envs.worker import make_bundle_env

            spec = EnvSpec.from_cfg(
                cfg,
                algo="ppo",
                reward="dsr",
                arch="tcn",
                bundle_dir=str(bundle_path),
            )
            env_fn = partial(make_bundle_env, spec)
            if "forkserver" in mp.get_all_start_methods():
                mp.set_forkserver_preload(
                    [
                        "stable_baselines3.common.vec_env.subproc_vec_env",
                        "quant_rl.envs.worker",
                        "quant_rl.envs.trading_env",
                    ]
                )
        payload = _memory_report(bars, features, env_fn)
        parent_env = env_fn()
        env_attribute_sizes = _env_attribute_sizes(parent_env)
        cgroup_start = _cgroup_memory_bytes()
        started = time.monotonic()
        vec_env = SubprocVecEnv([env_fn] * n_envs)
        vec_env.reset()
        warmup_pss_mib = _tree_pss_mib(os.getpid())
        for _ in range(10):
            actions = np.asarray([vec_env.action_space.sample() for _ in range(n_envs)])
            vec_env.step(actions)
        steady_pss_mib = _tree_pss_mib(os.getpid())
        cgroup_warmup = _cgroup_memory_bytes()
        start = time.monotonic()
        for _ in range(steps):
            actions = np.asarray([vec_env.action_space.sample() for _ in range(n_envs)])
            vec_env.step(actions)
        elapsed = time.monotonic() - start
        final_pss_mib = _tree_pss_mib(os.getpid())
        cgroup_steady = _cgroup_memory_bytes()
        vec_env.close()
        startup_seconds = time.monotonic() - started
    return {
        "n_envs": n_envs,
        **payload,
        "parent_env_attribute_bytes": env_attribute_sizes,
        "cgroup_start_bytes": cgroup_start,
        "warmup_tree_pss_mib": warmup_pss_mib,
        "steady_tree_pss_mib": steady_pss_mib,
        "final_tree_pss_mib": final_pss_mib,
        "cgroup_warmup_bytes": cgroup_warmup,
        "cgroup_steady_bytes": cgroup_steady,
        "vector_steps": steps,
        "env_steps_per_second": n_envs * steps / elapsed,
        "startup_seconds": startup_seconds,
    }


def _fit_worker_pss(results: list[dict[str, Any]]) -> tuple[float, float]:
    """Fit total steady PSS to an intercept plus a per-worker slope."""
    counts = np.asarray([item["n_envs"] for item in results], dtype=np.float64)
    pss = np.asarray([item["steady_tree_pss_mib"] for item in results], dtype=np.float64)
    slope, intercept = np.polyfit(counts, pss, 1)
    return float(intercept), float(slope)


def _parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--full-train", action="store_true", help="measure the complete train split"
    )
    parser.add_argument("--env-counts", nargs="+", type=int, default=[2, 8, 16])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--data-source", choices=["frames", "bundle"], default="frames")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the requested baseline worker counts and save JSON results."""
    args = _parse_args()
    if (
        (not args.full_train and args.days < 1)
        or args.steps < 1
        or any(count < 1 for count in args.env_counts)
    ):
        raise SystemExit("days, steps, and environment counts must be positive")
    cfg, bars, features, tickbook = _load_train_slice(None if args.full_train else args.days)
    bundle_path = None
    if args.data_source == "bundle":
        from quant_rl.envs.env_bundle_builder import build_env_bundle

        strategy = _strategy_from_cfg(cfg)[0]
        bundle_path = build_env_bundle(
            cfg.data.cache_dir,
            bars,
            features,
            cfg,
            raw_columns=list(strategy.raw_columns),
            feature_config_hash="phase0-probe",
            tickbook=tickbook,
            pre_ny_by_date=None,
        )
    results = [
        _measure_env_count(cfg, bars, features, tickbook, count, args.steps, bundle_path)
        for count in args.env_counts
    ]
    base_pss_mib, worker_pss_mib = _fit_worker_pss(results)
    report = {
        "data_source": args.data_source,
        "slice_days": None if args.full_train else args.days,
        "full_train_split": args.full_train,
        "bundle_size_bytes": (
            sum(path.stat().st_size for path in bundle_path.iterdir() if path.is_file())
            if bundle_path is not None
            else None
        ),
        "sb3_start_method": (
            "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ),
        "env_fn_pickled_per_worker": True,
        "fit_base_pss_mib": base_pss_mib,
        "fit_per_worker_pss_mib": worker_pss_mib,
        "measurements": results,
    }
    output = args.output or Path(f"outputs/mem_{args.data_source}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
