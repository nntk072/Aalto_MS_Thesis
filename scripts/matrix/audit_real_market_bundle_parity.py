#!/usr/bin/env python3
"""Compare frames and shared-bundle TradingEnv traces on the real test calendar."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from quant_rl.config import load_config
from quant_rl.data.pipeline import build_tick_books, run_pipeline
from quant_rl.data.split import get_split_config, make_train_mask, split_train_test
from quant_rl.envs.env_bundle_builder import build_env_bundle
from quant_rl.envs.env_spec import EnvSpec
from quant_rl.envs.worker import make_bundle_env
from quant_rl.features.build import build_features, feature_cache_path
from quant_rl.train.train_rl import (
    _load_merged_config,
    _publish_obs_memmap,
    _strategy_from_cfg,
    make_env,
)
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_input_hashes(cfg: Any) -> dict[str, str]:
    """Hash configured M1 and tick inputs directly, without campaign artifacts."""
    raw_dir = (ROOT / str(cfg.data.raw_dir)).resolve()
    filenames: set[str] = set()
    for field in ("m1_files", "tick_files"):
        configured = getattr(cfg.data, field, {})
        if configured is not None:
            filenames.update(str(value) for value in configured.values() if value)
    results: dict[str, str] = {}
    for filename in sorted(filenames):
        path = (raw_dir / filename).resolve()
        try:
            relative_path = path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"Configured market input escapes repository root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Configured market input not found: {relative_path}")
        results[relative_path.as_posix()] = _digest(path)
    if not results:
        raise ValueError("No configured M1 or tick input files found to hash")
    return results


def _audit_source_paths(root: Path | None = None) -> list[Path]:
    """Return all project code/config inputs that may affect environment parity."""
    project_root = root or ROOT
    paths = {
        path
        for tree in (project_root / "quant_rl", project_root / "config")
        for path in tree.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    }
    paths.update(
        path
        for path in (
            project_root / "scripts" / "variant_list.txt",
            project_root / "pyproject.toml",
            project_root / "uv.lock",
            project_root / "scripts" / "matrix" / "audit_real_market_bundle_parity.py",
        )
        if path.is_file()
    )
    return sorted(paths)


def _repository_manifest() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    )
    dirty_paths = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    return {
        "head": head,
        "branch": branch,
        "working_tree_clean": not dirty_paths,
        "working_tree_dirty_paths": dirty_paths,
    }


def _runtime_manifest() -> dict[str, Any]:
    packages = (
        "gymnasium",
        "numpy",
        "omegaconf",
        "pandas",
        "pyyaml",
        "stable-baselines3",
        "torch",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def _compare(left: Any, right: Any, path: str = "root", *, atol: float = 1e-12) -> float:
    """Raise on the first mismatch and return the largest scalar difference."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            raise AssertionError(f"{path}: shape {a.shape} != {b.shape}")
        if a.dtype.kind in "f" or b.dtype.kind in "f":
            if not np.array_equal(a, b, equal_nan=True):
                both = np.isfinite(a) & np.isfinite(b)
                delta = float(np.max(np.abs(a[both] - b[both]))) if both.any() else math.inf
                raise AssertionError(f"{path}: arrays differ; max_abs_delta={delta}")
        elif not np.array_equal(a, b):
            raise AssertionError(f"{path}: arrays differ")
        return 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: key set differs ({left.keys()} != {right.keys()})")
        return max(
            (_compare(left[key], right[key], f"{path}.{key}", atol=atol) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length {len(left)} != {len(right)}")
        return max(
            (
                _compare(a, b, f"{path}[{index}]", atol=atol)
                for index, (a, b) in enumerate(zip(left, right))
            ),
            default=0.0,
        )
    if isinstance(left, (pd.Timestamp, np.datetime64)) or isinstance(
        right, (pd.Timestamp, np.datetime64)
    ):
        left_timestamp = pd.Timestamp(left)
        right_timestamp = pd.Timestamp(right)
        if left_timestamp.tzinfo is None or right_timestamp.tzinfo is None:
            if left_timestamp != right_timestamp:
                raise AssertionError(f"{path}: {left_timestamp!r} != {right_timestamp!r}")
        elif left_timestamp.tz_convert("UTC").value != right_timestamp.tz_convert("UTC").value:
            raise AssertionError(f"{path}: {left_timestamp!r} != {right_timestamp!r}")
        return 0.0
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        left_number = float(left)
        right_number = float(right)
        if math.isnan(left_number) and math.isnan(right_number):
            return 0.0
        delta = abs(left_number - right_number)
        if not math.isfinite(delta) or delta > atol:
            raise AssertionError(f"{path}: {left_number!r} != {right_number!r}; abs_delta={delta}")
        return delta
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")
    return 0.0


def _active_action(env: Any, step: int) -> Any:
    space = env.action_space
    if not hasattr(space, "low"):
        return space.sample()
    action = np.zeros(space.shape, dtype=space.dtype)
    schema = getattr(env, "action_schema", None)
    if schema is None:
        return action
    indices = schema.indices
    action[indices["intensity"]] = 1.0
    action[indices["risk"]] = 0.6
    if "direction" in indices:
        action[indices["direction"]] = 1.0 if (step // 32) % 2 == 0 else -1.0
    if "exit" in indices:
        action[indices["exit"]] = -1.0
    for name in ("target", "tp1_sel", "tp2_sel", "tp3_sel"):
        if name in indices:
            action[indices[name]] = 0.5
    return action


def _trace_pair(
    frames: Any,
    shared: Any,
    *,
    name: str,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    left_obs, _ = frames.reset(seed=seed)
    right_obs, _ = shared.reset(seed=seed)
    maximum_delta = _compare(left_obs, right_obs, f"{name}.reset_obs")
    opened_before = len(frames.trade_log)
    compared = 0
    terminated = truncated = False
    for index in range(max_steps):
        action = (
            _active_action(frames, index)
            if name == "active"
            else np.zeros(frames.action_space.shape, dtype=frames.action_space.dtype)
        )
        left = frames.step(action)
        right = shared.step(action)
        maximum_delta = max(maximum_delta, _compare(left[0], right[0], f"{name}[{index}].obs"))
        maximum_delta = max(maximum_delta, _compare(left[1], right[1], f"{name}[{index}].reward"))
        _compare(left[2:4], right[2:4], f"{name}[{index}].done")
        maximum_delta = max(maximum_delta, _compare(left[4], right[4], f"{name}[{index}].info"))
        compared += 1
        left_obs, right_obs = left[0], right[0]
        terminated, truncated = bool(left[2]), bool(left[3])
        if terminated or truncated:
            break
    del left_obs, right_obs
    opens = sum(record.get("type") == "open" for record in frames.trade_log[opened_before:])
    for attr in (
        "trade_log",
        "equity_curve",
        "equity_times",
        "pnl_history",
        "breach_events",
        "_pending_entry_order",
        "episode_step_count",
        "step_idx",
    ):
        maximum_delta = max(
            maximum_delta,
            _compare(getattr(frames, attr), getattr(shared, attr), f"{name}.final.{attr}"),
        )
    for attr in ("account", "position"):
        left_obj, right_obj = getattr(frames, attr), getattr(shared, attr)
        left_value = asdict(cast(Any, left_obj)) if is_dataclass(left_obj) else left_obj
        right_value = asdict(cast(Any, right_obj)) if is_dataclass(right_obj) else right_obj
        maximum_delta = max(
            maximum_delta,
            _compare(left_value, right_value, f"{name}.final.{attr}"),
        )
    return {
        "trace": name,
        "steps_compared": compared,
        "terminated": terminated,
        "truncated": truncated,
        "opened_trades": int(opens),
        "trade_log_rows": len(frames.trade_log) - opened_before,
        "max_scalar_delta": maximum_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle-cache", type=Path, required=True)
    parser.add_argument("--active-bars", type=int, default=10_000)
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out.with_suffix(".progress.json")

    variants = [
        line.strip()
        for line in (ROOT / "scripts" / "variant_list.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    base_cfg = load_config(config_path="config/features/full_po3_mtf.yaml")
    market = run_pipeline(base_cfg)
    primary_symbol = str(base_cfg.data.primary)
    secondary_symbol = str(base_cfg.data.secondary)
    bars = market[primary_symbol]["M1"]
    secondary = market.get(secondary_symbol, {}).get("M1")
    train_end, test_start = get_split_config(base_cfg)
    train_mask = make_train_mask(cast(pd.DatetimeIndex, bars.index), train_end)
    _, test_bars_full, _, _ = split_train_test(bars, bars, train_end, test_start)
    tickbook = build_tick_books(base_cfg).get(primary_symbol)
    source_paths = _audit_source_paths()
    source_hashes = {path.relative_to(ROOT).as_posix(): _digest(path) for path in source_paths}
    raw_hashes = _raw_input_hashes(base_cfg)
    repository = _repository_manifest()
    runtime = _runtime_manifest()
    results: list[dict[str, Any]] = []

    for name in variants:
        item: dict[str, Any] = {"variant": name, "status": "running"}
        results.append(item)
        progress = {
            "repository": repository,
            "runtime": runtime,
            "source_hashes": source_hashes,
            "raw_data_sha256": raw_hashes,
            "calendar": {
                "rows": int(len(test_bars_full)),
                "start": str(test_bars_full.index[0]),
                "end": str(test_bars_full.index[-1]),
                "train_end": train_end,
                "test_start": test_start,
            },
            "variants": results,
        }
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        try:
            resolved = resolve_variant(name, steps=100_000, seeds=[42])
            cfg = _load_merged_config(
                argparse.Namespace(
                    config=resolved["base_config"],
                    strategy=resolved["strategy"],
                    overrides=resolved["overrides"],
                    variant=None,
                )
            )
            cache = feature_cache_path(
                Path(cfg.data.cache_dir),
                primary_symbol,
                cfg,
                bars,
                train_mask=train_mask,
                secondary=secondary,
            )
            features = build_features(
                bars,
                secondary=secondary,
                cfg=cfg,
                train_mask=train_mask,
                cache_path=cache,
            )
            _, test_bars, _, test_features = split_train_test(bars, features, train_end, test_start)
            strategy = _strategy_from_cfg(cfg)[0]
            episode_bars = test_bars
            episode_features = test_features
            raw_ticks = None
            if tickbook is not None:
                from quant_rl.data.ticks import ticks_covering

                raw_ticks = ticks_covering(tickbook, episode_bars)
            obs_mmap_path = args.bundle_cache / "obs_mmap" / f"{name}.npy"
            obs_mmap = _publish_obs_memmap(
                episode_features,
                episode_bars,
                cfg,
                obs_mmap_path,
            )
            frames = make_env(
                episode_bars,
                episode_features,
                cfg,
                algo=str(resolved["algo"]),
                reward=str(resolved["reward_mode"]),
                episodic=False,
                tickbook=raw_ticks,
                obs_features_mmap=obs_mmap,
                arch=str(resolved["arch"]),
            )
            bundle_path = build_env_bundle(
                args.bundle_cache.resolve(),
                episode_bars,
                episode_features,
                cfg,
                raw_columns=list(strategy.raw_columns),
                feature_config_hash=cache.stem,
                tickbook=raw_ticks,
                pre_ny_by_date=None,
            )
            spec = EnvSpec.from_cfg(
                cfg,
                algo=str(resolved["algo"]),
                reward=str(resolved["reward_mode"]),
                arch=str(resolved["arch"]),
                bundle_dir=str(bundle_path),
                episodic=False,
            )
            shared = make_bundle_env(spec)
            action_schema = getattr(frames, "action_schema", None)
            item.update(
                status="passed",
                feature_cache=str(cache.resolve().relative_to(ROOT)),
                feature_cache_sha256=_digest(cache),
                feature_columns=int(features.shape[1]),
                bar_rows=int(len(episode_bars)),
                action_schema=action_schema.to_dict() if action_schema else None,
                observation_space={
                    key: list(value.shape)
                    for key, value in getattr(frames.observation_space, "spaces", {}).items()
                },
                bundle_key=bundle_path.name,
                traces=[],
            )
            full_steps = max(0, len(episode_bars) - 1)
            item["traces"].append(
                _trace_pair(
                    frames,
                    shared,
                    name="full_calendar_zero_action",
                    max_steps=full_steps,
                    seed=41,
                )
            )
            active_steps = min(max(0, args.active_bars), max(0, full_steps))
            if active_steps:
                item["traces"].append(
                    _trace_pair(
                        frames,
                        shared,
                        name="active",
                        max_steps=active_steps,
                        seed=73,
                    )
                )
            item["status"] = "passed"
            frames.close()
            shared.close()
            del frames, shared, episode_bars, episode_features, test_bars, test_features, features
        except (
            Exception
        ) as exc:  # keep the full audit moving and preserve first failure per variant
            item.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"{name}: {item['status']}", flush=True)

    final = {
        "repository": repository,
        "runtime": runtime,
        "source_hashes": source_hashes,
        "raw_data_sha256": raw_hashes,
        "calendar": {
            "rows": int(len(test_bars_full)),
            "start": str(test_bars_full.index[0]),
            "end": str(test_bars_full.index[-1]),
            "train_end": train_end,
            "test_start": test_start,
        },
        "active_action_trace_bars": args.active_bars,
        "summary": {
            "passed": sum(item["status"] == "passed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "variants": len(results),
        },
        "variants": results,
    }
    out.write_text(json.dumps(final, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return int(final["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
