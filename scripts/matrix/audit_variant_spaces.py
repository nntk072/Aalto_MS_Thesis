#!/usr/bin/env python3
"""Construct each registered variant on real cached bars and record Gym spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import make_train_mask
from quant_rl.features.build import build_features, feature_cache_path
from quant_rl.train.train_rl import _load_merged_config, make_env
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]


def observation_shapes(observation: Any) -> dict[str, list[int]]:
    if isinstance(observation, dict):
        return {str(key): list(np.asarray(value).shape) for key, value in observation.items()}
    return {"observation": list(np.asarray(observation).shape)}


def gym_space_shapes(space: Any) -> dict[str, list[int]]:
    children = getattr(space, "spaces", None)
    if isinstance(children, dict):
        return {str(key): list(child.shape) for key, child in children.items()}
    return {"observation": list(space.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = [
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
    train_end = str(base_cfg.data.split.train_end)
    train_mask = make_train_mask(cast(pd.DatetimeIndex, bars.index), train_end)

    results: list[dict[str, Any]] = []
    feature_sets: dict[str, Any] = {}
    for name in names:
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
        cache_key = str(cache)
        if cache_key not in feature_sets:
            feature_sets[cache_key] = build_features(
                bars,
                secondary=secondary,
                cfg=cfg,
                train_mask=train_mask,
                cache_path=cache,
            )
        features = feature_sets[cache_key]
        sample_bars = bars.iloc[-512:]
        sample_features = features.iloc[-512:]
        env = make_env(
            sample_bars,
            sample_features,
            cfg,
            algo=str(resolved["algo"]),
            reward="dsr",
            episodic=False,
            arch=str(resolved["arch"]),
        )
        observation, _ = env.reset(seed=42)
        action_shape_raw = env.action_space.shape
        if action_shape_raw is None:
            raise RuntimeError(f"{name} has an action space without a shape")
        action_shape = list(action_shape_raw)
        result = {
            "variant": name,
            "strategy": resolved["strategy"],
            "feature_profile": resolved["feature_profile"],
            "feature_cache": cache.resolve().relative_to(ROOT).as_posix(),
            "feature_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "feature_columns": int(features.shape[1]),
            "action_space_width": int(np.prod(action_shape)),
            "action_space_shape": action_shape,
            "observation_space_shapes": gym_space_shapes(env.observation_space),
            "reset_observation_shapes": observation_shapes(observation),
            "env_flags": {
                key: bool(cfg.env.get(key, False))
                for key in (
                    "strategy_actions",
                    "allow_agent_sl_mode",
                    "allow_agent_tp_mode",
                    "allow_multi_tp",
                    "allow_simplex",
                    "agent_direction_control",
                    "entry_state_machine",
                    "entry_state_observation",
                    "arm_requires_retest",
                )
            },
        }
        results.append(result)
        env.close()
        print(f"{name}: action={action_shape} observation={result['observation_space_shapes']}")

    report = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "market_rows": int(len(bars)),
        "market_start": str(bars.index[0]),
        "market_end": str(bars.index[-1]),
        "variants": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
