#!/usr/bin/env python3
"""Build the shared bar cache once before concurrent thesis workers start."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from configuration import load_config
from market_data.loading.pipeline import BAR_CACHE_VERSION, build_tick_books, run_pipeline
from market_data.loading.ticks import TICK_CACHE_VERSION
from market_data.transforms.price_grid import (
    ROUNDING_MODE,
    configured_price_grid_increment,
    tick_cache_filename,
)
from training.configuration.variant_resolution import resolve_variant


def _campaign_data_overrides() -> list[str]:
    """Use the same data transform and cache policy as every campaign worker."""
    spec_value = os.environ.get("CAMPAIGN_SPEC")
    if not spec_value:
        return []
    spec_path = Path(spec_value)
    if not spec_path.is_file():
        raise FileNotFoundError(f"campaign spec is missing: {spec_path}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("variants"), list):
        raise ValueError("campaign spec must contain a variants list")
    data_prefixes = ("data.cache_dir=", "data.price_discretization.")
    policies: list[tuple[str, ...]] = []
    for name in spec["variants"]:
        resolved = resolve_variant(
            str(name),
            steps=int(spec.get("timesteps", 100_000)),
            seeds=[int(seed) for seed in spec.get("seeds", [42])],
        )
        policies.append(
            tuple(
                override for override in resolved["overrides"] if override.startswith(data_prefixes)
            )
        )
    if policies and any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("campaign variants must share one data transform and cache policy")
    return list(policies[0]) if policies else []


def main() -> None:
    cfg = load_config(overrides=_campaign_data_overrides())
    cache_dir = Path(str(cfg.data.cache_dir))
    if cache_dir.is_symlink():
        raise RuntimeError(f"campaign bar cache must be a real directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise RuntimeError(f"campaign bar cache must be a real directory: {cache_dir}")

    data = run_pipeline(cfg, force=True)
    configured_sets = cfg.data.get("datasets")
    cache_sets = list(configured_sets) if configured_sets else [None]
    expected = [
        cache_dir
        / (str(dataset_name) if dataset_name is not None else "")
        / f"{symbol}_{timeframe}_{BAR_CACHE_VERSION}.parquet"
        for dataset_name in cache_sets
        for symbol in cfg.data.symbols
        for timeframe in cfg.data.timeframes
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"bar cache build left missing/empty outputs: {missing}")

    tick_books = build_tick_books(cfg, force=True)
    grid_increment = configured_price_grid_increment(cfg)
    tick_cache_paths = [
        cache_dir
        / (str(dataset_name) if dataset_name is not None else "")
        / tick_cache_filename(symbol, grid_increment, TICK_CACHE_VERSION)
        for dataset_name in cache_sets
        for symbol in cfg.data.symbols
    ]
    missing_ticks = [
        str(path) for path in tick_cache_paths if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_ticks:
        raise RuntimeError(f"tick cache build left missing/empty outputs: {missing_ticks}")

    summary = {
        "cache_dir": str(cache_dir.resolve()),
        "price_discretization": {
            "enabled": grid_increment is not None,
            "increment": grid_increment,
            "rounding": ROUNDING_MODE,
        },
        "cache_files": len(expected),
        "tick_cache_files": len(tick_cache_paths),
        "bars": {
            symbol: {timeframe: len(frame) for timeframe, frame in frames.items()}
            for symbol, frames in data.items()
        },
        "ticks": {
            symbol: len(book) if book is not None else 0 for symbol, book in tick_books.items()
        },
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
