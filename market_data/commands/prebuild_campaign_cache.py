#!/usr/bin/env python3
"""Build the shared bar cache once before concurrent thesis workers start."""

from __future__ import annotations

import json
from pathlib import Path

from configuration import load_config
from market_data.loading.pipeline import BAR_CACHE_VERSION, build_tick_books, run_pipeline
from market_data.loading.ticks import TICK_CACHE_VERSION


def main() -> None:
    cfg = load_config()
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
    tick_cache_paths = [
        cache_dir
        / (str(dataset_name) if dataset_name is not None else "")
        / f"{symbol}_ticks_{TICK_CACHE_VERSION}.parquet"
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
