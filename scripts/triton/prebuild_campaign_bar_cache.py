#!/usr/bin/env python3
"""Build the shared bar cache once before concurrent thesis workers start."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_rl.config import load_config
from quant_rl.data.pipeline import BAR_CACHE_VERSION, run_pipeline


def main() -> None:
    cfg = load_config()
    cache_dir = Path(str(cfg.data.cache_dir))
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError(f"campaign bar cache must be a real directory: {cache_dir}")

    data = run_pipeline(cfg)
    expected = [
        cache_dir / f"{symbol}_{timeframe}_{BAR_CACHE_VERSION}.parquet"
        for symbol in cfg.data.symbols
        for timeframe in cfg.data.timeframes
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"bar cache build left missing/empty outputs: {missing}")

    summary = {
        "cache_dir": str(cache_dir.resolve()),
        "cache_files": len(expected),
        "bars": {
            symbol: {timeframe: len(frame) for timeframe, frame in frames.items()}
            for symbol, frames in data.items()
        },
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
