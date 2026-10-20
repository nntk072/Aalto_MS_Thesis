"""Prepare canonical market-data caches from configured raw inputs."""

from __future__ import annotations

import argparse
import logging

from configuration import load_config
from market_data.loading.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw CSV data into canonical bar caches")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--force", action="store_true", help="Ignore cached parquet files")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    log.info("Config loaded. raw_dir=%s cache_dir=%s", cfg.data.raw_dir, cfg.data.cache_dir)
    data = run_pipeline(cfg, force=args.force)
    for symbol, timeframes in data.items():
        for timeframe, frame in timeframes.items():
            log.info("%s/%s: %d bars", symbol, timeframe, len(frame))
    log.info("Market-data preparation complete")


if __name__ == "__main__":
    main()
