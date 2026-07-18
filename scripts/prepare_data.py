"""scripts/prepare_data.py – one-shot CLI: raw CSVs → cleaned parquet → feature cache.

Usage
-----
    cd Aalto_MS_Thesis
    python scripts/prepare_data.py
    python scripts/prepare_data.py data.cache_dir=my_cache features.window=60 --force
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.features.build import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data: raw CSV → parquet → features")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    parser.add_argument("--force", action="store_true", help="Ignore cached parquet files")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    log.info("Config loaded. raw_dir=%s  cache_dir=%s", cfg.data.raw_dir, cfg.data.cache_dir)

    # --- Step 1: data pipeline ---
    log.info("Running data pipeline …")
    data = run_pipeline(cfg, force=args.force)
    for sym, tfs in data.items():
        for tf, df in tfs.items():
            log.info("  %s/%s: %d bars", sym, tf, len(df))

    # --- Step 2: feature build for primary symbol (M1) ---
    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary

    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"

    log.info("Building features (cache=%s) …", feat_cache)
    features = build_features(
        primary_m1,
        secondary=secondary_m1,
        cfg=cfg,
        cache_path=feat_cache,
        force=args.force,
    )
    log.info("Features shape: %s", features.shape)
    log.info("Done. Run 'python -m quant_rl.train.run_baselines' to test baselines.")


if __name__ == "__main__":
    main()
