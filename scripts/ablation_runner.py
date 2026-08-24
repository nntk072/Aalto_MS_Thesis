"""Run the ablation experiment matrix from config/experiments.yaml.

Example:
    python scripts/ablation_runner.py --bars-csv data/us100_2025.csv \
        --config config/experiments.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.validation.fold_runner import load_config, train_and_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--features-csv")
    parser.add_argument("--rl-config", default="quant_rl/config/default.yaml")
    parser.add_argument("--experiments", default="config/experiments.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Override per-variant steps")
    parser.add_argument("--out-dir", default="results/ablations")
    return parser.parse_args()


def main() -> None:
    """Train and score every variant, writing one report each."""
    args = parse_args()
    spec: dict[str, Any] = yaml.safe_load(Path(args.experiments).read_text())
    defaults = spec.get("defaults", {})
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    rl_cfg = load_config(args.rl_config)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    steps = args.steps if args.steps is not None else int(defaults.get("steps", 50_000))
    seeds = list(defaults.get("seeds", [42]))

    for variant in spec.get("variants", []):
        name = str(variant["name"])
        algo = str(variant.get("algo", defaults.get("algo", "ppo")))
        arch = str(variant.get("arch", defaults.get("arch", "gru")))
        use_vae = int(variant.get("use_vae", defaults.get("use_vae", 0)))
        print(f"=== {name}: {algo} + {arch} + vae{use_vae} for {steps} steps ===")

        seed_reports = []
        for seed in seeds:
            report, _, _ = train_and_score(
                bars,
                features,
                rl_cfg,
                algo=algo,
                arch=arch,
                use_vae=bool(use_vae),
                steps=steps,
                seed=int(seed),
            )
            seed_reports.append(report)

        averaged: dict[str, float] = {
            key: round(float(sum(r[key] for r in seed_reports)) / len(seed_reports), 4)
            for key in seed_reports[0]
        }
        aggregated: dict[str, Any] = dict(averaged)
        aggregated.update({"name": name, "n_seeds": len(seeds), "per_seed": seed_reports})
        (out_root / f"{name}.json").write_text(json.dumps(aggregated, indent=2))
        print(json.dumps({k: v for k, v in aggregated.items() if k != "per_seed"}, indent=2))

    print(f"ablation reports saved under {out_root}")


if __name__ == "__main__":
    main()
