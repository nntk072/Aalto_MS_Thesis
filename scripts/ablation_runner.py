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


def run_experiments(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    rl_cfg: Any,
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    seeds: list[int] | None = None,
    out_dir: str | Path = "results/ablations",
    scorer: Any = None,
) -> list[dict[str, Any]]:
    """Train and score every variant in ``spec``, writing one report each.

    Args:
        bars: OHLCV DataFrame for the experiment window.
        features: Feature matrix aligned with ``bars``.
        rl_cfg: RL config containing the ``ppo``/``sac`` blocks.
        spec: Parsed experiments.yaml content (``defaults`` + ``variants``).
        steps: Override for per-variant training steps.
        seeds: Override for the seed list.
        out_dir: Directory receiving one JSON report per variant.
        scorer: Injectable replacement for ``train_and_score`` (testing).

    Returns:
        The aggregated per-variant reports, in specification order.
    """
    scorer_fn = scorer or train_and_score
    defaults = spec.get("defaults", {})
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_steps = steps if steps is not None else int(defaults.get("steps", 50_000))
    seed_list = list(seeds if seeds is not None else defaults.get("seeds", [42]))

    aggregated_reports: list[dict[str, Any]] = []
    for variant in spec.get("variants", []):
        name = str(variant["name"])
        algo = str(variant.get("algo", defaults.get("algo", "ppo")))
        arch = str(variant.get("arch", defaults.get("arch", "gru")))
        use_vae = int(variant.get("use_vae", defaults.get("use_vae", 0)))
        print(f"=== {name}: {algo} + {arch} + vae{use_vae} for {n_steps} steps ===")

        seed_reports = []
        for seed in seed_list:
            report, _, _ = scorer_fn(
                bars,
                features,
                rl_cfg,
                algo=algo,
                arch=arch,
                use_vae=bool(use_vae),
                steps=n_steps,
                seed=int(seed),
            )
            seed_reports.append(report)

        averaged: dict[str, float] = {
            key: round(float(sum(r[key] for r in seed_reports)) / len(seed_reports), 4)
            for key in seed_reports[0]
        }
        aggregated: dict[str, Any] = dict(averaged)
        aggregated.update({"name": name, "n_seeds": len(seed_list), "per_seed": seed_reports})
        (out_root / f"{name}.json").write_text(json.dumps(aggregated, indent=2))
        print(json.dumps({k: v for k, v in aggregated.items() if k != "per_seed"}, indent=2))
        aggregated_reports.append(aggregated)

    print(f"ablation reports saved under {out_root}")
    return aggregated_reports


def main() -> None:
    """Parse CLI arguments and dispatch to :func:`run_experiments`."""
    args = parse_args()
    spec: dict[str, Any] = yaml.safe_load(Path(args.experiments).read_text())
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    rl_cfg = load_config(args.rl_config)
    run_experiments(
        bars,
        features,
        rl_cfg,
        spec,
        steps=args.steps,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
