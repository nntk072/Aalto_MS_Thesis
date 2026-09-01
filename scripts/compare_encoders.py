"""Compare GRU vs Transformer encoders on the same data and step budget.

Trains a short-horizon RL run per architecture on the in-sample slice
(≤ --train-end), evaluates each through the shared evaluation pipeline on
the held-out out-of-sample slice (≥ --test-start) and reports Sharpe plus
Sweep Delay — the metric that decides whether the Transformer replaces the
GRU (delay > 3s rule).

Example:
    python scripts/compare_encoders.py --bars-csv data/us100_2025.csv \
        --steps 50000 --archs gru transformer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import DictConfig, OmegaConf  # noqa: E402

from quant_rl.data.split import split_train_test  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.eval.rollout import make_action_fn  # noqa: E402
from quant_rl.evaluation import build_run_report, run_episode  # noqa: E402
from quant_rl.models.agent import build_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True, help="OHLCV CSV with DatetimeIndex")
    parser.add_argument("--features-csv", help="Feature CSV; defaults to numeric bars columns")
    parser.add_argument("--config", default="quant_rl/config/default.yaml", help="Env config YAML")
    parser.add_argument("--steps", type=int, default=50_000, help="Training steps per arch")
    parser.add_argument(
        "--archs", nargs="+", default=["gru", "transformer"], choices=["tcn", "gru", "transformer"]
    )
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-end", default="2025-12-31", help="Last in-sample date")
    parser.add_argument("--test-start", default="2026-01-01", help="First out-of-sample date")
    parser.add_argument("--out", default="results/encoder_comparison.json")
    return parser.parse_args()


def evaluate_split(
    model: Any, algo: str, bars: pd.DataFrame, features: pd.DataFrame
) -> dict[str, Any]:
    """Run one deterministic episode of *model* over *bars* and score it."""
    env = TradingEnv(
        bars=bars,
        features=features,
        continuous_actions=algo == "sac",
        # Eval mode: a breach blocks trading for the rest of the session
        # instead of truncating the episode (mirrors run_backtest).
        episodic=False,
    )
    metrics = run_episode(env, action_fn=make_action_fn(model, continuous_actions=algo == "sac"))
    return build_run_report(metrics, env.trade_log)


def main() -> None:
    """Train one agent per encoder on the train slice and compare OOS results."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    if args.features_csv:
        features = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    else:
        features = bars.select_dtypes(include=["number"])
    loaded = OmegaConf.load(args.config)
    cfg = OmegaConf.create(loaded)
    assert isinstance(cfg, DictConfig)

    train_bars, test_bars, train_features, test_features = split_train_test(
        bars, features, args.train_end, args.test_start
    )
    if train_bars.empty or test_bars.empty:
        raise SystemExit(
            f"empty split with --train-end {args.train_end} / --test-start {args.test_start}: "
            f"train={len(train_bars)} bars, test={len(test_bars)} bars"
        )
    print(f"split: train={len(train_bars)} bars, test={len(test_bars)} bars (held-out)")

    continuous = args.algo == "sac"
    results: dict[str, dict[str, Any]] = {}
    for arch in args.archs:
        print(f"=== training {args.algo.upper()} + {arch} for {args.steps} steps ===")
        env = TradingEnv(bars=train_bars, features=train_features, continuous_actions=continuous)
        model = build_agent(env, cfg, arch=arch, algo=args.algo)
        model.set_random_seed(args.seed)
        model.learn(total_timesteps=args.steps)

        row = {
            "in_sample": evaluate_split(model, args.algo, train_bars, train_features),
            "out_of_sample": evaluate_split(model, args.algo, test_bars, test_features),
        }
        results[arch] = row
        print(
            json.dumps(
                {
                    "in_sample_sharpe": row["in_sample"]["sharpe"],
                    "out_of_sample_sharpe": row["out_of_sample"]["sharpe"],
                    "oos_sweep_delay_mean_s": row["out_of_sample"]["sweep_delay"][
                        "sweep_delay_mean_s"
                    ],
                },
                indent=2,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    gru = results.get("gru", {}).get("out_of_sample", {}).get("sweep_delay", {})
    gru_delay = float(gru.get("sweep_delay_mean_s", 0.0))
    if "gru" in results and "transformer" in results and gru_delay > 3.0:
        print("GRU mean out-of-sample Sweep Delay > 3s -> prioritise the Transformer encoder.")
    print(f"saved comparison to {out_path}")


if __name__ == "__main__":
    main()
