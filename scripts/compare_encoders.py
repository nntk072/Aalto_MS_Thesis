"""Compare GRU vs Transformer encoders on the same data and step budget.

Trains a short-horizon RL run per architecture, evaluates each through the
shared evaluation pipeline and reports Sharpe plus Sweep Delay — the metric
that decides whether the Transformer replaces the GRU (delay > 3s rule).

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

from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import run_episode, sweep_delay_breakdown  # noqa: E402
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
    parser.add_argument("--out", default="results/encoder_comparison.json")
    return parser.parse_args()


def main() -> None:
    """Train one agent per encoder and compare evaluation results."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    if args.features_csv:
        features = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    else:
        features = bars.select_dtypes(include=["number"])
    loaded = OmegaConf.load(args.config)
    cfg = OmegaConf.create(loaded) if not isinstance(loaded, OmegaConf) else loaded
    assert isinstance(cfg, DictConfig)

    results: dict[str, dict[str, float]] = {}
    continuous = args.algo == "sac"
    for arch in args.archs:
        print(f"=== training {args.algo.upper()} + {arch} for {args.steps} steps ===")
        env = TradingEnv(bars=bars, features=features, continuous_actions=continuous)
        model = build_agent(env, cfg, arch=arch, algo=args.algo)
        model.set_random_seed(args.seed)
        model.learn(total_timesteps=args.steps)

        eval_env = TradingEnv(bars=bars, features=features, continuous_actions=continuous)

        def action_fn(obs: dict[str, Any]) -> Any:
            return model.predict(obs, deterministic=True)[0]

        metrics = run_episode(eval_env, action_fn=action_fn)
        delays = sweep_delay_breakdown(eval_env.trade_log)

        row = {
            "sharpe": round(metrics.sharpe, 3),
            "sortino": round(metrics.sortino, 3),
            "max_drawdown": round(metrics.max_drawdown, 4),
            "total_return_pct": round(metrics.total_return_pct, 3),
            **{k: round(v, 3) for k, v in delays.items()},
        }
        results[arch] = row
        print(json.dumps(row, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    gru_delay = results.get("gru", {}).get("sweep_delay_mean_s", 0.0)
    if "gru" in results and "transformer" in results and gru_delay > 3.0:
        print("GRU mean Sweep Delay > 3s -> prioritise the Transformer encoder.")
    print(f"saved comparison to {out_path}")


if __name__ == "__main__":
    main()
