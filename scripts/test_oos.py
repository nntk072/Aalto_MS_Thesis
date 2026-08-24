"""Out-of-sample testing harness for the strict 2026 test set.

Loads a trained SB3 model, runs it on OOS data and reports metrics under
a grid of spread/slippage assumptions (robustness sensitivity).

Example:
    python scripts/test_oos.py --model-path models/rl_runs/<run>/model.zip \
        --bars-csv data/us100_2026.csv --features-csv data/us100_feat_2026.csv \
        --algo sac
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.backtest.costs import CostModel  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import run_episode, sweep_delay_breakdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to saved SB3 .zip")
    parser.add_argument("--bars-csv", required=True, help="OOS bars (2026)")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--algo", default="sac", choices=["ppo", "sac"])
    parser.add_argument("--spreads", nargs="+", type=float, default=[0.6, 1.0, 1.5])
    parser.add_argument("--slippages", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--out", default="results/oos_report.json")
    return parser.parse_args()


def _load_model(path: str, algo: str) -> Any:
    """Load a saved SB3 model of the requested algorithm."""
    from stable_baselines3 import PPO, SAC

    loader = SAC if algo == "sac" else PPO
    return loader.load(path)


def main() -> None:
    """Evaluate the model across the cost sensitivity grid."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    model = _load_model(args.model_path, args.algo)
    continuous = args.algo == "sac"

    report: dict[str, Any] = {
        "model": args.model_path,
        "bars": len(bars),
        "scenarios": {},
    }
    for spread in args.spreads:
        for slippage in args.slippages:
            key = f"spread{spread}_slip{slippage}"
            env = TradingEnv(
                bars=bars,
                features=features,
                use_sweep_reward=True,
                continuous_actions=continuous,
                cost_model=CostModel(spread_points=spread, slippage_points=slippage),
            )

            def action_fn(obs: dict[str, Any]) -> Any:
                return model.predict(obs, deterministic=True)[0]

            metrics = run_episode(env, action_fn=action_fn)
            delays = sweep_delay_breakdown(env.trade_log)
            report["scenarios"][key] = {
                "sharpe": round(metrics.sharpe, 3),
                "sortino": round(metrics.sortino, 3),
                "max_drawdown": round(metrics.max_drawdown, 4),
                "total_return_pct": round(metrics.total_return_pct, 3),
                "breach_count": metrics.breach_count,
                **{k: round(v, 3) for k, v in delays.items()},
            }
            print(f"{key}: sharpe={report['scenarios'][key]['sharpe']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"OOS report saved to {out_path}")


if __name__ == "__main__":
    main()
