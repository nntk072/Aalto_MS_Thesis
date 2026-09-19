"""Out-of-sample cost-sensitivity harness for a saved SB3 model.

Loads a trained PPO/SAC zip, runs deterministic episodes on OOS bars under a
grid of ``CostModel(spread_points, slippage_points)`` assumptions, and writes
a JSON report. Slippage is applied by widening the effective spread
(``spread_points + slippage_points``) because fills use ``bar_quote`` spread.

Example:
    python scripts/test_oos.py --model-path outputs/<run>/model/ppo_final.zip \\
        --bars-csv data/us100_oos.csv --features-csv data/us100_feat_oos.csv \\
        --algo ppo
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
from quant_rl.config import load_config  # noqa: E402
from quant_rl.data.split import get_split_config, split_train_test  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.eval.rollout import make_action_fn  # noqa: E402
from quant_rl.evaluation import build_run_report, run_episode, sweep_delay_breakdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to saved SB3 .zip")
    parser.add_argument("--bars-csv", required=True, help="Bars CSV (full or OOS-only)")
    parser.add_argument("--features-csv", help="Feature CSV; defaults to numeric bars columns")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--config", default=None, help="Optional RL config for split dates")
    parser.add_argument(
        "--train-end",
        default=None,
        help="If set with --test-start, slice OOS from a full bars CSV",
    )
    parser.add_argument("--test-start", default=None)
    parser.add_argument(
        "--spreads",
        nargs="+",
        type=float,
        default=[0.6, 1.0, 1.5],
        help="CostModel.spread_points values",
    )
    parser.add_argument(
        "--slippages",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2],
        help="CostModel.slippage_points values (added into effective spread)",
    )
    parser.add_argument("--out", default="results/oos_report.json")
    return parser.parse_args()


def load_model(path: str, algo: str) -> Any:
    """Load a saved SB3 model of the requested algorithm."""
    from stable_baselines3 import PPO, SAC

    loader = SAC if algo == "sac" else PPO
    return loader.load(path)


def resolve_oos_frames(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    train_end: str | None,
    test_start: str | None,
    cfg_train_end: str | None,
    cfg_test_start: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return OOS bars/features, optionally slicing via train/test dates."""
    end = train_end or cfg_train_end
    start = test_start or cfg_test_start
    if end and start:
        _, test_bars, _, test_feat = split_train_test(bars, features, end, start)
        if test_bars.empty:
            raise SystemExit(f"empty OOS slice with train_end={end} test_start={start}")
        return test_bars, test_feat
    return bars, features


def make_cost_model(spread: float, slippage: float) -> CostModel:
    """Build a CostModel whose fills see spread + slippage as effective spread."""
    return CostModel(
        spread_points=float(spread) + float(slippage),
        slippage_points=float(slippage),
    )


def eval_scenario(
    model: Any,
    *,
    algo: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> dict[str, Any]:
    """One deterministic OOS episode under ``cost_model``."""
    continuous = algo == "sac"
    env = TradingEnv(
        bars=bars,
        features=features,
        continuous_actions=continuous,
        cost_model=cost_model,
        episodic=False,
    )
    metrics = run_episode(
        env,
        action_fn=make_action_fn(model, continuous_actions=continuous),
    )
    report = build_run_report(metrics, env.trade_log)
    report["sweep_delay_breakdown"] = sweep_delay_breakdown(env.trade_log)
    report["cost"] = {
        "spread_points": cost_model.spread_points,
        "slippage_points": cost_model.slippage_points,
        "nominal_spread": cost_model.spread_points - cost_model.slippage_points,
    }
    return report


def run_cost_grid(
    model: Any,
    *,
    algo: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    spreads: list[float],
    slippages: list[float],
) -> dict[str, Any]:
    """Evaluate every (spread, slippage) pair and return the scenarios map."""
    scenarios: dict[str, Any] = {}
    for spread in spreads:
        for slippage in slippages:
            key = f"spread{spread}_slip{slippage}"
            cost = make_cost_model(spread, slippage)
            scenarios[key] = eval_scenario(
                model,
                algo=algo,
                bars=bars,
                features=features,
                cost_model=cost,
            )
    return scenarios


def main() -> None:
    """Evaluate the model across the cost sensitivity grid."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )

    cfg_train_end = cfg_test_start = None
    if args.config:
        cfg = load_config(config_path=args.config)
        cfg_train_end, cfg_test_start = get_split_config(cfg)

    oos_bars, oos_feat = resolve_oos_frames(
        bars,
        features,
        train_end=args.train_end,
        test_start=args.test_start,
        cfg_train_end=cfg_train_end,
        cfg_test_start=cfg_test_start,
    )

    model = load_model(args.model_path, args.algo)
    scenarios = run_cost_grid(
        model,
        algo=args.algo,
        bars=oos_bars,
        features=oos_feat,
        spreads=list(args.spreads),
        slippages=list(args.slippages),
    )

    report: dict[str, Any] = {
        "model": args.model_path,
        "algo": args.algo,
        "bars": len(oos_bars),
        "scenarios": scenarios,
        "summary": {
            key: {
                "sharpe": sc.get("sharpe"),
                "max_drawdown": sc.get("max_drawdown"),
                "n_trades": sc.get("n_trades"),
                "total_return_pct": sc.get("total_return_pct"),
            }
            for key, sc in scenarios.items()
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"saved OOS cost report to {out_path}")


if __name__ == "__main__":
    main()
