"""Evaluate rule-based baselines on the held-out out-of-sample split.

Runs each registered `BaseStrategy` through the shared evaluation pipeline
(`run_episode` + `TradingEnv`) on the same held-out test slice used by
`scripts/train_rl.py`, so RL agents and baselines can be compared in one
table.  Writes one `metrics.json` per strategy, in the same report shape
produced by `train_rl.py` (an `out_of_sample` block readable by
`scripts/report_g3.py`).

Example:
    python scripts/run_baseline_eval.py --bars-csv data/us100_2025.csv \
        --out models/rl_runs/baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.baselines import (  # noqa: E402
    BaseStrategy,
    BuyAndHoldStrategy,
    EMAMACDRSIStrategy,
    MultiLevelBreakoutStrategy,
)
from quant_rl.data.split import split_train_test  # noqa: E402
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import build_run_report, run_episode  # noqa: E402

OBS_WINDOW = 60  # must match TradingEnv's default obs_window


def build_strategies(bars: pd.DataFrame) -> dict[str, BaseStrategy]:
    """Instantiate every registered baseline over the evaluation bars."""
    strategies: dict[str, BaseStrategy] = {
        "buy_and_hold": BuyAndHoldStrategy(n_bars=len(bars)),
        "ema_macd_rsi": EMAMACDRSIStrategy(bars),
        "breakout": MultiLevelBreakoutStrategy(bars),
    }
    # The env starts stepping at bar `obs_window`; skip the first
    # pre-computed signals so signal k lines up with the traded bar.
    for strategy in strategies.values():
        strategy.fast_forward(OBS_WINDOW)
    return strategies


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", required=True, help="OHLCV CSV with DatetimeIndex")
    parser.add_argument("--features-csv", help="Feature CSV; defaults to numeric bars columns")
    parser.add_argument("--train-end", default="2025-12-31", help="Last in-sample date")
    parser.add_argument("--test-start", default="2026-01-01", help="First out-of-sample date")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["buy_and_hold", "ema_macd_rsi", "breakout"],
        choices=["buy_and_hold", "ema_macd_rsi", "breakout"],
    )
    parser.add_argument("--out-dir", default="models/rl_runs", help="Base directory for reports")
    return parser.parse_args()


def main() -> None:
    """Evaluate each baseline on the held-out split and persist metrics.json."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )

    _, test_bars, _, test_features = split_train_test(
        bars, features, args.train_end, args.test_start
    )
    if test_bars.empty:
        raise SystemExit(f"empty test split with --test-start {args.test_start}")
    print(f"held-out split: {len(test_bars)} bars (≥{args.test_start})")

    strategies = build_strategies(test_bars)
    for name in args.strategies:
        strategy = strategies[name]

        def action_fn(obs: Any, _strategy: BaseStrategy = strategy) -> np.ndarray[Any, Any]:
            # Baselines use the continuous action contract ([-1, 1] fraction);
            # TradingEnv's continuous decoder expects an ndarray action.
            return np.array([_strategy.act(obs)], dtype=np.float32)

        # Baselines use the continuous action contract ([-1, 1] fraction).
        env = TradingEnv(
            bars=test_bars,
            features=test_features,
            continuous_actions=True,
            # Eval mode: a breach blocks trading for the rest of the session
            # instead of truncating the episode (mirrors run_backtest).
            episodic=False,
        )
        metrics = run_episode(env, action_fn=action_fn)
        report: dict[str, Any] = {
            "run_name": f"baseline_{name}",
            "split": {
                "train_end": args.train_end,
                "test_start": args.test_start,
                "test_bars": int(len(test_bars)),
            },
            "out_of_sample": build_run_report(metrics, env.trade_log),
        }

        out_dir = Path(args.out_dir) / f"baseline_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

        oos = report["out_of_sample"]
        print(
            f"{name:16} sharpe={oos['sharpe']:7.3f}  mdd={oos['max_drawdown']:7.4f}  "
            f"pnl={oos['total_pnl']:10.2f}  trades={oos['n_trades']}  -> {out_dir / 'metrics.json'}"
        )


if __name__ == "__main__":
    main()
