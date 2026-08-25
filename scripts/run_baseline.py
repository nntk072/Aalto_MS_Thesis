"""Run a baseline strategy through TradingEnv and report its metrics.

Example:
    python scripts/run_baseline.py --strategy breakout \
        --bars-csv data/us100_2025.csv --features-csv data/us100_feat.csv \
        --out results/baselines/breakout.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.baselines import (  # noqa: E402
    BaseStrategy,
    BuyAndHoldStrategy,
    EMAMACDRSIStrategy,
    MultiLevelBreakoutStrategy,
)
from quant_rl.envs.trading_env import TradingEnv  # noqa: E402
from quant_rl.evaluation import run_episode, sweep_delay_breakdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", required=True, choices=["buy_hold", "ema_macd_rsi", "breakout"]
    )
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--features-csv")
    parser.add_argument("--out", default=None, help="Output JSON path")
    return parser.parse_args()


def build_strategy(name: str, bars: pd.DataFrame) -> BaseStrategy:
    """Instantiate the requested baseline strategy over ``bars``.

    Args:
        name: One of ``buy_hold``, ``ema_macd_rsi``, ``breakout``.
        bars: OHLCV DataFrame used to pre-compute signals.

    Returns:
        A ready-to-run strategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    if name == "buy_hold":
        return BuyAndHoldStrategy(n_bars=len(bars))
    if name == "ema_macd_rsi":
        return EMAMACDRSIStrategy(bars)
    if name == "breakout":
        return MultiLevelBreakoutStrategy(bars)
    raise ValueError(f"unknown strategy: {name}")


def main() -> None:
    """Run the chosen baseline and persist its evaluation report."""
    args = parse_args()
    bars = pd.read_csv(args.bars_csv, index_col=0, parse_dates=True)
    features = (
        pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
        if args.features_csv
        else bars.select_dtypes(include=["number"])
    )
    env = TradingEnv(bars=bars, features=features, continuous_actions=True)
    strategy = build_strategy(args.strategy, bars)

    metrics = run_episode(env, action_fn=strategy.act)
    delays = sweep_delay_breakdown(env.trade_log)

    report: dict[str, Any] = {
        "strategy": args.strategy,
        "sharpe": round(metrics.sharpe, 3),
        "sortino": round(metrics.sortino, 3),
        "calmar": round(metrics.calmar, 3),
        "max_drawdown": round(metrics.max_drawdown, 4),
        "total_return_pct": round(metrics.total_return_pct, 3),
        "profit_factor": (metrics.profit_factor if math.isfinite(metrics.profit_factor) else None),
        "win_rate": round(metrics.win_rate, 4),
        "expectancy": round(metrics.expectancy, 3),
        "n_trades": metrics.n_trades,
        "breach_count": metrics.breach_count,
        **{k: round(v, 3) for k, v in delays.items()},
    }

    out_path = Path(args.out or f"results/baselines/{args.strategy}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
