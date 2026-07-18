"""Run backtester with a random / placeholder policy.

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.run_backtest
    python -m quant_rl.train.run_backtest env.obs_window=30 --seed=42
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import random

import numpy as np

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.features.build import build_features
from quant_rl.backtest.engine import run_backtest
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.report import print_report, aggregate_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _random_policy(obs: np.ndarray) -> int:
    return random.choice([-1, 0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    log.info("Running backtest with random policy (seed=%d) …", args.seed)
    result = run_backtest(
        bars=primary_m1,
        features=features,
        policy=_random_policy,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
    )

    equity = result["equity"]
    trades = result["trades"]
    m = calculate_metrics(equity, trades=trades, n_breach_sessions=len(result["breaches"]))

    log.info("Sharpe=%.3f  Sortino=%.3f  MaxDD=%.2f%%  Trades=%d  Breaches=%d",
             m.sharpe, m.sortino, m.max_drawdown * 100, m.total_trades, len(result["breaches"]))


if __name__ == "__main__":
    main()
