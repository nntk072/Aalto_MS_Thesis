"""Run backtester with a random / placeholder policy.

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.run_backtest
    python -m quant_rl.train.run_backtest env.obs_window=30 --seed=42
    python -m quant_rl.train.run_backtest --no-save
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import random

import numpy as np
import pandas as pd

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline, build_tick_books
from quant_rl.data.split import split_train_test, get_split_config
from quant_rl.features.build import build_features
from quant_rl.backtest.engine import run_backtest
from quant_rl.eval.metrics import calculate_metrics
from quant_rl.eval.export import save_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _random_policy(obs: np.ndarray) -> int:
    return random.choice([-1, 0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Skip saving artifacts")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym   = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1    = data[primary_sym]["M1"]
    secondary_m1  = data.get(secondary_sym, {}).get("M1")

    cache_dir  = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features   = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    # Date-based train / test split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Tick books for bid/ask fill execution (sliced per split)
    tick_books    = build_tick_books(cfg, force=args.force)
    primary_ticks = tick_books.get(cfg.data.primary)
    train_ticks   = primary_ticks.slice(
        pd.Timestamp("2000-01-01", tz=cfg.data.tz),
        pd.Timestamp(train_end + " 23:59:59", tz=cfg.data.tz),
    ) if primary_ticks is not None else None
    test_ticks    = primary_ticks.slice(
        pd.Timestamp(test_start + " 00:00:00", tz=cfg.data.tz),
        pd.Timestamp("2099-01-01", tz=cfg.data.tz),
    ) if primary_ticks is not None else None

    max_loss_per_trade = None
    take_profit_per_trade = None
    try:
        max_loss_per_trade = cfg.backtest.validation.max_loss_per_trade_usd
        take_profit_per_trade = cfg.backtest.validation.take_profit_per_trade_usd
    except Exception:
        pass
    
    contract_size = cfg.account.contract_size if hasattr(cfg.account, 'contract_size') else 1.0
    default_risk_frac = cfg.risk.default_risk_frac if hasattr(cfg, 'risk') else 0.01

    def _run(bars, feats, ticks, label: str) -> tuple[dict, object]:
        log.info("Running backtest [%s] seed=%d …", label, args.seed)
        result = run_backtest(
            bars=bars, features=feats,
            policy=_random_policy,
            obs_window=cfg.env.obs_window,
            initial_balance=cfg.account.initial_balance,
            max_loss_per_trade_usd=max_loss_per_trade,
            take_profit_per_trade_usd=take_profit_per_trade,
            tickbook=ticks,
            use_structure_sl_tp=True,
            contract_size=contract_size,
            default_risk_frac=default_risk_frac,
        )
        result["initial_balance"] = cfg.account.initial_balance
        m = calculate_metrics(
            result["equity"], trades=result["trades"],
            n_sessions=result.get("n_sessions", 1),
            n_breach_sessions=result.get("n_breach_sessions", 0),
        )
        log.info(
            "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Breaches=%d/%d  Return=%.2f%%",
            label, m.sharpe, m.max_drawdown * 100, m.total_trades,
            result.get("n_breach_sessions", 0), result.get("n_sessions", 1),
            m.total_return * 100,
        )
        return result, m

    train_result, train_m = _run(train_bars, train_feat, train_ticks, "training")
    test_result,  test_m  = _run(test_bars,  test_feat,  test_ticks,  "testing")

    if not args.no_save:
        run_dir = save_run(
            out_dir=args.out, name=f"random_seed{args.seed}",
            train_result=train_result, train_metrics=train_m, train_bars=train_bars,
            test_result=test_result,   test_metrics=test_m,   test_bars=test_bars,
            cfg=cfg,
        )
        log.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
