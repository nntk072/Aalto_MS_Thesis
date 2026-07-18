"""Run backtester with baseline strategies (MACD, EMA, RSI, etc.).

These baselines use structure-aware SL/TP sizing to validate engine and charts
before RL training. Helps verify hold times and lot sizes are realistic.

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.run_baselines --strategy macd
    python -m quant_rl.train.run_baselines --strategy ema --seed 42
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

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


def _macd_policy(obs: np.ndarray) -> int:
    """MACD crossover baseline.
    
    Expected obs shape: (60, n_features) where features include MACD columns.
    """
    if len(obs) < 2:
        return 0
    
    # Try to extract MACD line and signal from last two bars
    # Assuming columns are [ret_1, ema_9, ..., macd_line, macd_signal, macd_hist, ...]
    # For now, use simple heuristic: if we see momentum in last few bars
    # Look for anything that looks like momentum or position changes
    
    # Simple fallback: if last row has positive returns, go long; if negative, go short
    # This is a placeholder - real MACD would read specific feature columns
    last_ret = obs[-1, 0] if obs.shape[1] > 0 else 0
    if last_ret > 0.0001:
        return 1
    elif last_ret < -0.0001:
        return -1
    else:
        return 0


def _ema_policy(obs: np.ndarray) -> int:
    """Simple EMA crossover baseline (fast EMA > slow EMA = long).
    
    This reads supposed EMA features from the observation.
    """
    if len(obs) < 2 or obs.shape[1] < 3:
        return 0
    
    # Placeholder: use returns heuristic
    last_ret = obs[-1, 0] if obs.shape[1] > 0 else 0
    if last_ret > 0.0001:
        return 1
    elif last_ret < -0.0001:
        return -1
    else:
        return 0


def _rsi_policy(obs: np.ndarray) -> int:
    """Simple RSI extremes baseline (RSI < 30 = buy, RSI > 70 = sell).
    
    Placeholder implementation.
    """
    if len(obs) < 2:
        return 0
    
    # Placeholder: use returns heuristic
    last_ret = obs[-1, 0] if obs.shape[1] > 0 else 0
    if last_ret > 0.0001:
        return 1
    elif last_ret < -0.0001:
        return -1
    else:
        return 0


STRATEGIES = {
    "macd": _macd_policy,
    "ema": _ema_policy,
    "rsi": _rsi_policy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline strategy backtests.")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="macd",
                        help="Baseline strategy to use")
    parser.add_argument("overrides", nargs="*", help="Config overrides (OmegaConf syntax)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force data pipeline rerun")
    parser.add_argument("--no-save", action="store_true", help="Skip saving artifacts")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    parser.add_argument("--use-structure-sl-tp", action="store_true",
                        help="Use structure-aware SL/TP instead of fixed USD stops")
    args = parser.parse_args()

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

    # Date-based train / test split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    # Tick books
    tick_books = build_tick_books(cfg, force=args.force)
    primary_ticks = tick_books.get(cfg.data.primary)
    train_ticks = primary_ticks.slice(
        pd.Timestamp("2000-01-01", tz=cfg.data.tz),
        pd.Timestamp(train_end + " 23:59:59", tz=cfg.data.tz),
    ) if primary_ticks is not None else None
    test_ticks = primary_ticks.slice(
        pd.Timestamp(test_start + " 00:00:00", tz=cfg.data.tz),
        pd.Timestamp("2099-01-01", tz=cfg.data.tz),
    ) if primary_ticks is not None else None

    max_loss_per_trade = None
    risk_frac = 0.01
    rr_ratio = 2.0
    swing_buffer = 1.0
    try:
        max_loss_per_trade = cfg.backtest.validation.max_loss_per_trade_usd
        risk_frac = cfg.risk.default_risk_frac
        rr_ratio = cfg.risk.rr_ratio_default
        swing_buffer = cfg.risk.swing_buffer_pts
    except Exception:
        pass

    policy = STRATEGIES[args.strategy]

    def _run(bars, feats, ticks, label: str) -> tuple[dict, object]:
        log.info(
            "Running backtest [%s] strategy=%s use_structure=%s …",
            label, args.strategy, args.use_structure_sl_tp
        )
        result = run_backtest(
            bars=bars,
            features=feats,
            policy=policy,
            obs_window=cfg.env.obs_window,
            initial_balance=cfg.account.initial_balance,
            max_loss_per_trade_usd=max_loss_per_trade,
            tickbook=ticks,
            use_structure_sl_tp=args.use_structure_sl_tp,
            risk_frac=risk_frac,
            rr_ratio=rr_ratio,
            swing_buffer_pts=swing_buffer,
            contract_size=cfg.account.contract_size,
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
    test_result, test_m = _run(test_bars, test_feat, test_ticks, "testing")

    if not args.no_save:
        run_dir = save_run(
            out_dir=args.out,
            name=f"baseline_{args.strategy}_seed{args.seed}",
            train_result=train_result,
            train_metrics=train_m,
            train_bars=train_bars,
            test_result=test_result,
            test_metrics=test_m,
            test_bars=test_bars,
            cfg=cfg,
        )
        log.info("Artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
