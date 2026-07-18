"""Run rule-based baselines through the backtester.

Usage
-----
    cd Aalto_MS_Thesis
    python -m quant_rl.train.run_baselines
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging

import numpy as np

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.features.build import build_features
from quant_rl.backtest.engine import run_backtest
from quant_rl.baselines.rule_based import ema_crossover, macd_baseline, rsi_mean_reversion
from quant_rl.baselines.buy_and_hold import buy_and_hold_returns
from quant_rl.eval.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def signal_to_policy(signal_series):
    """Convert a signal pd.Series into a stateless policy function."""
    arr = signal_series.values
    idx_map = {dt: i for i, dt in enumerate(signal_series.index)}

    def policy(obs: np.ndarray) -> int:
        # obs is the feature window; we use the current step counter via closure
        # Simple fallback: return the last signal
        return int(arr[policy._step]) if policy._step < len(arr) else 0

    policy._step = 0
    return policy


def _run_baseline(name: str, bars, features, signal, cfg) -> None:
    vals = signal.reindex(bars.index).fillna(0).values.astype(int)
    obs_window = cfg.env.obs_window
    step_counter = [obs_window]   # engine starts at obs_window

    def policy(obs: np.ndarray) -> int:
        s = int(vals[step_counter[0]]) if step_counter[0] < len(vals) else 0
        step_counter[0] += 1
        return s

    result = run_backtest(
        bars=bars,
        features=features,
        policy=policy,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
    )
    m = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_breach_sessions=len(result["breaches"]),
    )
    log.info(
        "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Breaches=%d  Return=%.2f%%",
        name,
        m.sharpe,
        m.max_drawdown * 100,
        m.total_trades,
        len(result["breaches"]),
        m.total_return * 100,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1 = data[primary_sym]["M1"]
    secondary_m1 = data.get(secondary_sym, {}).get("M1")

    cache_dir = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    log.info("Running baselines …")

    _run_baseline("EMA crossover", primary_m1, features, ema_crossover(primary_m1), cfg)
    _run_baseline("MACD", primary_m1, features, macd_baseline(primary_m1), cfg)
    _run_baseline("RSI mean-rev", primary_m1, features, rsi_mean_reversion(primary_m1), cfg)

    bah = buy_and_hold_returns(primary_m1)
    log.info("[Buy-and-Hold] Final equity factor: %.4f", float(bah.iloc[-1]))


if __name__ == "__main__":
    main()
