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
from quant_rl.eval.export import save_run
from quant_rl.eval.plots import plot_baseline_comparison
from quant_rl.eval.plots_interactive import plot_baseline_comparison as plot_baseline_comparison_html

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


def _run_baseline(name: str, bars, features, signal, cfg, max_loss_per_trade: float | None = None) -> tuple[dict, object]:
    vals = signal.reindex(bars.index).fillna(0).values.astype(int)
    obs_window = cfg.env.obs_window
    step_counter = [obs_window]

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
        max_loss_per_trade_usd=max_loss_per_trade,
    )
    result["initial_balance"] = cfg.account.initial_balance
    m = calculate_metrics(
        result["equity"],
        trades=result["trades"],
        n_sessions=result.get("n_sessions", 1),
        n_breach_sessions=result.get("n_breach_sessions", 0),
    )
    log.info(
        "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Breaches=%d/%d  Return=%.2f%% "
        "| sessions active=%d breached=%d",
        name,
        m.sharpe,
        m.max_drawdown * 100,
        m.total_trades,
        result.get("n_breach_sessions", 0),
        result.get("n_sessions", 1),
        m.total_return * 100,
        result.get("n_sessions_with_trades", 0),
        result.get("n_breach_sessions", 0),
    )
    return result, m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Skip saving artifacts")
    parser.add_argument("--out", default="outputs", help="Base output directory")
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

    max_loss_per_trade = None
    try:
        max_loss_per_trade = cfg.backtest.validation.max_loss_per_trade_usd
        log.info("Max loss per trade: $%.2f", max_loss_per_trade)
    except Exception:
        pass

    strategy_results: dict[str, tuple] = {}
    for strat_name, signal_fn in [
        ("EMA crossover", lambda: ema_crossover(primary_m1)),
        ("MACD",          lambda: macd_baseline(primary_m1)),
        ("RSI mean-rev",  lambda: rsi_mean_reversion(primary_m1)),
    ]:
        result, m = _run_baseline(strat_name, primary_m1, features, signal_fn(), cfg, max_loss_per_trade)
        strategy_results[strat_name] = (result, m)
        if not args.no_save:
            save_run(result, m, out_dir=args.out, name=strat_name.replace(" ", "_"),
                     bars=primary_m1, cfg=cfg)

    bah = buy_and_hold_returns(primary_m1)
    log.info("[Buy-and-Hold] Final equity factor: %.4f", float(bah.iloc[-1]))

    if not args.no_save and strategy_results:
        import pandas as pd
        from pathlib import Path as _Path
        from datetime import datetime

        # Comparison chart: overlay all strategies
        equity_dict = {name: res["equity"] for name, (res, _) in strategy_results.items()}
        # Add buy-and-hold as normalised equity
        equity_dict["Buy-and-Hold"] = bah * cfg.account.initial_balance

        out_base = _Path(args.out)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_dir = out_base / f"{ts}_baseline_comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)

        plot_baseline_comparison(equity_dict, out_path=comp_dir / "comparison.png")
        try:
            plot_baseline_comparison_html(equity_dict, out_path=comp_dir / "comparison.html")
        except Exception:
            pass

        # Combined metrics CSV
        rows = []
        for name, (_, m) in strategy_results.items():
            from dataclasses import asdict
            row = {"strategy": name, **asdict(m)}
            rows.append(row)
        pd.DataFrame(rows).to_csv(comp_dir / "baselines_metrics.csv", index=False)
        log.info("Comparison artifacts saved to %s", comp_dir)


if __name__ == "__main__":
    main()
