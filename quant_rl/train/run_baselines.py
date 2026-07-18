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
from datetime import datetime

import numpy as np
import pandas as pd

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import split_train_test, get_split_config
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


def _run_baseline(
    name: str,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    signal: pd.Series,
    cfg,
    max_loss_per_trade: float | None = None,
) -> tuple[dict, object]:
    vals = signal.reindex(bars.index).fillna(0).values.astype(int)
    obs_window = cfg.env.obs_window
    step_counter = [obs_window]

    def policy(obs: np.ndarray) -> int:
        s = int(vals[step_counter[0]]) if step_counter[0] < len(vals) else 0
        step_counter[0] += 1
        return s

    result = run_backtest(
        bars=bars, features=features, policy=policy,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        max_loss_per_trade_usd=max_loss_per_trade,
    )
    result["initial_balance"] = cfg.account.initial_balance
    m = calculate_metrics(
        result["equity"], trades=result["trades"],
        n_sessions=result.get("n_sessions", 1),
        n_breach_sessions=result.get("n_breach_sessions", 0),
    )
    log.info(
        "[%s] Sharpe=%.3f  MaxDD=%.2f%%  Trades=%d  Return=%.2f%%  Breaches=%d/%d",
        name, m.sharpe, m.max_drawdown * 100, m.total_trades, m.total_return * 100,
        result.get("n_breach_sessions", 0), result.get("n_sessions", 1),
    )
    return result, m


def _save_comparison(
    equity_dict: dict[str, pd.Series],
    metrics_rows: list[dict],
    comp_dir: Path,
    label: str,
) -> None:
    """Write comparison chart + metrics CSV for one split."""
    (comp_dir / label).mkdir(parents=True, exist_ok=True)
    split_dir = comp_dir / label
    plot_baseline_comparison(equity_dict, out_path=split_dir / "comparison.png")
    try:
        plot_baseline_comparison_html(equity_dict, out_path=split_dir / "comparison.html")
    except Exception:
        pass
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(split_dir / "baselines_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Skip saving artifacts")
    parser.add_argument("--out", default="outputs", help="Base output directory")
    args = parser.parse_args()

    cfg  = load_config(args.overrides)
    data = run_pipeline(cfg, force=args.force)

    primary_sym   = cfg.data.primary
    secondary_sym = cfg.data.secondary
    primary_m1    = data[primary_sym]["M1"]
    secondary_m1  = data.get(secondary_sym, {}).get("M1")

    cache_dir  = Path(cfg.data.cache_dir)
    feat_cache = cache_dir / f"{primary_sym}_features.parquet"
    features   = build_features(primary_m1, secondary=secondary_m1, cfg=cfg, cache_path=feat_cache)

    # Date-based split
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, train_feat, test_feat = split_train_test(
        primary_m1, features, train_end, test_start
    )
    log.info("Split: train=%d bars (≤%s)  test=%d bars (≥%s)",
             len(train_bars), train_end, len(test_bars), test_start)

    max_loss_per_trade = None
    try:
        max_loss_per_trade = cfg.backtest.validation.max_loss_per_trade_usd
        log.info("Max loss per trade: $%.2f", max_loss_per_trade)
    except Exception:
        pass

    log.info("Running baselines …")

    strategies = [
        ("EMA crossover", lambda b=None: ema_crossover(primary_m1)),
        ("MACD",          lambda b=None: macd_baseline(primary_m1)),
        ("RSI mean-rev",  lambda b=None: rsi_mean_reversion(primary_m1)),
    ]

    # Run each strategy on both splits
    # per_strat[name] = {"train": (result, m), "test": (result, m)}
    per_strat: dict[str, dict] = {}
    for strat_name, signal_fn in strategies:
        signal = signal_fn()
        log.info("=== %s ===", strat_name)
        tr_result, tr_m = _run_baseline(f"{strat_name}/train", train_bars, train_feat,
                                         signal, cfg, max_loss_per_trade)
        te_result, te_m = _run_baseline(f"{strat_name}/test",  test_bars,  test_feat,
                                         signal, cfg, max_loss_per_trade)
        per_strat[strat_name] = {"train": (tr_result, tr_m), "test": (te_result, te_m)}

        if not args.no_save:
            save_run(
                out_dir=args.out, name=strat_name.replace(" ", "_"),
                train_result=tr_result, train_metrics=tr_m, train_bars=train_bars,
                test_result=te_result,  test_metrics=te_m,  test_bars=test_bars,
                cfg=cfg,
            )

    # Buy-and-hold reference (computed on full data, then sliced)
    bah_full  = buy_and_hold_returns(primary_m1) * cfg.account.initial_balance
    bah_train = bah_full.loc[bah_full.index <= train_bars.index.max()] if not train_bars.empty else bah_full
    bah_test  = bah_full.loc[bah_full.index >= test_bars.index.min()]  if not test_bars.empty else bah_full
    log.info("[Buy-and-Hold] Final equity factor: %.4f", float(bah_full.iloc[-1]) / cfg.account.initial_balance)

    if not args.no_save and per_strat:
        from dataclasses import asdict
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_dir = Path(args.out) / f"{ts}_baseline_comparison"

        for split_label, bah_eq, split_key in [
            ("training", bah_train, "train"),
            ("testing",  bah_test,  "test"),
        ]:
            eq_dict = {name: v[split_key][0]["equity"] for name, v in per_strat.items()
                       if not v[split_key][0]["equity"].empty}
            eq_dict["Buy-and-Hold"] = bah_eq

            rows = [{"strategy": name, **asdict(v[split_key][1])}
                    for name, v in per_strat.items()]
            _save_comparison(eq_dict, rows, comp_dir, split_label)

        log.info("Comparison artifacts saved to %s", comp_dir)


if __name__ == "__main__":
    main()


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
