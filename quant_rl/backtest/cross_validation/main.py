"""Standalone backtest runner using the cross-validation strategies.

This module exposes ``run_backtest`` for ad-hoc use, but the recommended
entrypoint is ``python -m quant_rl.backtest.cross_validation.run`` which
runs both engines and compares them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import backtrader as bt
import pandas as pd

from .strategies.cross_over import CrossOverStrategy
from .strategies.signal_following import SignalFollowingStrategy
from .utils.data_loader import load_data_from_csv


def run_backtest(
    data_feed: bt.feeds.DataBase,
    strategy: type[bt.Strategy] = CrossOverStrategy,
    **kwargs: Any,
) -> bt.Cerebro:
    """Run a backtest with the given data feed and strategy.

    Returns the ``cerebro`` instance after the run so callers can inspect
    analyzers or portfolio value.
    """
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy, **kwargs)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    strat = results[0]
    print("Sharpe Ratio:", strat.analyzers.sharpe.get_analysis().get("sharperatio"))
    print("DrawDown:", strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown"))
    print("Return:", strat.analyzers.returns.get_analysis().get("rtot"))

    trade_analysis = strat.analyzers.trades.get_analysis()
    print("==== Trade Analysis ====")
    print(f"Total Trades: {trade_analysis.get('total', {}).get('total', 0)}")
    print(f"Won: {trade_analysis.get('won', {}).get('total', 0)}")
    print(f"Lost: {trade_analysis.get('lost', {}).get('total', 0)}")
    won = trade_analysis.get("won", {}).get("total", 0)
    total = trade_analysis.get("total", {}).get("total", 0)
    if won and total:
        print(f"Win Rate: {won / total * 100:.2f}%")

    cerebro.plot(style="candlestick")
    return cerebro


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single backtrader backtest.")
    parser.add_argument("--data-path", required=True, help="Path to a local CSV data file")
    parser.add_argument(
        "--strategy",
        default="crossover",
        choices=["crossover", "macd", "ema_macd"],
        help="Strategy to run",
    )
    parser.add_argument("--ma-short", type=int, default=20)
    parser.add_argument("--ma-long", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.data_path)
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    data_feed = load_data_from_csv(path)

    strategy_map: dict[str, type[bt.Strategy]] = {
        "crossover": CrossOverStrategy,
        "macd": SignalFollowingStrategy,
        "ema_macd": SignalFollowingStrategy,
    }
    strategy_cls = strategy_map[args.strategy]

    if args.strategy in ("macd", "ema_macd"):
        from quant_rl.baselines.rule_based import macd_ema50_baseline

        bars = pd.DataFrame(data_feed.dataname)
        if not isinstance(bars.index, pd.DatetimeIndex):
            bars.index = pd.to_datetime(bars.index)
        actions = macd_ema50_baseline(bars)
        cerebro = run_backtest(
            data_feed,
            strategy=strategy_cls,
            actions=actions,
            printlog=True,
        )
    else:
        cerebro = run_backtest(
            data_feed,
            strategy=strategy_cls,
            ma_short_period=args.ma_short,
            ma_long_period=args.ma_long,
            printlog=True,
        )
