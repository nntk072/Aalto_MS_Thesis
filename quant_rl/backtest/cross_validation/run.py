"""CLI entrypoint for engine cross-validation.

Runs both the custom event-driven engine (quant_rl.backtest.engine) and
backtrader on identical data/signals and prints a diff report.

Usage:
    python -m quant_rl.backtest.cross_validation.run [--data-path DATA_PATH] [--tolerance TOLERANCE]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import backtrader as bt
import pandas as pd

from quant_rl.backtest.costs import CostModel
from quant_rl.backtest.engine import run_backtest

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class _SignalFeed(bt.feeds.PandasData):
    lines = ("signal",)
    params = (("signal", "signal"),)


class _DelayedSignalStrategy(bt.Strategy):
    """Execute a precomputed action at the matching custom-engine close."""

    params = (("last_bar", 0),)

    def __init__(self) -> None:
        self.order = None

    def notify_order(self, order: bt.Order) -> None:
        if order.status not in (order.Submitted, order.Accepted):
            self.order = None

    def next(self) -> None:
        if self.order is not None:
            return
        if len(self) == self.p.last_bar + 1:
            if self.position:
                self.order = self.close()
            return
        if len(self) > self.p.last_bar + 1:
            return
        if len(self) < 3:
            return

        # The custom engine observes action[i - 1] on bar i and fills at
        # bar i + 1 close. This delayed signal reaches the same close in
        # Backtrader with cheat-on-close enabled.
        action = int(self.data.signal[-2])
        if action == 2 and self.position:
            self.order = self.close()
        elif action == 1 and not self.position:
            self.order = self.buy(size=1)
        elif action == -1 and not self.position:
            self.order = self.sell(size=1)
        elif len(self) == self.p.last_bar and self.position:
            self.order = self.close()


def _crossover_actions(close: pd.Series) -> pd.Series:
    """Encode long-entry and flat-exit SMA crossovers as discrete actions."""
    short = close.rolling(20, min_periods=20).mean()
    long = close.rolling(50, min_periods=50).mean()
    previous_difference = (short - long).shift(1)
    difference = short - long
    actions = pd.Series(0, index=close.index, dtype="int64")
    actions.loc[(difference > 0) & (previous_difference <= 0)] = 1
    actions.loc[(difference < 0) & (previous_difference >= 0)] = 2
    return actions


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load data from CSV file."""
    from .utils.data_loader import load_data_from_csv

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data_feed = load_data_from_csv(path)
    # Convert to DataFrame for custom engine
    df = pd.DataFrame(data_feed.dataname)
    if df.empty:
        raise ValueError(f"No data loaded from {data_path}")

    # Ensure we have the required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in data")

    return df


def run_backtrader_engine(
    df: pd.DataFrame,
    tolerance: float = 0.01,
    *,
    actions: pd.Series | None = None,
) -> dict[str, Any]:
    """Run Backtrader with the same precomputed action trace as the custom engine.

    Returns dict with trade_count, total_pnl, max_drawdown.
    """
    del tolerance  # Kept for call compatibility with the CLI.
    action_series = _crossover_actions(df["close"]) if actions is None else actions
    if not action_series.index.equals(df.index):
        raise ValueError("Cross-validation actions must align exactly with bar timestamps")
    # Duplicate the final quote so a final liquidation submitted at the last
    # real close gets one broker cycle to execute at that same close.
    final_quote = df.iloc[[-1]].assign(signal=0)
    feed_bars = pd.concat(
        [df.assign(signal=action_series), final_quote, final_quote], ignore_index=False
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(_SignalFeed(dataname=feed_bars))
    cerebro.addstrategy(_DelayedSignalStrategy, last_bar=len(df))
    cerebro.broker.setcash(100000.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.broker.set_coc(True)
    results = cerebro.run()
    strat = results[0]

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("closed", 0)
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100.0

    return {
        "engine": "backtrader",
        "trade_count": int(total_trades),
        "total_pnl": float(cerebro.broker.getvalue() - 100000.0),
        "max_drawdown": float(max_drawdown),
    }


def run_custom_engine(
    df: pd.DataFrame,
    tolerance: float = 0.01,
    *,
    actions: pd.Series | None = None,
) -> dict[str, Any]:
    """Run the custom engine with the shared precomputed action trace.

    Returns dict with trade_count, total_pnl, max_drawdown.
    """
    del tolerance  # Kept for call compatibility with the CLI.
    action_series = _crossover_actions(df["close"]) if actions is None else actions
    if not action_series.index.equals(df.index):
        raise ValueError("Cross-validation actions must align exactly with bar timestamps")
    features = pd.DataFrame({"signal": action_series}, index=df.index)
    result = run_backtest(
        bars=df,
        features=features,
        policy=lambda observation: int(observation[-1, 0]),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, slippage_points=0.0),
        initial_balance=100000.0,
        lots=1.0,
        hold_on_zero=True,
        exit_action=2,
        block_overnight=False,
    )
    trade_count = int(result["trades"].get("position_complete", pd.Series(dtype=bool)).sum())
    equity = result["equity"]
    if equity.empty:
        max_drawdown = 0.0
    else:
        max_drawdown = float((1.0 - equity / equity.cummax()).max())
    return {
        "engine": "custom",
        "trade_count": trade_count,
        "total_pnl": float(result["account"].equity - 100000.0),
        "max_drawdown": max_drawdown,
    }


def compare_results(
    backtrader_result: dict[str, Any], custom_result: dict[str, Any], tolerance: float = 0.01
) -> dict[str, Any]:
    """Compare results from both engines."""
    comparison = {
        "backtrader": backtrader_result,
        "custom": custom_result,
        "agreement": {},
    }

    # Compare trade counts
    trade_count_match = backtrader_result["trade_count"] == custom_result["trade_count"]
    comparison["agreement"]["trade_count_match"] = trade_count_match
    comparison["agreement"]["trade_count_diff"] = (
        backtrader_result["trade_count"] - custom_result["trade_count"]
    )

    # Compare PnL within tolerance
    pnl_diff = abs(backtrader_result["total_pnl"] - custom_result["total_pnl"])
    pnl_tolerance = tolerance * abs(backtrader_result["total_pnl"])
    # Use a small epsilon to handle floating point precision
    pnl_within_tolerance = pnl_diff <= pnl_tolerance + 1e-6
    comparison["agreement"]["pnl_within_tolerance"] = pnl_within_tolerance
    comparison["agreement"]["pnl_diff"] = pnl_diff
    comparison["agreement"]["pnl_tolerance"] = pnl_tolerance

    # Compare max drawdown within tolerance
    dd_diff = abs(backtrader_result["max_drawdown"] - custom_result["max_drawdown"])
    dd_tolerance = tolerance * abs(backtrader_result["max_drawdown"])
    # Use a small epsilon to handle floating point precision
    dd_within_tolerance = dd_diff <= dd_tolerance + 1e-10
    comparison["agreement"]["max_drawdown_within_tolerance"] = dd_within_tolerance
    comparison["agreement"]["max_drawdown_diff"] = dd_diff
    comparison["agreement"]["max_drawdown_tolerance"] = dd_tolerance

    # Overall pass/fail
    overall_pass = trade_count_match and pnl_within_tolerance and dd_within_tolerance
    comparison["agreement"]["overall_pass"] = overall_pass

    return comparison


def print_report(comparison: dict[str, Any]) -> None:
    """Print the comparison report."""
    print("=" * 60)
    print("ENGINE CROSS-VALIDATION REPORT")
    print("=" * 60)

    bt = comparison["backtrader"]
    custom = comparison["custom"]
    agree = comparison["agreement"]

    print("\nBacktrader Engine:")
    print(f"  Trade Count: {bt['trade_count']}")
    print(f"  Total PnL: ${bt['total_pnl']:.2f}")
    print(f"  Max Drawdown: {bt['max_drawdown']:.2%}")

    print("\nCustom Engine:")
    print(f"  Trade Count: {custom['trade_count']}")
    print(f"  Total PnL: ${custom['total_pnl']:.2f}")
    print(f"  Max Drawdown: {custom['max_drawdown']:.2%}")

    print("\nAgreement Analysis:")
    print(f"  Trade Count Match: {'✓ PASS' if agree['trade_count_match'] else '✗ FAIL'}")
    if not agree["trade_count_match"]:
        print(f"    Difference: {agree['trade_count_diff']}")

    print(f"  PnL within tolerance: {'✓ PASS' if agree['pnl_within_tolerance'] else '✗ FAIL'}")
    if not agree["pnl_within_tolerance"]:
        print(
            f"    Difference: ${agree['pnl_diff']:.2f} (tolerance: ${agree['pnl_tolerance']:.2f})"
        )

    print(
        f"  Max Drawdown within tolerance: {'✓ PASS' if agree['max_drawdown_within_tolerance'] else '✗ FAIL'}"
    )
    if not agree["max_drawdown_within_tolerance"]:
        print(
            f"    Difference: {agree['max_drawdown_diff']:.2%} (tolerance: {agree['max_drawdown_tolerance']:.2%})"
        )

    print(f"\n{'✓ OVERALL PASS' if agree['overall_pass'] else '✗ OVERALL FAIL'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run engine cross-validation comparing backtrader and custom engines"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/US100.cash_M1_202412300105_202610162349.csv",
        help="Path to CSV data file (default: US100 M1 data)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for PnL and drawdown comparison (default: 0.01, 1 percent)",
    )
    parser.add_argument("--json-output", type=str, default=None, help="Path to save JSON report")

    args = parser.parse_args()

    try:
        # Load data
        log.info(f"Loading data from: {args.data_path}")
        df = load_data(args.data_path)
        log.info(f"Loaded {len(df)} bars")

        actions = _crossover_actions(df["close"])
        log.info("Running backtrader engine...")
        bt_result = run_backtrader_engine(df, args.tolerance, actions=actions)

        log.info("Running custom engine...")
        custom_result = run_custom_engine(df, args.tolerance, actions=actions)

        # Compare results
        comparison = compare_results(bt_result, custom_result, args.tolerance)

        # Print report
        print_report(comparison)

        # Save JSON if requested
        if args.json_output:
            with open(args.json_output, "w") as f:
                json.dump(comparison, f, indent=2)
            log.info(f"Report saved to: {args.json_output}")

        # Exit with appropriate code
        if comparison["agreement"]["overall_pass"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        log.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
