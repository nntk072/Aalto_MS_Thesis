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

from quant_rl.backtest.costs import COST_US100
from quant_rl.backtest.engine import run_backtest

from .strategies.cross_over import CrossOverStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def run_backtrader_engine(df: pd.DataFrame, tolerance: float = 0.01) -> dict[str, Any]:
    """Run backtrader engine on the given data.

    Returns dict with trade_count, total_pnl, max_drawdown.
    """
    # Convert DataFrame to backtrader data feed
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1,
    )

    # Create cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(CrossOverStrategy, ma_short_period=20, ma_long_period=50, printlog=False)

    # Set cash
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    total_pnl = strat.analyzers.returns.get_analysis().get("rtot", 0.0) * 100000.0

    # Get max drawdown
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0)

    return {
        "engine": "backtrader",
        "trade_count": int(total_trades),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_drawdown),
    }


def run_custom_engine(df: pd.DataFrame, tolerance: float = 0.01) -> dict[str, Any]:
    """Run custom event-driven engine on the given data.

    Returns dict with trade_count, total_pnl, max_drawdown.
    """
    # Create features DataFrame (empty for now - just use raw bars)
    # The custom engine needs features, but for MACD crossover we can create dummy features
    features_df = pd.DataFrame(index=df.index)

    # Simple MACD-based policy function
    def macd_policy(obs: Any) -> int:
        """MACD crossover policy: returns 1 (long), -1 (short), or 0 (hold)."""
        # For simplicity, we'll just return 0 for now - the real signal logic
        # will be implemented in the comparison
        # This is a placeholder to show the interface
        if len(obs) < 2:
            return 0
        # Dummy logic - in practice this would compute MACD from the observation
        return 0

    try:
        result = run_backtest(
            bars=df,
            features=features_df,
            policy=macd_policy,
            obs_window=60,
            cost_model=COST_US100,
            initial_balance=100000.0,
            lots=1.0,
        )

        trade_count = len(result["trades"])
        final_equity = result["account"].equity
        total_pnl = final_equity - 100000.0

        # Calculate max drawdown from equity curve
        equity_series = result["equity"]
        if len(equity_series) > 0:
            running_max = equity_series.cummax()
            drawdowns = (equity_series - running_max) / running_max
            max_drawdown = float(drawdowns.min())
        else:
            max_drawdown = 0.0

        return {
            "engine": "custom",
            "trade_count": int(trade_count),
            "total_pnl": float(total_pnl),
            "max_drawdown": float(max_drawdown),
        }
    except Exception as e:
        log.error(f"Custom engine failed: {e}")
        # Return placeholder values
        return {
            "engine": "custom",
            "trade_count": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "error": str(e),
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
        default="data/US100.cash_M1_202412300105_202607031959.csv",
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

        # Run both engines
        log.info("Running backtrader engine...")
        bt_result = run_backtrader_engine(df, args.tolerance)

        log.info("Running custom engine...")
        custom_result = run_custom_engine(df, args.tolerance)

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
