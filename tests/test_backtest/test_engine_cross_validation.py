"""Tests for engine cross-validation harness."""

import pandas as pd
import pytest

from quant_rl.backtest.cross_validation.run import compare_results


@pytest.fixture
def synthetic_data():
    """Create synthetic OHLCV data for testing."""
    np = pytest.importorskip("numpy")

    # Create 1000 bars of synthetic data with a trend
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")

    # Start at 10000 and add a slight upward trend with noise
    base_price = 10000.0
    trend = np.linspace(0, 100, n_bars)
    noise = np.random.normal(0, 10, n_bars)
    close_prices = base_price + trend + noise

    # Create OHLC data from close prices
    open_prices = close_prices - np.abs(np.random.normal(0, 2, n_bars))
    high_prices = close_prices + np.abs(np.random.normal(0, 5, n_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 5, n_bars))
    volumes = np.random.randint(100, 1000, n_bars)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=dates,
    )

    return df


class TestEngineCrossValidation:
    """Test engine cross-validation functionality."""

    def test_backtrader_engine_import(self):
        """Test that we can import the backtrader engine function."""
        from quant_rl.backtest.cross_validation.run import run_backtrader_engine

        assert callable(run_backtrader_engine)

    def test_compare_results_exact_match(self):
        """Test comparison when results are exactly the same."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.05,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.05,
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # All should pass
        assert comparison["agreement"]["trade_count_match"] is True
        assert comparison["agreement"]["pnl_within_tolerance"] is True
        assert comparison["agreement"]["max_drawdown_within_tolerance"] is True
        assert comparison["agreement"]["overall_pass"] is True

    def test_compare_results_trade_count_mismatch(self):
        """Test comparison when trade counts don't match."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.05,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 11,  # Different
            "total_pnl": 1000.0,
            "max_drawdown": 0.05,
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # Trade count should not match
        assert comparison["agreement"]["trade_count_match"] is False
        assert comparison["agreement"]["trade_count_diff"] == -1
        assert comparison["agreement"]["overall_pass"] is False

    def test_compare_results_pnl_outside_tolerance(self):
        """Test comparison when PnL is outside tolerance."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.05,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 10,
            "total_pnl": 1050.0,  # 5% difference, > 1% tolerance
            "max_drawdown": 0.05,
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # Trade count should match but PnL should not
        assert comparison["agreement"]["trade_count_match"] is True
        assert comparison["agreement"]["pnl_within_tolerance"] is False
        assert comparison["agreement"]["overall_pass"] is False

    def test_compare_results_drawdown_outside_tolerance(self):
        """Test comparison when drawdown is outside tolerance."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.10,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.12,  # 20% difference, > 1% tolerance
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # Trade count and PnL should match but drawdown should not
        assert comparison["agreement"]["trade_count_match"] is True
        assert comparison["agreement"]["pnl_within_tolerance"] is True
        assert comparison["agreement"]["max_drawdown_within_tolerance"] is False
        assert comparison["agreement"]["overall_pass"] is False

    def test_compare_results_within_tolerance(self):
        """Test comparison when all metrics are within tolerance."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 10,
            "total_pnl": 1000.0,
            "max_drawdown": 0.10,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 10,
            "total_pnl": 1005.0,  # 0.5% difference, < 1% tolerance
            "max_drawdown": 0.101,  # 1% difference, within tolerance
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # All should pass
        assert comparison["agreement"]["trade_count_match"] is True
        assert comparison["agreement"]["pnl_within_tolerance"] is True
        assert comparison["agreement"]["max_drawdown_within_tolerance"] is True
        assert comparison["agreement"]["overall_pass"] is True

    def test_compare_results_zero_pnl(self):
        """Test comparison when PnL is zero."""
        bt_result = {
            "engine": "backtrader",
            "trade_count": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }
        custom_result = {
            "engine": "custom",
            "trade_count": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }

        comparison = compare_results(bt_result, custom_result, tolerance=0.01)

        # Should pass - zero values should be considered equal
        assert comparison["agreement"]["trade_count_match"] is True
        assert comparison["agreement"]["pnl_within_tolerance"] is True
        assert comparison["agreement"]["max_drawdown_within_tolerance"] is True
        assert comparison["agreement"]["overall_pass"] is True
