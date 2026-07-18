"""Unit tests for trade metrics computation (MAE/MFE/SL/TP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.eval.trade_metrics import compute_trade_metrics


@pytest.fixture()
def sample_bars() -> pd.DataFrame:
    """Create sample M1 bars for testing."""
    idx = pd.date_range("2025-01-02 16:30", periods=100, freq="1min")
    data = {
        "open": 100.0 + np.sin(np.arange(100) * 0.1) * 5,
        "high": 102.0 + np.sin(np.arange(100) * 0.1) * 5,
        "low": 98.0 + np.sin(np.arange(100) * 0.1) * 5,
        "close": 100.5 + np.sin(np.arange(100) * 0.1) * 5,
    }
    return pd.DataFrame(data, index=idx)


def test_long_mae_mfe(sample_bars: pd.DataFrame) -> None:
    """Test MAE/MFE calculation for a long trade."""
    open_row = pd.Series(
        {
            "time": sample_bars.index[10],
            "direction": 1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[50],
            "price": 101.0,
            "pnl": 50.0,
        }
    )

    metrics = compute_trade_metrics(sample_bars, open_row, close_row)

    # For a long trade, MFE should be max high, MAE should be min low
    assert metrics.direction == 1
    assert metrics.entry_price == 100.0
    assert metrics.exit_price == 101.0

    # MAE should be minimum low in window
    trade_window = sample_bars.loc[sample_bars.index[10] : sample_bars.index[50]]
    expected_mae = trade_window["low"].min()
    expected_mfe = trade_window["high"].max()

    np.testing.assert_approx_equal(metrics.mae_price, expected_mae, significant=5)
    np.testing.assert_approx_equal(metrics.mfe_price, expected_mfe, significant=5)


def test_short_mae_mfe(sample_bars: pd.DataFrame) -> None:
    """Test MAE/MFE calculation for a short trade."""
    open_row = pd.Series(
        {
            "time": sample_bars.index[10],
            "direction": -1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[50],
            "price": 99.0,
            "pnl": 50.0,
        }
    )

    metrics = compute_trade_metrics(sample_bars, open_row, close_row)

    # For a short trade, MFE should be min low, MAE should be max high
    assert metrics.direction == -1

    trade_window = sample_bars.loc[sample_bars.index[10] : sample_bars.index[50]]
    expected_mae = trade_window["high"].max()  # worst case for short
    expected_mfe = trade_window["low"].min()  # best case for short

    np.testing.assert_approx_equal(metrics.mae_price, expected_mae, significant=5)
    np.testing.assert_approx_equal(metrics.mfe_price, expected_mfe, significant=5)


def test_sl_calculation_long() -> None:
    """Test SL calculation for a long trade."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": 1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 100.5,
            "pnl": 0.0,
        }
    )

    metrics = compute_trade_metrics(
        sample_bars,
        open_row,
        close_row,
        max_loss_per_trade_usd=10.0,
        lots=1.0,
        contract_size=1.0,
    )

    # For a long trade: SL = entry - (max_loss / (lots * contract_size))
    expected_sl = 100.0 - 10.0
    assert metrics.sl_price == expected_sl


def test_sl_calculation_short() -> None:
    """Test SL calculation for a short trade."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": -1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 99.5,
            "pnl": 50.0,
        }
    )

    metrics = compute_trade_metrics(
        sample_bars,
        open_row,
        close_row,
        max_loss_per_trade_usd=10.0,
        lots=1.0,
        contract_size=1.0,
    )

    # For a short trade: SL = entry + (max_loss / (lots * contract_size))
    expected_sl = 100.0 + 10.0
    assert metrics.sl_price == expected_sl


def test_tp_calculation_long() -> None:
    """Test TP calculation for a long trade."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": 1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 100.5,
            "pnl": 20.0,
        }
    )

    metrics = compute_trade_metrics(
        sample_bars,
        open_row,
        close_row,
        take_profit_per_trade_usd=20.0,
        lots=1.0,
        contract_size=1.0,
    )

    # For a long trade: TP = entry + (tp_profit / (lots * contract_size))
    expected_tp = 100.0 + 20.0
    assert metrics.tp_price == expected_tp


def test_tp_calculation_short() -> None:
    """Test TP calculation for a short trade."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": -1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 99.5,
            "pnl": 20.0,
        }
    )

    metrics = compute_trade_metrics(
        sample_bars,
        open_row,
        close_row,
        take_profit_per_trade_usd=20.0,
        lots=1.0,
        contract_size=1.0,
    )

    # For a short trade: TP = entry - (tp_profit / (lots * contract_size))
    expected_tp = 100.0 - 20.0
    assert metrics.tp_price == expected_tp


def test_sl_tp_none_when_not_configured() -> None:
    """Test that SL/TP are None when not configured."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": 1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 100.5,
            "pnl": 0.0,
        }
    )

    metrics = compute_trade_metrics(sample_bars, open_row, close_row)

    assert metrics.sl_price is None
    assert metrics.tp_price is None


def test_sl_tp_with_different_lots_and_contract_size() -> None:
    """Test SL/TP calculation with non-standard lots and contract size."""
    sample_bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
        },
        index=[pd.Timestamp("2025-01-02 16:30")],
    )

    open_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "direction": 1,
            "price": 100.0,
        }
    )
    close_row = pd.Series(
        {
            "time": sample_bars.index[0],
            "price": 100.5,
            "pnl": 0.0,
        }
    )

    # With lots=2 and contract_size=0.5, price_move = 10 / (2 * 0.5) = 10
    metrics = compute_trade_metrics(
        sample_bars,
        open_row,
        close_row,
        max_loss_per_trade_usd=10.0,
        take_profit_per_trade_usd=20.0,
        lots=2.0,
        contract_size=0.5,
    )

    assert metrics.sl_price == 100.0 - 10.0
    assert metrics.tp_price == 100.0 + 20.0
