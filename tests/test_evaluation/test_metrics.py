"""Unit tests for quant_rl.evaluation.metrics (hand-computed fixtures)."""

from __future__ import annotations

import math

import pytest

from quant_rl.evaluation.metrics import PerformanceMetrics, compute_metrics, max_drawdown


@pytest.mark.unit
class TestMaxDrawdown:
    def test_zero_when_equity_only_rises(self) -> None:
        # Arrange
        equity = [100.0, 110.0, 120.0]

        # Act
        mdd = max_drawdown(equity)

        # Assert
        assert mdd == 0.0

    def test_peak_to_trough_decline(self) -> None:
        # Arrange
        equity = [100.0, 120.0, 90.0, 95.0]

        # Act
        mdd = max_drawdown(equity)

        # Assert
        assert mdd == pytest.approx(0.25)


@pytest.mark.unit
class TestComputeMetrics:
    def test_flat_equity_gives_zero_sharpe_and_no_trades(self) -> None:
        # Arrange
        equity = [100_000.0] * 10

        # Act
        metrics = compute_metrics(equity, initial_balance=100_000.0)

        # Assert
        assert metrics.sharpe == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.n_trades == 0

    def test_total_pnl_and_return(self) -> None:
        # Arrange
        equity = [100_000.0, 101_000.0, 102_000.0]

        # Act
        metrics = compute_metrics(equity, initial_balance=100_000.0)

        # Assert
        assert metrics.total_pnl == pytest.approx(2_000.0)
        assert metrics.total_return_pct == pytest.approx(2.0)

    def test_trade_stats_mixed_outcomes(self) -> None:
        # Arrange
        equity = [100_000.0, 101_000.0]
        trades = [500.0, -200.0, 300.0, -100.0]

        # Act
        metrics = compute_metrics(equity, initial_balance=100_000.0, trade_pnls=trades)

        # Assert
        assert metrics.n_trades == 4
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.expectancy == pytest.approx(125.0)
        assert metrics.profit_factor == pytest.approx(800.0 / 300.0)

    def test_profit_factor_infinite_without_losses(self) -> None:
        # Arrange
        equity = [100_000.0, 100_500.0]

        # Act
        metrics = compute_metrics(equity, initial_balance=100_000.0, trade_pnls=[250.0])

        # Assert
        assert math.isinf(metrics.profit_factor)

    def test_breach_count_is_propagated(self) -> None:
        # Arrange
        equity = [100_000.0, 90_000.0]

        # Act
        metrics = compute_metrics(
            equity,
            initial_balance=100_000.0,
            breach_count=1,
        )

        # Assert
        assert metrics.breach_count == 1

    def test_extras_are_merged(self) -> None:
        # Arrange
        equity = [100_000.0, 100_001.0]

        # Act
        metrics = compute_metrics(
            equity,
            initial_balance=100_000.0,
            extras={"sweep_delay_s": 2.5},
        )

        # Assert
        assert metrics.extras["sweep_delay_s"] == pytest.approx(2.5)

    def test_empty_curve_raises(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="equity_curve"):
            compute_metrics([], initial_balance=100_000.0)

    def test_non_positive_balance_raises(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="initial_balance"):
            compute_metrics([1.0], initial_balance=0.0)

    def test_result_is_immutable(self) -> None:
        # Arrange
        metrics = compute_metrics([100_000.0], initial_balance=100_000.0)

        # Act / Assert
        assert isinstance(metrics, PerformanceMetrics)
        with pytest.raises(AttributeError):
            metrics.sharpe = 1.0  # type: ignore[misc]
