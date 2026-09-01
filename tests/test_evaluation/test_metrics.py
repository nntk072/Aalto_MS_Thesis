"""Unit tests for quant_rl.evaluation.metrics (hand-computed fixtures)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from quant_rl.evaluation.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    compute_metrics,
    max_drawdown,
    sweep_delay_breakdown,
)


@pytest.mark.unit
class TestSweepDelayBreakdown:
    def test_summarises_delays_and_level_split(self) -> None:
        # Arrange
        trade_log: list[dict[str, Any]] = [
            {"type": "open", "sweep_delay_s": 10.0, "level_type": "london_high"},
            {"type": "close", "pnl": 100.0},
            {"type": "open", "sweep_delay_s": 30.0, "level_type": "asian_low"},
            {"type": "open", "sweep_delay_s": float("nan"), "level_type": "asian_high"},
            {"type": "open", "level_type": None},  # entry without level info
        ]

        # Act
        stats = sweep_delay_breakdown(trade_log)

        # Assert
        assert stats["n_entries"] == 4
        assert stats["sweep_delay_mean_s"] == pytest.approx(20.0)
        assert stats["sweep_delay_median_s"] == pytest.approx(20.0)
        assert stats["london_pct"] == pytest.approx(100.0 / 3.0)
        assert stats["asian_pct"] == pytest.approx(200.0 / 3.0)

    def test_empty_log_gives_zeros(self) -> None:
        # Act
        stats = sweep_delay_breakdown([])

        # Assert
        assert stats["n_entries"] == 0
        assert stats["sweep_delay_mean_s"] == 0.0


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

    def test_legacy_fields_are_populated(self) -> None:
        # Arrange — 2 losses, 1 win, 2 losses in a row at the end
        equity = [100_000.0, 100_000.0, 100_000.0, 100_000.0, 100_000.0, 100_000.0]
        trades = [-100.0, 200.0, -50.0, -30.0]

        # Act
        metrics = compute_metrics(
            equity,
            initial_balance=100_000.0,
            trade_pnls=trades,
            breach_count=2,
            n_sessions=4,
        )

        # Assert
        assert metrics.max_consec_loss == 2
        assert metrics.avg_trade == pytest.approx(metrics.expectancy)
        assert metrics.turnover == pytest.approx(4 / 6)
        assert metrics.breach_rate == pytest.approx(2 / 4)

    def test_breach_rate_zero_sessions_is_safe(self) -> None:
        # Act
        metrics = compute_metrics(
            [100_000.0, 100_001.0], initial_balance=100_000.0, breach_count=1, n_sessions=0
        )

        # Assert
        assert metrics.breach_rate == 0.0
        assert metrics.breach_count == 1


@pytest.mark.unit
class TestCalculateMetricsAdapter:
    def test_accepts_pandas_series_and_trade_frame(self) -> None:
        # Arrange
        equity = pd.Series([100_000.0, 100_500.0, 100_200.0])
        trades = pd.DataFrame({"pnl": [600.0, -400.0, 100.0]})

        # Act
        metrics = calculate_metrics(equity, trades=trades, n_sessions=2, n_breach_sessions=1)

        # Assert
        assert metrics.total_pnl == pytest.approx(200.0)
        assert metrics.n_trades == 3
        assert metrics.win_rate == pytest.approx(2 / 3)
        assert metrics.max_consec_loss == 1
        assert metrics.breach_rate == pytest.approx(0.5)

    def test_m1_annualisation_default_is_preserved(self) -> None:
        # The adapter keeps the legacy 252*390 default so migrated runners
        # produce the same annualised numbers as before the unification.
        from quant_rl.evaluation.metrics import LEGACY_M1_PERIODS_PER_YEAR

        assert LEGACY_M1_PERIODS_PER_YEAR == 252 * 390

    def test_none_trades_gives_no_trade_stats(self) -> None:
        # Act
        metrics = calculate_metrics(pd.Series([100.0, 101.0]))

        # Assert
        assert metrics.n_trades == 0
        assert metrics.max_consec_loss == 0
