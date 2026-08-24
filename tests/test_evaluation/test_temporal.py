"""Tests for temporal distributions (PLAN 9, WP-C): holding time + sweep delay."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_rl.evaluation import (
    conditional_pnl_distributions,
    holding_time_distribution,
    sweep_delay_distribution,
)


def _log_open_close(entry_idx: int, exit_idx: int) -> list[dict[str, object]]:
    """Build an open + close pair at fixed 5-min timestamps."""
    start = pd.Timestamp("2025-01-02 16:30")
    return [
        {"type": "open", "time": start + pd.Timedelta(minutes=5 * entry_idx)},
        {"type": "close", "time": start + pd.Timedelta(minutes=5 * exit_idx)},
    ]


@pytest.mark.unit
class TestHoldingTime:
    def test_holding_time_from_entry_to_exit(self) -> None:
        # Arrange — open at 1, close at 4 => 3 bars = 15 minutes = 900s
        trade_log = _log_open_close(1, 4)

        # Act
        dist = holding_time_distribution(trade_log)

        # Assert
        assert dist.count == 1
        assert dist.mean == pytest.approx(900.0)

    def test_fifo_pairs_multiple_trades(self) -> None:
        # Arrange — 2 completed trades
        trade_log = _log_open_close(0, 2) + _log_open_close(3, 5)

        # Act
        dist = holding_time_distribution(trade_log)

        # Assert
        assert dist.count == 2
        assert dist.mean == pytest.approx(600.0)

    def test_empty_log_gives_count_zero(self) -> None:
        # Act
        dist = holding_time_distribution([])

        # Assert
        assert dist.count == 0
        assert math.isnan(dist.mean)


@pytest.mark.unit
class TestSweepDelay:
    def test_collects_finite_delays_only(self) -> None:
        # Arrange
        trade_log = [
            {"type": "open", "sweep_delay_s": 3.0},
            {"type": "open", "sweep_delay_s": float("nan")},
            {"type": "open", "sweep_delay_s": 9.0},
        ]

        # Act
        dist = sweep_delay_distribution(trade_log)

        # Assert
        assert dist.count == 2
        assert dist.dropped == 1
        assert dist.mean == pytest.approx(6.0)

    def test_ignores_non_open_entries(self) -> None:
        # Arrange
        trade_log = [
            {"type": "close", "sweep_delay_s": 99.0},
            {"type": "open", "sweep_delay_s": 2.0},
        ]

        # Act
        dist = sweep_delay_distribution(trade_log)

        # Assert
        assert dist.count == 1
        assert dist.mean == pytest.approx(2.0)


@pytest.mark.unit
class TestConditionalDistributions:
    def _log(self) -> list[dict[str, object]]:
        return [
            {"type": "open", "direction": 1, "level_type": "london_high"},
            {"type": "close", "pnl": 100.0},
            {"type": "open", "direction": -1, "level_type": "asian_low"},
            {"type": "close", "pnl": -50.0},
            {"type": "open", "direction": 1, "level_type": None},
            {"type": "close", "pnl": 25.0},
        ]

    def test_groups_are_present(self) -> None:
        # Act
        groups = conditional_pnl_distributions(self._log())

        # Assert
        assert set(groups) == {"overall", "long", "short", "london", "asian"}
        assert groups["overall"].count == 3

    def test_long_short_split(self) -> None:
        # Arrange
        dist = conditional_pnl_distributions(self._log())

        # Assert
        assert dist["long"].count == 2
        assert dist["short"].count == 1
        assert dist["long"].mean == pytest.approx(62.5)

    def test_level_split(self) -> None:
        # Arrange
        dist = conditional_pnl_distributions(self._log())

        # Assert
        assert dist["london"].count == 1
        assert dist["asian"].count == 1
        assert dist["london"].mean == pytest.approx(100.0)

    def test_empty_log_all_groups_empty(self) -> None:
        # Act
        dist = conditional_pnl_distributions([])

        # Assert
        assert dist["overall"].count == 0
        assert dist["long"].count == 0
