"""Unit tests for quant_rl.evaluation.report and evaluation.walkforward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_rl.evaluation import PerformanceMetrics, compute_metrics
from quant_rl.evaluation.report import (
    aggregate_seeds,
    build_comparison_table,
    build_run_report,
    build_summary_table,
    save_metrics_json,
)
from quant_rl.evaluation.walkforward import purged_walk_forward


def _metrics(total_pnl: float, n_trades: int) -> PerformanceMetrics:
    equity = [100_000.0, 100_000.0 + total_pnl]
    pnls = [total_pnl / n_trades] * n_trades if n_trades else []
    return compute_metrics(equity, initial_balance=100_000.0, trade_pnls=pnls)


@pytest.mark.unit
class TestAggregateSeeds:
    def test_one_row_per_seed(self) -> None:
        # Arrange
        results = [_metrics(100.0, 2), _metrics(-50.0, 3), _metrics(0.0, 0)]

        # Act
        df = aggregate_seeds(results)

        # Assert
        assert len(df) == 3
        assert list(df["seed"]) == [0, 1, 2]
        assert {"sharpe", "max_drawdown", "n_trades", "breach_rate"} <= set(df.columns)

    def test_rejects_non_metrics(self) -> None:
        with pytest.raises(ValueError, match="PerformanceMetrics"):
            aggregate_seeds([{"sharpe": 1.0}])  # type: ignore[list-item]

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_seeds([])


@pytest.mark.unit
class TestReportTables:
    def test_summary_table_contains_key_rows(self) -> None:
        # Act
        table = build_summary_table(_metrics(150.0, 2))

        # Assert
        for label in ("Sharpe", "Max Drawdown", "Breach Rate", "Max Consec Loss", "Turnover"):
            assert label in table

    def test_comparison_table_train_and_test_columns(self) -> None:
        # Act
        table = build_comparison_table(_metrics(100.0, 2), _metrics(-30.0, 1))

        # Assert
        assert "Train" in table
        assert "Test" in table
        assert "Sharpe" in table

    def test_save_metrics_json_roundtrip(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "metrics.json"

        # Act
        save_metrics_json(_metrics(120.0, 2), path)
        data = json.loads(path.read_text())

        # Assert
        assert data["n_trades"] == 2
        assert "breach_rate" in data
        assert "turnover" in data


@pytest.mark.unit
class TestBuildRunReport:
    def test_merges_metrics_delay_and_context(self) -> None:
        # Arrange
        trade_log = [{"type": "open", "sweep_delay_s": 5.0, "level_type": "london_high"}]

        # Act
        report = build_run_report(_metrics(80.0, 1), trade_log, run_name="r1")

        # Assert
        assert report["run_name"] == "r1"
        assert report["total_pnl"] == pytest.approx(80.0)
        assert report["sweep_delay"]["sweep_delay_mean_s"] == pytest.approx(5.0)
        assert report["sweep_delay"]["n_entries"] == 1

    def test_none_trade_log_is_safe(self) -> None:
        report = build_run_report(_metrics(0.0, 0), None)
        assert report["sweep_delay"]["n_entries"] == 0


@pytest.mark.unit
class TestPurgedWalkForward:
    def test_folds_are_purged_and_embargoed(self) -> None:
        # Act
        splits = list(
            purged_walk_forward(1000, n_splits=3, test_size=0.2, purge_bars=10, embargo_bars=5)
        )

        # Assert
        assert len(splits) == 3
        for split in splits:
            assert split.train_idx[-1] < split.test_idx[0]
            # embargo + purge gap between train end and test start
            gap = split.test_idx[0] - split.train_idx[-1]
            assert gap >= 10 + 5

    def test_extreme_purge_yields_no_fold(self) -> None:
        splits = list(
            purged_walk_forward(
                50, n_splits=5, test_size=0.2, purge_bars=10_000, embargo_bars=10_000
            )
        )
        assert splits == []
