"""Unit tests for the distributional metrics engine (PLAN 9, WP-A)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_rl.evaluation.distributions import (
    compute_distribution_metrics,
)


@pytest.mark.unit
class TestDistributionMetrics:
    def test_empty_input_yields_nan_fields(self) -> None:
        # Act
        metrics = compute_distribution_metrics([])

        # Assert
        assert metrics.count == 0
        assert math.isnan(metrics.mean)
        assert math.isnan(metrics.min_value)
        assert math.isnan(metrics.var_05)

    def test_single_observation(self) -> None:
        # Act
        metrics = compute_distribution_metrics([10.0])

        # Assert
        assert metrics.count == 1
        assert metrics.mean == pytest.approx(10.0)
        assert metrics.median == pytest.approx(10.0)
        assert metrics.min_value == pytest.approx(10.0)
        assert metrics.max_value == pytest.approx(10.0)
        assert metrics.std == 0.0

    def test_symmetric_zero_skew(self) -> None:
        # Arrange
        values = [-2.0, -1.0, 0.0, 1.0, 2.0]

        # Act
        metrics = compute_distribution_metrics(values)

        # Assert
        assert metrics.mean == pytest.approx(metrics.median)
        assert metrics.skewness == pytest.approx(0.0, abs=1e-9)

    def test_positive_tail_positive_skew(self) -> None:
        # Arrange
        values = [-1.0, -1.0, 0.0, 1.0, 10.0]

        # Act
        metrics = compute_distribution_metrics(values)

        # Assert
        assert metrics.skewness > 0.0

    def test_negative_tail_negative_skew(self) -> None:
        # Arrange
        values = [-10.0, -1.0, 0.0, 1.0, 1.0]

        # Act
        metrics = compute_distribution_metrics(values)

        # Assert
        assert metrics.skewness < 0.0

    def test_var_and_cvar(self) -> None:
        # Arrange
        values = [-100.0, -80.0, 10.0, 20.0, 30.0]
        expected_var = np.quantile(np.array(values), 0.05)

        # Act
        metrics = compute_distribution_metrics(values)

        # Assert
        assert metrics.var_05 == pytest.approx(expected_var)
        tail = [v for v in values if v <= metrics.var_05]
        if tail:
            assert metrics.cvar_05 == pytest.approx(float(np.mean(tail)))


@pytest.mark.unit
class TestNoDataDeterminism:
    def test_nan_and_inf_are_dropped_and_counted(self) -> None:
        # Arrange
        values = [1.0, float("nan"), 2.0, float("inf"), 3.0, float("-inf")]

        # Act
        metrics = compute_distribution_metrics(values)

        # Assert
        assert metrics.count == 3
        assert metrics.dropped == 3
        assert metrics.mean == pytest.approx(2.0)

    def test_all_invalid_gives_empty_result(self) -> None:
        # Act / Assert
        metrics = compute_distribution_metrics([float("nan"), float("inf")])
        assert metrics.count == 0
        assert math.isnan(metrics.max_value)
