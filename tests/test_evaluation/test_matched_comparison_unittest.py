"""Fail closed on an unmatched intervention/control experiment."""

import unittest
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics.matched_comparison import compare_matched_returns


def _contract() -> dict[str, Any]:
    return {
        "dataset_sha256": "data-hash",
        "split_id": "fold-1",
        "features_sha256": "feat-hash",
        "reward_sha256": "reward-hash",
        "costs_sha256": "cost-hash",
        "execution_sha256": "execution-hash",
        "timesteps": 100_000,
        "seed": 1,
        "initial_capital": 10_000.0,
        "model_settings_sha256": "model-hash",
        "environment_sha256": "environment-hash",
        "evaluation_start": "2025-01-01T00:00:00",
        "evaluation_end": "2025-01-01T19:00:00",
    }


def _contract_for(series: pd.Series) -> dict[str, Any]:
    if not isinstance(series.index, pd.DatetimeIndex):
        return _contract()
    return _contract() | {
        "evaluation_start": series.index[0].isoformat(),
        "evaluation_end": series.index[-1].isoformat(),
    }


class MatchedComparisonTest(unittest.TestCase):
    def test_nonfinite_paired_returns_cannot_produce_a_confidence_interval(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        valid = pd.Series([0.01, 0.02, -0.01], index=times)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            treatment = valid.copy()
            treatment.iloc[1] = invalid
            with (
                self.subTest(side="treatment", invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                compare_matched_returns(
                    treatment, valid, _contract_for(valid), _contract_for(valid), n_boot=20
                )
            control = valid.copy()
            control.iloc[0] = invalid
            with (
                self.subTest(side="control", invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                compare_matched_returns(
                    valid, control, _contract_for(valid), _contract_for(valid), n_boot=20
                )

    def test_single_observation_cannot_yield_a_bootstrap_interval(self):
        times = pd.date_range("2025-01-01", periods=1, freq="h")
        values = pd.Series([0.01], index=times)
        with self.assertRaisesRegex(ValueError, "at least two"):
            compare_matched_returns(
                values, values, _contract_for(values), _contract_for(values), n_boot=20
            )

    def test_invalid_confidence_level_is_rejected(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        values = pd.Series([0.01, -0.02, 0.005], index=times)
        for level in (0.0, -0.5, 1.0, 1.5, float("nan")):
            with self.subTest(level=level), self.assertRaisesRegex(ValueError, "level"):
                compare_matched_returns(
                    values, values, _contract_for(values), _contract_for(values), level=level
                )

    def test_empty_paired_returns_cannot_produce_research_ci(self):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        with self.assertRaisesRegex(ValueError, "at least one"):
            compare_matched_returns(empty, empty, _contract(), _contract(), n_boot=10)

    def test_nonpositive_bootstrap_count_cannot_claim_interval(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        values = pd.Series([0.01, -0.02, 0.005], index=times)
        with self.assertRaisesRegex(ValueError, "n_boot"):
            compare_matched_returns(
                values, values, _contract_for(values), _contract_for(values), n_boot=0
            )

    def test_integer_index_cannot_prove_timestamp_matching(self):
        returns = pd.Series([0.01, 0.02, -0.01])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            compare_matched_returns(
                returns, returns, _contract_for(returns), _contract_for(returns), n_boot=10
            )

    def test_empty_string_provenance_is_missing(self):
        returns = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2))
        missing = _contract_for(returns) | {"dataset_sha256": ""}
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            compare_matched_returns(returns, returns, missing, missing)

    def test_identical_matched_paths_have_zero_paired_difference(self):
        times = pd.date_range("2025-01-01", periods=20, freq="h")
        series = pd.Series(np.linspace(-0.01, 0.01, 20), index=times)
        ci = compare_matched_returns(
            series, series.copy(), _contract_for(series), _contract_for(series), n_boot=20
        )
        self.assertAlmostEqual(ci.estimate, 0.0)
        self.assertAlmostEqual(ci.lower, 0.0)

    def test_mismatched_costs_and_missing_evidence_are_rejected(self):
        series = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2))
        unmatched = _contract_for(series) | {"costs_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "costs_sha256"):
            compare_matched_returns(series, series, _contract_for(series), unmatched)
        missing = _contract_for(series)
        del missing["features_sha256"]
        with self.assertRaisesRegex(ValueError, "features_sha256"):
            compare_matched_returns(series, series, missing, _contract_for(series))

    def test_initial_capital_model_environment_and_period_must_match(self):
        times = pd.date_range("2025-01-01", periods=20, freq="h")
        series = pd.Series(np.linspace(-0.01, 0.01, 20), index=times)
        for field, value in (
            ("initial_capital", 20_000.0),
            ("model_settings_sha256", "other-model"),
            ("environment_sha256", "other-env"),
            ("evaluation_start", "2025-01-02T00:00:00"),
            ("evaluation_end", "2025-01-02T19:00:00"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare_matched_returns(
                    series, series, _contract_for(series), _contract_for(series) | {field: value}
                )

    def test_one_explicit_intervention_dimension_may_differ(self):
        times = pd.date_range("2025-01-01", periods=20, freq="h")
        control = pd.Series(np.linspace(-0.01, 0.01, 20), index=times)
        treatment = control + 0.001
        common = _contract_for(control)
        intervention = common | {"environment_sha256": "treatment-environment"}

        result = compare_matched_returns(
            treatment,
            control,
            intervention,
            common,
            intervention_field="environment_sha256",
            n_boot=20,
        )

        self.assertAlmostEqual(result.estimate, 0.001)

    def test_declared_intervention_must_be_the_only_difference(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        values = pd.Series([0.01, -0.01, 0.02], index=times)
        control = _contract_for(values)
        treatment = control | {
            "environment_sha256": "treatment-environment",
            "costs_sha256": "unmatched-costs",
        }
        with self.assertRaisesRegex(ValueError, "costs_sha256"):
            compare_matched_returns(
                values,
                values,
                treatment,
                control,
                intervention_field="environment_sha256",
                n_boot=20,
            )

    def test_unsupported_intervention_dimension_is_rejected(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        values = pd.Series([0.01, -0.01, 0.02], index=times)
        contract = _contract_for(values)
        with self.assertRaisesRegex(ValueError, "Unsupported comparison intervention"):
            compare_matched_returns(
                values,
                values,
                contract,
                contract,
                intervention_field="execution_sha256",
                n_boot=20,
            )

    def test_cost_sensitivity_is_a_supported_intervention(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        control_values = pd.Series([0.01, -0.01, 0.02], index=times)
        treatment_values = pd.Series([0.0, -0.02, 0.01], index=times)
        control = _contract_for(control_values)
        treatment = control | {"costs_sha256": "stress-costs"}
        result = compare_matched_returns(
            treatment_values,
            control_values,
            treatment,
            control,
            intervention_field="costs_sha256",
            n_boot=20,
        )
        self.assertAlmostEqual(result.estimate, -0.01)

    def test_declared_evaluation_period_must_cover_return_index(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h")
        series = pd.Series([0.01, -0.01, 0.02], index=times)
        wrong_period = _contract_for(series) | {"evaluation_end": "2025-01-02T02:00:00"}
        with self.assertRaisesRegex(ValueError, "evaluation period"):
            compare_matched_returns(series, series, wrong_period, wrong_period)

    def test_same_length_different_timestamps_are_rejected(self):
        left = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2))
        right = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-02", periods=2))
        with self.assertRaisesRegex(ValueError, "index"):
            compare_matched_returns(left, right, _contract_for(left), _contract_for(left))


if __name__ == "__main__":
    unittest.main()
