"""Evaluation rewards must reconcile across steps and missing diagnostics."""

import unittest
from typing import Any, cast

from evaluation.metrics.reward_attribution import RewardAttribution


class RewardAttributionTest(unittest.TestCase):
    def test_reported_and_missing_components_reconcile(self):
        totals = RewardAttribution()
        totals.add(2.0, {"pnl": 1.2, "penalty": 0.5})
        totals.add(-1.0, {"pnl": -1.0})
        totals.add(3.0, None)
        report = cast(dict[str, Any], totals.as_dict())
        self.assertEqual(report["reported_steps"], 2)
        self.assertEqual(report["unreported_steps"], 1)
        self.assertAlmostEqual(report["component_sums"]["pnl"], 0.2)
        self.assertAlmostEqual(report["unattributed_sum"], 3.3)
        self.assertAlmostEqual(report["reward_sum"], 4.0)
        self.assertAlmostEqual(report["reconciliation_error"], 0.0)

    def test_nonfinite_part_is_rejected(self):
        with self.assertRaises(ValueError):
            RewardAttribution().add(1.0, {"pnl": float("nan")})


if __name__ == "__main__":
    unittest.main()
