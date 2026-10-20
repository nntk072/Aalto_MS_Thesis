"""Dependency-light regression for the first observed equity loss."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.reporting.diagnostics import write_bootstrap_cis


class InitialBalanceBootstrapTest(unittest.TestCase):
    def test_first_step_loss_is_in_drawdown_estimate(self):
        equity = pd.Series([90_000.0, 95_000.0])
        with tempfile.TemporaryDirectory() as directory:
            result = write_bootstrap_cis(
                Path(directory),
                equity,
                pd.DataFrame(),
                n_boot=20,
                initial_balance=100_000.0,
            )
        self.assertAlmostEqual(result["max_drawdown"]["estimate"], 0.1)


if __name__ == "__main__":
    unittest.main()
