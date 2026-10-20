"""Numerical metric adapters must accept ordinary ndarray PnL series."""

import unittest

import pandas as pd

from evaluation.metrics import calculate_metrics, compute_metrics


class MetricInputContractTest(unittest.TestCase):
    def test_numpy_trade_pnls_count_completed_trades(self):
        result = compute_metrics(
            [100_000.0, 99_000.0, 101_000.0],
            initial_balance=100_000.0,
            trade_pnls=[-1_000.0, 2_000.0],
        )
        self.assertEqual(result.n_trades, 2)
        self.assertAlmostEqual(result.win_rate, 0.5)
        self.assertAlmostEqual(result.expectancy, 500.0)

    def test_initial_first_loss_is_in_drawdown(self):
        result = compute_metrics([90_000.0], initial_balance=100_000.0)
        self.assertAlmostEqual(result.max_drawdown, 0.10)

    def test_completed_position_stats_include_partial_fills_once(self):
        events = pd.DataFrame(
            [
                {"position_id": "p1", "type": "open", "pnl": None},
                {"position_id": "p1", "type": "tp_partial_close", "pnl": 200.0},
                {"position_id": "p1", "type": "tp_close", "pnl": -50.0},
                {"position_id": "p2", "type": "open", "pnl": None},
                {"position_id": "p2", "type": "tp_partial_close", "pnl": 300.0},
            ]
        )
        result = calculate_metrics([100_000.0, 100_150.0], events, initial_balance=100_000.0)
        self.assertEqual(result.n_trades, 1)
        self.assertAlmostEqual(result.win_rate, 1.0)
        self.assertAlmostEqual(result.expectancy, 150.0)
        self.assertEqual(result.extras["fill_count"], 3.0)
        self.assertEqual(result.extras["incomplete_positions"], 1.0)

    def test_position_summary_uses_net_pnl_and_counts_fills_separately(self):
        events = pd.DataFrame(
            [
                {
                    "position_id": "p1",
                    "type": "open",
                    "pnl": None,
                    "position_complete": True,
                    "position_net_pnl": 130.0,
                },
                {
                    "position_id": "p1",
                    "type": "tp_partial_close",
                    "pnl": 100.0,
                    "net_pnl": 90.0,
                    "position_complete": False,
                },
                {
                    "position_id": "p1",
                    "type": "tp_close",
                    "pnl": 50.0,
                    "net_pnl": 40.0,
                    "position_complete": True,
                    "position_net_pnl": 130.0,
                },
                {
                    "position_id": "p2",
                    "type": "open",
                    "pnl": None,
                    "position_complete": False,
                },
                {
                    "position_id": "p2",
                    "type": "tp_partial_close",
                    "pnl": 300.0,
                    "net_pnl": 280.0,
                    "position_complete": False,
                },
            ]
        )
        result = calculate_metrics([100_000.0, 100_130.0], events, initial_balance=100_000.0)

        self.assertEqual(result.n_trades, 1)
        self.assertEqual(result.expectancy, 130.0)
        self.assertEqual(result.extras["fill_count"], 3.0)
        self.assertEqual(result.extras["incomplete_positions"], 1.0)


if __name__ == "__main__":
    unittest.main()
