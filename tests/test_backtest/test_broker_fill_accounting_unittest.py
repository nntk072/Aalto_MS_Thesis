"""Quantity and commission accounting at the isolated broker boundary."""

import unittest

from trading.account.state import AccountState
from trading.execution.broker import Broker
from trading.execution.costs import CostModel


class BrokerAccountingTest(unittest.TestCase):
    def setUp(self):
        self.acc = AccountState(initial_balance=100_000.0)
        self.broker = Broker(cost_model=CostModel(commission_per_lot=2.0))

    def test_partial_and_final_fills_reconcile_commissions_once(self):
        pos = self.broker.open_position(self.acc, (100.0, 101.0), 2.0, 1)
        self.assertIsNotNone(pos)
        assert pos is not None
        self.assertEqual(self.acc.balance, 99_996.0)
        partial, _ = self.broker.close_position(self.acc, pos, (105.0, 106.0), 0.5)
        self.assertAlmostEqual(partial, 1.0)
        self.assertAlmostEqual(pos.size, 1.5)
        self.assertAlmostEqual(self.acc.balance, 99_997.0)
        final, _ = self.broker.close_position(self.acc, pos, (107.0, 108.0))
        self.assertAlmostEqual(final, 6.0)
        self.assertEqual(pos.size, 0.0)
        self.assertEqual(pos.margin_used, 0.0)
        self.assertAlmostEqual(self.acc.balance, 100_003.0)

    def test_invalid_open_quantity_does_not_change_cash(self):
        self.assertIsNone(self.broker.open_position(self.acc, (100.0, 101.0), -1.0, 1))
        self.assertIsNone(self.broker.open_position(self.acc, (100.0, 101.0), 0.0, 1))
        self.assertEqual(self.acc.balance, 100_000.0)

    def test_invalid_partial_quantity_cannot_create_or_undo_fills(self):
        pos = self.broker.open_position(self.acc, (100.0, 101.0), 1.0, -1)
        assert pos is not None
        before = (self.acc.balance, pos.size)
        for requested in (-0.1, float("nan")):
            with self.assertRaises(ValueError):
                self.broker.close_position(self.acc, pos, (98.0, 99.0), requested)
            self.assertEqual((self.acc.balance, pos.size), before)

    def test_already_closed_position_rejects_second_realisation(self):
        pos = self.broker.open_position(self.acc, (100.0, 101.0), 1.0, 1)
        assert pos is not None
        self.broker.close_position(self.acc, pos, (103.0, 104.0))
        before = self.acc.balance
        with self.assertRaises(ValueError):
            self.broker.close_position(self.acc, pos, (105.0, 106.0))
        self.assertEqual(self.acc.balance, before)

    def test_invalid_open_quotes_do_not_mutate_account(self):
        for quote in ((float("nan"), 101.0), (102.0, 101.0), (0.0, 1.0)):
            with self.subTest(quote=quote):
                with self.assertRaisesRegex(ValueError, "quote"):
                    self.broker.open_position(self.acc, quote, 1.0, 1)
                self.assertEqual((self.acc.balance, self.acc.equity), (100_000.0, 100_000.0))

    def test_invalid_close_and_valuation_quotes_preserve_position_and_account(self):
        pos = self.broker.open_position(self.acc, (100.0, 101.0), 1.0, 1)
        assert pos is not None
        before = (self.acc.balance, self.acc.equity, pos.size)
        for quote in ((100.0, float("inf")), (103.0, 102.0)):
            with self.subTest(quote=quote):
                with self.assertRaisesRegex(ValueError, "quote"):
                    self.broker.close_position(self.acc, pos, quote, size=0.2)
                with self.assertRaisesRegex(ValueError, "quote"):
                    self.broker.mark_to_market(self.acc, pos, quote)
                self.assertEqual((self.acc.balance, self.acc.equity, pos.size), before)


if __name__ == "__main__":
    unittest.main()
