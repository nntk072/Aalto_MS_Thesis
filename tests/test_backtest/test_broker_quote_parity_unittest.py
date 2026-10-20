"""Deterministic bid/ask parity at the isolated broker boundary."""

import unittest

from trading.account.state import AccountState
from trading.execution.broker import Broker
from trading.execution.costs import CostModel


class BrokerQuoteParityTest(unittest.TestCase):
    def test_mirrored_long_short_round_trip_and_mark_to_market(self):
        quote = (99.5, 100.5)
        for direction in (1, -1):
            with self.subTest(direction=direction):
                account = AccountState(initial_balance=100_000.0)
                broker = Broker(cost_model=CostModel(commission_per_lot=2.0))
                position = broker.open_position(account, quote, 1.0, direction)
                self.assertIsNotNone(position)
                assert position is not None
                self.assertEqual(position.entry_price, 100.5 if direction == 1 else 99.5)
                self.assertAlmostEqual(account.balance, 99_998.0)
                self.assertAlmostEqual(broker.mark_to_market(account, position, quote), -1.0)
                self.assertAlmostEqual(account.equity, 99_997.0)

                pnl1, fill1 = broker.close_position(account, position, quote, 0.4)
                pnl2, fill2 = broker.close_position(account, position, quote)
                expected_exit = 99.5 if direction == 1 else 100.5
                self.assertEqual((fill1, fill2), (expected_exit, expected_exit))
                self.assertAlmostEqual(pnl1, -1.2)
                self.assertAlmostEqual(pnl2, -1.8)
                self.assertAlmostEqual(position.size, 0.0)
                self.assertAlmostEqual(account.balance, 99_995.0)
                self.assertAlmostEqual(account.equity, 99_995.0)


if __name__ == "__main__":
    unittest.main()
