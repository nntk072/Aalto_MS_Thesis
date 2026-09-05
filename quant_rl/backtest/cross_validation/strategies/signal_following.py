"""Backtrader strategy that follows precomputed signal actions.

Used by the engine cross-validation harness to run the exact same signal
series through backtrader and the custom event-driven engine.
"""

from __future__ import annotations

import backtrader as bt


class SignalFollowingStrategy(bt.Strategy):
    """Execute precomputed actions (+1 long, -1 short, 2 exit, 0 hold).

    Parameters
    ----------
    actions : pd.Series
        Precomputed action series aligned to the data feed bars.
    printlog : bool
        Enable per-bar logging.
    """

    params = (
        ("actions", None),
        ("printlog", False),
    )

    def __init__(self) -> None:
        self.order = None
        self.bar_index = 0

    def log(self, txt: str, dt: object = None, doprint: bool = False) -> None:
        if self.p.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            else:
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def next(self) -> None:
        if self.order:
            self.bar_index += 1
            return

        actions = self.p.actions
        if actions is None or self.bar_index >= len(actions):
            self.bar_index += 1
            return

        action = int(actions.iloc[self.bar_index])

        if action == 1:
            self.order = self.buy()
        elif action == -1:
            self.order = self.sell()
        elif action == 2:
            self.order = self.close()

        self.bar_index += 1
