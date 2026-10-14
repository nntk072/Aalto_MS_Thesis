"""Bounded event parity between the custom engine and Backtrader.

Both engines consume the same precomputed action trace and close-price fills;
this checks their shared accounting path, not strategy-indicator parity.
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from quant_rl.backtest.costs import CostModel
from quant_rl.backtest.engine import run_backtest


class _SignalFeed(bt.feeds.PandasData):  # type: ignore[misc]
    lines = ("signal",)
    params = (("signal", "signal"),)


class _DelayedSignalStrategy(bt.Strategy):  # type: ignore[misc]
    params = (("last_bar", 0),)

    def __init__(self) -> None:
        self.order = None

    def notify_order(self, order) -> None:
        if order.status not in (order.Submitted, order.Accepted):
            self.order = None

    def next(self) -> None:
        if self.order is not None:
            return
        if len(self) == self.p.last_bar:
            if self.position:
                self.order = self.close()
            return
        if len(self) < 3:
            return

        # run_backtest observes feature[i - 1] while iterating bar i, then
        # executes at bar i + 1 close. Read that same signal at this bar close.
        action = int(self.data.signal[-2])
        if action == 2 and self.position:
            self.order = self.close()
        elif action == 1 and not self.position:
            self.order = self.buy(size=1)
        elif action == -1 and not self.position:
            self.order = self.sell(size=1)


@pytest.mark.parametrize(
    ("direction", "close_prices"),
    [
        (1, [100.0, 100.0, 100.0, 102.0, 101.0, 104.0, 103.0, 105.0]),
        (-1, [100.0, 100.0, 100.0, 98.0, 99.0, 96.0, 97.0, 95.0]),
    ],
)
def test_same_action_trace_reconciles_between_custom_engine_and_backtrader(
    direction: int, close_prices: list[float]
) -> None:
    # A final duplicate quote gives Backtrader one broker cycle to acknowledge
    # the close-on-close order submitted on the preceding close.
    close_prices = [*close_prices, close_prices[-1]]
    index = pd.date_range("2026-02-02T14:00:00Z", periods=len(close_prices), freq="min")
    bars = pd.DataFrame(
        {
            "open": close_prices,
            "high": [price + 0.25 for price in close_prices],
            "low": [price - 0.25 for price in close_prices],
            "close": close_prices,
            "volume": 1000.0,
        },
        index=index,
    )
    actions = pd.Series(0, index=index, dtype="int64")
    actions.iloc[2] = direction
    actions.iloc[5] = 2
    features = pd.DataFrame({"signal": actions}, index=index)
    cost = CostModel(
        spread_points=0.0,
        slippage_points=0.0,
        commission_per_lot=0.0,
    )

    custom = run_backtest(
        bars,
        features,
        lambda observation: int(observation[-1, 0]),
        obs_window=1,
        cost_model=cost,
        initial_balance=100_000.0,
        lots=1.0,
        hold_on_zero=True,
        exit_action=2,
        block_overnight=False,
    )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.set_coc(True)
    cerebro.adddata(_SignalFeed(dataname=bars.assign(signal=actions)))
    cerebro.addstrategy(_DelayedSignalStrategy, last_bar=len(bars))
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    strategy = cerebro.run()[0]

    custom_closes = custom["trades"].loc[custom["trades"]["type"] == "close"]
    bt_analysis = strategy.analyzers.trades.get_analysis()
    bt_closed = int(bt_analysis.get("total", {}).get("closed", 0))

    assert len(custom_closes) == 1
    assert bt_closed == 1
    assert custom["account"].equity == pytest.approx(cerebro.broker.getvalue())
    bt_realized = sum(
        float(bucket.get("pnl", {}).get("total", 0.0))
        for outcome in ("won", "lost")
        if (bucket := bt_analysis.get(outcome, {}))
    )
    assert custom["trades"].iloc[-1]["net_pnl"] == pytest.approx(bt_realized)
