"""Tests: cost model correctness."""
from __future__ import annotations

import pytest

from quant_rl.backtest.account import AccountState
from quant_rl.backtest.broker import Broker
from quant_rl.backtest.costs import CostModel


def test_cost_zero_slippage():
    c = CostModel(spread_points=0.6, slippage_points=0.0, commission_per_lot=0.0)
    cost = c.total_cost(lots=1.0, direction=1)
    assert cost == pytest.approx(0.0, rel=1e-6)


def test_fill_price_long():
    c = CostModel(spread_points=1.0, slippage_points=0.0)
    fill = c.fill_price(mid_price=100.0, direction=1)
    assert fill == pytest.approx(100.5)


def test_fill_price_short():
    c = CostModel(spread_points=1.0, slippage_points=0.0)
    fill = c.fill_price(mid_price=100.0, direction=-1)
    assert fill == pytest.approx(99.5)


def test_commission_added():
    c = CostModel(spread_points=0.6, commission_per_lot=3.0)
    cost = c.total_cost(lots=2.0, direction=1)
    assert cost == pytest.approx(6.0)


def test_broker_roundtrip_uses_bid_ask_once_long_and_short():
    c = CostModel(spread_points=0.6, slippage_points=0.0, commission_per_lot=0.0)
    b = Broker(cost_model=c, contract_size=1.0)

    for direction in (1, -1):
        acc = AccountState(initial_balance=100_000.0)
        pos = b.open_position(acc, price=100.0, lots=1.0, direction=direction)
        assert pos is not None

        # close at same mid; result should be exactly one full spread loss
        pnl = b.close_position(acc, pos, price=100.0)
        assert pnl == pytest.approx(-0.6, rel=1e-6)

