"""Tests: cost model + broker bid/ask correctness."""

from __future__ import annotations

import pytest

from quant_rl.backtest.account import AccountState
from quant_rl.backtest.broker import Broker
from quant_rl.backtest.costs import CostModel


def test_fill_price_buy_is_ask():
    c = CostModel(spread_points=1.0, slippage_points=0.0)
    fill = c.fill_price(bid=99.5, ask=100.5, direction=1)
    assert fill == pytest.approx(100.5)


def test_fill_price_sell_is_bid():
    c = CostModel(spread_points=1.0, slippage_points=0.0)
    fill = c.fill_price(bid=99.5, ask=100.5, direction=-1)
    assert fill == pytest.approx(99.5)


def test_total_cost_is_commission_only():
    c = CostModel(spread_points=0.6, commission_per_lot=3.0)
    assert c.total_cost(lots=2.0) == pytest.approx(6.0)


def test_bar_quote_derives_ask_from_close():
    c = CostModel(spread_points=0.6, point_size=0.01)
    bid, ask = c.bar_quote(close=20000.0)
    assert bid == pytest.approx(20000.0)
    assert ask == pytest.approx(20000.6)


def test_long_opens_at_ask_closes_at_bid():
    cm = CostModel(spread_points=1.0, slippage_points=0.0, commission_per_lot=0.0)
    broker = Broker(cost_model=cm, margin_pct=0.02)
    acc = AccountState(initial_balance=100_000.0)

    pos = broker.open_position(acc, quote=(99.5, 100.5), lots=1.0, direction=1)
    assert pos is not None
    assert pos.entry_price == pytest.approx(100.5)

    pnl, fill_price = broker.close_position(acc, pos, quote=(99.5, 100.5))
    assert pnl == pytest.approx(-1.0)
    assert fill_price == pytest.approx(99.5)


def test_round_trip_spread_charged_once():
    cm = CostModel(spread_points=1.0, slippage_points=0.0, commission_per_lot=0.0)
    broker = Broker(cost_model=cm, margin_pct=0.02)
    acc = AccountState(initial_balance=100_000.0)

    quote = (100.0, 101.0)
    pos = broker.open_position(acc, quote=quote, lots=1.0, direction=1)
    pnl, fill_price = broker.close_position(acc, pos, quote=quote)
    assert pnl == pytest.approx(-1.0)
    assert fill_price == pytest.approx(100.0)
