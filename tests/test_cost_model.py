"""Tests: cost model correctness."""
from __future__ import annotations

import pytest

from quant_rl.backtest.costs import CostModel


def test_cost_zero_slippage():
    c = CostModel(spread_points=0.6, slippage_points=0.0, commission_per_lot=0.0)
    cost = c.total_cost(lots=1.0, direction=1)
    assert cost == pytest.approx(0.3, rel=1e-6)


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
    # half spread * 2 + commission * 2 = 0.3*2 + 3*2
    assert cost == pytest.approx(0.6 + 6.0)

