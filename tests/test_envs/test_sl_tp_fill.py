"""SL/TP fills are the level price, not the next quote."""

from __future__ import annotations

import pytest

from quant_rl.envs.feature_row import BarView
from tests.test_envs.test_ny_steps_eod import _make_env


@pytest.mark.unit
def test_wick_through_stop_fills_at_stop_not_next_quote() -> None:
    env = _make_env()
    env.reset()
    pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
    assert pos is not None
    pos.entry_price = 100.0
    pos.sl_price = 99.0
    pos.tp_price = 101.5
    env.position = pos

    # Low trades through the stop; high trades through the target; the next
    # quote is far above entry. Stop wins, and the fill is the stop.
    bar = BarView(open=100.2, high=103.0, low=98.5, close=102.5, spread=0.6)
    env._check_sl_tp(bar, fill_bid=120.0, fill_ask=120.6)

    assert env.position is None
    closes = [t for t in env.trade_log if t.get("type") == "stop_close"]
    assert len(closes) == 1
    assert closes[0]["price"] == pytest.approx(99.0)
    assert closes[0]["pnl"] < 0
    assert closes[0]["reason"] == "structure_sl"
