"""SL/TP fills are the level, or the tick that crosses it. Delay is 0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.data.ticks import TickBook
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


@pytest.mark.unit
def test_through_tick_is_the_fill_not_a_later_bounce() -> None:
    """A $100 stop fills on the tick that trades through it, not 30 ms later.

    The later tick has bounced back inside the stop. A prop-firm fill stays
    on the through-tick, so the loss is worse than $100.
    """
    env = _make_env(max_loss_per_trade_usd=100.0)
    env.reset()
    pos = env.broker.open_position(env.account, (100.0, 100.0), 100.0, 1)
    assert pos is not None
    pos.entry_price = 100.0
    pos.sl_price = 99.0
    pos.tp_price = 102.0
    pos.size = 100.0
    env.position = pos

    bar_time = pd.Timestamp(env._bar_times[env.step_idx])
    cross = bar_time + pd.Timedelta(milliseconds=200)
    later = cross + pd.Timedelta(milliseconds=30)
    ts = np.array([cross.value, later.value], dtype=np.int64)
    env._tickbook = TickBook(
        ts,
        np.array([98.5, 99.4], dtype=np.float64),
        np.array([98.7, 99.6], dtype=np.float64),
    )
    env._fill_delay_ms = 0

    bar = BarView(open=100.0, high=100.4, low=98.4, close=99.2, spread=0.6)
    env._check_sl_tp(bar, fill_bid=100.0, fill_ask=100.6)

    assert env.position is None
    closes = [t for t in env.trade_log if t.get("type") == "stop_close"]
    assert len(closes) == 1
    assert closes[0]["price"] == pytest.approx(98.5)
    assert closes[0]["pnl"] == pytest.approx(-150.0)
    assert closes[0]["pnl"] < -100.0
