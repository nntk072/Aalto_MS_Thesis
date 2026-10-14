"""RED/GREEN tests for position-level and incremental fill accounting."""

from types import SimpleNamespace
from typing import Any

import pytest

from quant_rl.backtest.broker import Position
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.evaluation.runner import run_episode


def test_partial_fills_accumulate_once_and_complete_only_at_final_fill() -> None:
    pos = Position(
        direction=1,
        size=0.7,
        entry_price=100.0,
        margin_used=0.0,
        sl_ref="swing_low",
        tp_ref="target",
        stop_distance=5.0,
        planned_risk_usd=50.0,
        tp_initial_size=1.0,
    )
    pos.position_id = "e1:p1"
    pos.initial_size = 1.0
    env: Any = SimpleNamespace(
        position=pos,
        trade_log=[{"type": "open", "position_id": "e1:p1"}],
        _last_realized_close_pnl=None,
        _last_realized_close_r=None,
        _pending_realized_r=None,
        broker=SimpleNamespace(cost_model=SimpleNamespace(total_cost=lambda qty: 0.0)),
    )
    first = {"type": "tp_partial_close", "pnl": 30.0, "size": 0.3}
    TradingEnv._stamp_exit(env, first, 110.0)
    assert first["position_complete"] is False
    assert env._last_realized_close_pnl == pytest.approx(30.0)
    env.trade_log.append(first)

    pos.size = 0.0
    second = {"type": "tp_close", "pnl": 20.0, "size": 0.7}
    TradingEnv._stamp_exit(env, second, 110.0)
    assert second["position_complete"] is True
    assert first["fill_id"] != second["fill_id"]
    assert env._last_realized_close_pnl == pytest.approx(50.0)
    assert env._last_realized_close_r == pytest.approx(1.0)
    assert env.trade_log[0]["position_net_pnl"] == pytest.approx(50.0)


def test_evaluation_counts_two_fills_as_one_completed_position() -> None:
    class Episode:
        initial_balance = 100.0
        breach_events: list[object] = []
        trade_log = [
            {"type": "open", "position_id": "e1:p1"},
            {
                "type": "tp_partial_close",
                "position_id": "e1:p1",
                "pnl": -10.0,
                "net_pnl": -10.0,
                "position_complete": False,
            },
            {
                "type": "tp_close",
                "position_id": "e1:p1",
                "pnl": 20.0,
                "net_pnl": 20.0,
                "position_complete": True,
            },
        ]

        def reset(self):
            return {}, {}

        def step(self, action):
            return {}, 0.0, True, False, {"equity": 110.0}

    metrics = run_episode(Episode(), lambda _obs: 0)
    assert metrics.n_trades == 1
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.expectancy == pytest.approx(10.0)
