"""Bounded deterministic real TickBook/Broker/TradingEnv fill reconciliation."""

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.account import AccountState
from quant_rl.backtest.broker import Broker
from quant_rl.backtest.costs import CostModel
from quant_rl.data.ticks import TickBook, TickExecutionUnavailable
from quant_rl.envs.feature_row import BarView
from quant_rl.envs.trading_env import TradingEnv


def test_tick_quote_sides_match_buy_ask_and_sell_bid_for_entries_and_exits() -> None:
    quote = (99.0, 101.0)
    broker = Broker(cost_model=CostModel(spread_points=0.0, commission_per_lot=0.0))

    long_account = AccountState(initial_balance=100_000.0)
    long_position = broker.open_position(long_account, quote, 1.0, 1)
    assert long_position is not None
    assert long_position.entry_price == pytest.approx(101.0)  # Buy at ask.
    _, long_exit = broker.close_position(long_account, long_position, quote)
    assert long_exit == pytest.approx(99.0)  # Sell at bid.

    short_account = AccountState(initial_balance=100_000.0)
    short_position = broker.open_position(short_account, quote, 1.0, -1)
    assert short_position is not None
    assert short_position.entry_price == pytest.approx(99.0)  # Sell at bid.
    _, short_exit = broker.close_position(short_account, short_position, quote)
    assert short_exit == pytest.approx(101.0)  # Buy at ask.


def test_tick_exit_ignores_crossings_before_position_entry_time() -> None:
    start = pd.Timestamp("2026-01-12T12:00:00Z")
    trigger = start + pd.Timedelta(seconds=10)
    entry_time = start + pd.Timedelta(seconds=30)
    book = TickBook(
        np.array([trigger.value], dtype=np.int64),
        np.array([94.0]),
        np.array([94.5]),
    )

    event = book.delayed_exit_event(
        start,
        direction=1,
        sl=95.0,
        tp=None,
        delay=pd.Timedelta(0),
        earliest_time=entry_time,
    )

    assert event is None


@pytest.mark.parametrize("direction", [1, -1])
def test_tick_tp_partial_then_stop_reconciles_commissions_and_reward(direction: int) -> None:
    start = pd.Timestamp("2026-01-12T12:00:00Z")
    ticks = [start + pd.Timedelta(seconds=i) for i in (5, 20)]
    bids = (106.0, 94.0) if direction == 1 else (93.5, 105.5)
    asks = (106.5, 94.5) if direction == 1 else (94.0, 106.0)
    book = TickBook(
        np.array([t.value for t in ticks], dtype=np.int64),
        np.asarray(bids),
        np.asarray(asks),
    )
    account = AccountState(initial_balance=100_000.0)
    broker = Broker(cost_model=CostModel(commission_per_lot=2.0))
    position = broker.open_position(account, (100.0, 100.0), 1.0, direction)
    assert position is not None
    position.position_id = f"trace:{direction}"
    position.entry_timestamp = start
    position.sl_price = 95.0 if direction == 1 else 105.0
    position.tp1_price = 105.0 if direction == 1 else 95.0
    position.tp2_price = 110.0 if direction == 1 else 90.0
    position.tp_initial_size = 1.0
    position.tp_lot_fractions = (0.3, 0.7)
    position.planned_risk_usd = 5.0
    env: Any = SimpleNamespace(
        _tickbook=book,
        position=position,
        account=account,
        broker=broker,
        _fill_delay_ms=0,
        _breakeven_trigger_r=0.0,
        _last_realized_close_pnl=None,
        _last_realized_close_r=None,
        _pending_realized_r=None,
        strategy_actions=False,
        trade_log=[{"type": "open", "position_id": position.position_id}],
        sessions_with_trades=set(),
        _current_session_id=lambda: 1,
    )
    env._stamp_exit = MethodType(TradingEnv._stamp_exit, env)
    env._note_close = MethodType(TradingEnv._note_close, env)

    assert TradingEnv._check_multi_tp_ticks(env, start, 0)
    assert env.position is None
    fills = env.trade_log[1:]
    assert [f["type"] for f in fills] == ["tp_partial_close", "stop_close"]
    assert [f["size"] for f in fills] == pytest.approx([0.3, 0.7])
    assert [f["position_complete"] for f in fills] == [False, True]
    assert len({f["fill_id"] for f in fills}) == 2
    assert sum(f["entry_commission_allocated"] for f in fills) == pytest.approx(2.0)
    assert sum(f["net_pnl"] for f in fills) == pytest.approx(account.balance - 100_000.0)
    assert position.position_net_pnl == pytest.approx(account.balance - 100_000.0)
    assert env._last_realized_close_pnl == pytest.approx(position.position_net_pnl)
    assert env._last_realized_close_r == pytest.approx(position.position_net_pnl / 5.0)
    assert {row["fill_quote_source"] for row in fills} == {"tick"}


def test_tick_crosses_three_tp_slots_once_with_exact_remaining_quantity() -> None:
    start = pd.Timestamp("2026-01-12T12:00:00Z")
    tick = start + pd.Timedelta(seconds=5)
    book = TickBook(np.array([tick.value], dtype=np.int64), np.array([107.0]), np.array([107.5]))
    account = AccountState(initial_balance=100_000.0)
    broker = Broker(cost_model=CostModel(commission_per_lot=0.0))
    pos = broker.open_position(account, (100.0, 100.0), 1.0, 1)
    assert pos is not None
    pos.position_id = "trace:three"
    pos.entry_timestamp = start
    pos.sl_price = 95.0
    pos.tp1_price, pos.tp2_price, pos.tp3_price = (102.0, 104.0, 106.0)
    pos.tp_initial_size = 1.0
    pos.tp_lot_fractions = (0.3, 0.3, 0.4)
    pos.planned_risk_usd = 5.0
    env: Any = SimpleNamespace(
        _tickbook=book,
        position=pos,
        account=account,
        broker=broker,
        _fill_delay_ms=0,
        _breakeven_trigger_r=0.0,
        _last_realized_close_pnl=None,
        _last_realized_close_r=None,
        _pending_realized_r=None,
        strategy_actions=False,
        trade_log=[{"type": "open", "position_id": pos.position_id}],
        sessions_with_trades=set(),
        _current_session_id=lambda: 1,
    )
    env._stamp_exit = MethodType(TradingEnv._stamp_exit, env)
    env._note_close = MethodType(TradingEnv._note_close, env)

    assert TradingEnv._check_multi_tp_ticks(env, start, 0)
    assert env.position is None
    fills = env.trade_log[1:]
    assert [f["tp_slot"] for f in fills] == [1, 2, 3]
    assert [f["size"] for f in fills] == pytest.approx([0.3, 0.3, 0.4])
    assert len({f["fill_id"] for f in fills}) == 3
    assert sum(f["pnl"] for f in fills) == pytest.approx(account.balance - 100_000)
    assert env._last_realized_close_pnl == pytest.approx(account.balance - 100_000)
    assert {row["fill_quote_source"] for row in fills} == {"tick"}


def test_tick_and_bar_match_for_identical_unambiguous_target_crossings() -> None:
    start = pd.Timestamp("2026-01-12T12:00:00Z")
    tick_times = [start + pd.Timedelta(seconds=s) for s in (5, 20)]
    book = TickBook(
        np.array([t.value for t in tick_times], dtype=np.int64),
        np.array([105.0, 110.0]),
        np.array([105.5, 110.5]),
    )

    def make_env(with_ticks: bool) -> Any:
        account = AccountState(initial_balance=100_000.0)
        broker = Broker(cost_model=CostModel(commission_per_lot=1.0))
        pos = broker.open_position(account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        pos.position_id = "trace:parity"
        pos.entry_timestamp = start
        pos.sl_price = 95.0
        pos.tp1_price, pos.tp2_price = 105.0, 110.0
        pos.tp_initial_size = 1.0
        pos.tp_lot_fractions = (0.4, 0.6)
        pos.planned_risk_usd = 5.0
        env: Any = SimpleNamespace(
            position=pos,
            _tickbook=book if with_ticks else None,
            _fill_delay_ms=0,
            _breakeven_trigger_r=0.0,
            _last_realized_close_pnl=None,
            _last_realized_close_r=None,
            _pending_realized_r=None,
            strategy_actions=False,
            account=account,
            broker=broker,
            cost_model=broker.cost_model,
            trade_log=[{"type": "open", "position_id": pos.position_id}],
            sessions_with_trades=set(),
            _current_session_id=lambda: 1,
            _bar_times=[start],
            _bar_time=lambda index: start,
            _tick_exit_quote=lambda at: None,
            _maybe_ema_exit=lambda bar, idx, log_time: None,
        )
        for method in ("_stamp_exit", "_note_close", "_sl_tp_fill_quote", "_check_multi_tp_ticks"):
            setattr(env, method, MethodType(getattr(TradingEnv, method), env))
        return env

    tick_env = make_env(True)
    bar_env = make_env(False)
    bar: Any = SimpleNamespace(low=100.0, high=110.0, close=110.0)
    TradingEnv._check_sl_tp(tick_env, bar, 110.0, 110.5, bar_idx=0)
    TradingEnv._check_sl_tp(bar_env, bar, 110.0, 110.5, bar_idx=0)
    assert tick_env.position is None and bar_env.position is None
    assert {row["fill_quote_source"] for row in tick_env.trade_log[1:]} == {"tick"}
    assert {row["fill_quote_source"] for row in bar_env.trade_log[1:]} == {"bar_range_level"}
    for key in ("tp_slot", "size", "net_pnl", "position_complete"):
        assert [row[key] for row in tick_env.trade_log[1:]] == pytest.approx(
            [row[key] for row in bar_env.trade_log[1:]]
        )
    assert tick_env.account.balance == pytest.approx(bar_env.account.balance)
    assert tick_env._last_realized_close_r == pytest.approx(bar_env._last_realized_close_r)


def test_trading_env_step_reconciles_partial_target_then_stop_reward() -> None:
    index = pd.date_range("2026-01-05T14:00:00Z", periods=8, freq="min")
    close = np.full(len(index), 100.0)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 1000.0),
            "spread": np.zeros(len(index)),
            "session_id": np.zeros(len(index), dtype=np.int64),
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "last_swing_high": np.full(len(index), 105.0),
            "last_swing_low": np.full(len(index), 95.0),
            "london_high": np.full(len(index), 110.0),
            "london_low": np.full(len(index), 90.0),
            "asian_high": np.full(len(index), 110.0),
            "asian_low": np.full(len(index), 90.0),
            "volume_spike": np.ones(len(index)),
        },
        index=index,
    )
    initial_balance = 100_000.0
    env = TradingEnv(
        bars,
        features,
        obs_window=1,
        initial_balance=initial_balance,
        episodic=False,
        use_sweep_reward=False,
        reward_mode="pnl",
        block_overnight=False,
        cost_model=CostModel(spread_points=0.0, commission_per_lot=1.0),
    )
    env.reset(seed=1)

    position = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
    assert position is not None
    position.position_id = "step:partial-stop"
    position.initial_size = 1.0
    position.entry_timestamp = env._bar_time(env.step_idx)
    position.sl_price = 99.5
    position.sl_initial_price = 99.5
    position.planned_risk_usd = 0.5
    position.tp1_price = 101.0
    position.tp2_price = 102.0
    position.tp_initial_size = 1.0
    position.tp_lot_fractions = (0.4, 0.6)
    position.n_tp = 2
    env.position = position

    # The first stepped bar touches TP1 without touching the stop.
    first_idx = env.step_idx
    bars.loc[bars.index[first_idx], "high"] = 101.1
    bars.loc[bars.index[first_idx], "low"] = 99.8
    env._sync_bar_arrays()
    _, partial_reward, _, _, partial_info = env.step(0)

    partial_fill = next(row for row in env.trade_log if row.get("fill_id"))
    assert env.position is position
    assert position.size == pytest.approx(0.6)
    assert partial_fill["size"] == pytest.approx(0.4)
    assert partial_fill["fill_quote_source"] == "bar_range_level"
    assert partial_fill["position_complete"] is False
    assert partial_reward == pytest.approx(partial_fill["net_pnl"] / initial_balance)
    assert partial_info["reward_parts"]["pnl"] == pytest.approx(partial_reward)

    # The next stepped bar stops only the remaining quantity.
    stop_idx = env.step_idx
    bars.loc[bars.index[stop_idx], "low"] = 99.0
    env._sync_bar_arrays()
    _, stop_reward, _, _, stop_info = env.step(0)

    fills = [row for row in env.trade_log if row.get("fill_id")]
    assert [row["size"] for row in fills] == pytest.approx([0.4, 0.6])
    assert [row["position_complete"] for row in fills] == [False, True]
    assert {row["fill_quote_source"] for row in fills} == {"bar_range_level"}
    assert len({row["fill_id"] for row in fills}) == 2
    assert env.position is None
    assert position.position_net_pnl == pytest.approx(env.account.balance - initial_balance)
    assert sum(row["net_pnl"] for row in fills) == pytest.approx(
        env.account.balance - initial_balance
    )
    assert stop_reward == pytest.approx(fills[-1]["net_pnl"] / initial_balance)
    assert stop_info["reward_parts"]["pnl"] == pytest.approx(stop_reward)


@pytest.mark.parametrize("latency_ms", [0.0, 30.0])
def test_tick_mode_uses_correct_ask_and_records_actual_entry_quote(latency_ms: float) -> None:
    index = pd.date_range("2026-01-05T14:30:00Z", periods=8, freq="min")
    close = np.full(len(index), 100.0)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 1000.0),
            "spread": np.zeros(len(index)),
            "session_id": np.ones(len(index), dtype=np.int64),
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "last_swing_high": np.full(len(index), 105.0),
            "last_swing_low": np.full(len(index), 95.0),
            "london_high": np.full(len(index), 99.0),
            "london_low": np.full(len(index), 90.0),
            "asian_high": np.full(len(index), 99.0),
            "asian_low": np.full(len(index), 90.0),
            "volume_spike": np.full(len(index), 2.0),
        },
        index=index,
    )
    tick_times = index + pd.Timedelta(milliseconds=100)
    book = TickBook(
        tick_times.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        np.full(len(index), 100.0),
        np.full(len(index), 100.2),
    )
    env = TradingEnv(
        bars,
        features,
        obs_window=1,
        episodic=False,
        reward_mode="pnl",
        block_overnight=False,
        tickbook=book,
        tick_execution_required=True,
        randomized_fill_latency_ms=latency_ms,
    )
    env.reset(seed=73)

    env.step(1)

    opened = next(row for row in env.trade_log if row.get("type") == "open")
    assert opened["fill_quote_source"] == "tick"
    assert opened["price"] == pytest.approx(100.2)  # Long buys execute at ask.
    if latency_ms:
        assert 0 <= opened["execution_latency_ns"] < int(latency_ms * 1_000_000)
    else:
        assert opened["execution_latency_ns"] == 0
    assert opened["actual_tick_time"] == opened["fill_time"]
    assert opened["execution_arrival_time"] <= opened["actual_tick_time"]
    assert opened["execution_order_time"] == opened["order_submission_time"]
    # TickBook timestamps are UTC, while the bar spine uses the broker zone.
    # The equity index must stay homogeneous when a tick falls after the bar
    # valuation boundary.
    assert env.equity_times[-1].tzinfo is not None
    assert env.equity_times[-1].utcoffset() == pd.Timedelta(hours=3)


@pytest.mark.parametrize("latency_ms", [0.0, 30.0])
@pytest.mark.parametrize("tick_in_bar", [False, True])
def test_tick_only_mode_uses_quotes_for_candle_only_stop_fill(
    latency_ms: float, tick_in_bar: bool
) -> None:
    index = pd.date_range("2026-01-05T14:30:00Z", periods=4, freq="min")
    close = np.full(len(index), 100.0)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 1000.0),
            "spread": np.zeros(len(index)),
            "session_id": np.ones(len(index), dtype=np.int64),
        },
        index=index,
    )
    features = pd.DataFrame(index=index)
    tick_time = (index[1] if tick_in_bar else index[-1]) + pd.Timedelta(seconds=10)
    book = TickBook(np.array([tick_time.value]), np.array([100.0]), np.array([100.2]))
    env = TradingEnv(
        bars,
        features,
        obs_window=1,
        block_overnight=False,
        tickbook=book,
        tick_execution_required=True,
        randomized_fill_latency_ms=latency_ms,
    )
    env.reset(seed=73)
    pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
    assert pos is not None
    pos.entry_timestamp = env._bar_time(env.step_idx) - pd.Timedelta(minutes=1)
    pos.sl_price = 99.5
    env.position = pos
    bar = BarView(open=100.0, low=99.0, high=100.5, close=100.0)

    if tick_in_bar:
        env._check_sl_tp(bar, 100.0, 100.2, env.step_idx)
        assert env.position is pos
        assert not any(row.get("type") == "stop_close" for row in env.trade_log)
    else:
        with pytest.raises(TickExecutionUnavailable, match="no tick observations"):
            env._check_sl_tp(bar, 100.0, 100.2, env.step_idx)


if __name__ == "__main__":
    for side in (1, -1):
        test_tick_tp_partial_then_stop_reconciles_commissions_and_reward(side)
    test_tick_crosses_three_tp_slots_once_with_exact_remaining_quantity()
    test_tick_and_bar_match_for_identical_unambiguous_target_crossings()
    print("4 bounded tick/broker/env trace cases PASSED")
