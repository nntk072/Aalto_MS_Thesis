"""End-to-end bounded checks for structural exits in the backtest engine."""

import numpy as np
import pandas as pd

from quant_rl.backtest.costs import CostModel
from quant_rl.backtest.engine import _sl_tp_fill_quote, run_backtest
from quant_rl.data.ticks import TickBook


def _bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 98.0, 98.0],
            "high": [101.0, 101.0, 102.0, 99.0, 99.0],
            "low": [99.0, 99.0, 97.0, 97.5, 97.5],
            "close": [100.0, 100.0, 100.0, 98.0, 98.0],
        },
        index=index,
    )


def test_protective_fill_quote_uses_price_unit_spread_and_adverse_slippage() -> None:
    model = CostModel(
        spread_points=0.6,
        point_size=0.01,
        slippage_points=5.0,
    )

    long_bid, long_ask = _sl_tp_fill_quote(1, 100.0, model)
    short_bid, short_ask = _sl_tp_fill_quote(-1, 100.0, model)

    assert (long_bid, long_ask) == (95.0, 95.6)
    assert (short_bid, short_ask) == (104.4, 105.0)


def test_structural_stop_executes_without_optional_max_loss_cap() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    bars = _bars(index)
    features = pd.DataFrame({"last_swing_low": [99.0] * len(index)}, index=index)
    actions = iter([1, 0, 0, 0])

    result = run_backtest(
        bars,
        features,
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01, commission_per_lot=2.0),
        use_structure_sl_tp=True,
        swing_buffer_pts=1.0,
        risk_frac=0.01,
        min_lot=0.01,
        max_lot=1.0,
        hold_on_zero=True,
        block_overnight=False,
    )

    stops = result["trades"].loc[result["trades"]["type"] == "stop_close"]
    assert len(stops) == 1
    assert stops.iloc[0]["reason"] == "structure_sl"
    opened = result["trades"].loc[result["trades"]["type"] == "open"].iloc[0]
    stop = stops.iloc[0]
    assert opened["decision_time"] == index[1]
    assert opened["fill_time"] == index[3]
    assert opened["fill_quote_source"] == "bar_close"
    assert stop["position_id"] == opened["position_id"]
    assert stop["fill_id"] == f"{opened['position_id']}:f1"
    assert stop["position_complete"] is True
    assert stop["time_precision"] == "bar_interval"
    assert stop["event_time_start"] >= opened["fill_time"]
    assert stop["event_time_start"] <= stop["time"] < stop["event_time_end"]
    assert stop["position_net_pnl"] == result["account"].balance - 100_000.0


def test_delayed_tick_exit_records_the_actual_fill_timestamp() -> None:
    index = pd.date_range("2026-01-05 12:00", periods=5, freq="1min", tz="UTC")
    fill_time = index[2]
    actual_fill_time = fill_time + pd.Timedelta(seconds=20)
    ticks = [fill_time, fill_time + pd.Timedelta(seconds=10), actual_fill_time]
    tickbook = TickBook(
        np.asarray([ts.value for ts in ticks], dtype=np.int64),
        np.asarray([100.0, 98.5, 98.0]),
        np.asarray([100.5, 99.0, 98.5]),
    )
    features = pd.DataFrame({"last_swing_low": [99.0] * len(index)}, index=index)
    actions = iter([1, 0, 0, 0])

    result = run_backtest(
        _bars(index),
        features,
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        tickbook=tickbook,
        fill_delay_ms=5_000,
        use_structure_sl_tp=True,
        swing_buffer_pts=0.0,
        risk_frac=0.01,
        min_lot=0.01,
        max_lot=1.0,
        hold_on_zero=True,
        block_overnight=False,
    )

    stop = result["trades"].loc[result["trades"]["type"] == "stop_close"].iloc[0]
    assert stop["time"] == actual_fill_time


def test_backtest_does_not_submit_entry_without_a_following_bar() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    actions = iter([0, 0, 0, 1])
    result = run_backtest(
        _bars(index),
        pd.DataFrame(index=index),
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        hold_on_zero=True,
        block_overnight=False,
    )

    assert result["trades"].empty


def test_backtest_gap_stop_fills_at_the_open_or_worse() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    bars = _bars(index)
    bars.loc[index[3], ["open", "high", "low", "close"]] = [97.0, 98.0, 96.0, 97.0]
    bars.loc[index[2], ["high", "low", "close"]] = [100.5, 99.5, 100.0]
    features = pd.DataFrame({"last_swing_low": [99.0] * len(index)}, index=index)
    actions = iter([1, 0, 0, 0])

    result = run_backtest(
        bars,
        features,
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        use_structure_sl_tp=True,
        swing_buffer_pts=0.0,
        min_lot=0.01,
        max_lot=1.0,
        hold_on_zero=True,
        block_overnight=False,
    )

    stop = result["trades"].loc[result["trades"]["type"] == "stop_close"].iloc[0]
    assert stop["price"] <= 97.0


def test_session_close_uses_last_ny_bar_close_not_after_hours_quote() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    bars = _bars(index)
    bars.loc[index[2], ["open", "high", "low", "close"]] = [100.0, 102.0, 99.0, 101.0]
    bars.loc[index[3], ["open", "high", "low", "close"]] = [102.0, 103.0, 101.0, 102.0]
    bars.loc[index[4], ["open", "high", "low", "close"]] = [120.0, 121.0, 119.0, 120.0]
    bars["session"] = ["ny", "ny", "ny", "ny", "asian"]
    actions = iter([1, 0, 0, 0])

    result = run_backtest(
        bars,
        pd.DataFrame(index=index),
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        hold_on_zero=True,
        block_overnight=True,
    )

    close = result["trades"].loc[result["trades"]["type"] == "eod_close"].iloc[0]
    assert close["price"] == 102.0
    assert close["fill_quote_source"] == "permitted_session_bar_close"
    assert close["time"] == index[3] + pd.Timedelta(minutes=1)
    assert result["equity"].loc[close["time"]] == result["account"].equity


def test_backtest_rejects_entry_filling_on_final_ny_bar() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    bars = _bars(index)
    bars["session"] = ["ny", "ny", "ny", "asian", "asian"]
    actions = iter([1, 0, 0])

    result = run_backtest(
        bars,
        pd.DataFrame(index=index),
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        hold_on_zero=True,
        block_overnight=True,
    )

    assert result["trades"].empty


def test_terminal_liquidation_updates_equity_at_bar_close_time() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=5, freq="1min", tz="UTC")
    actions = iter([1, 1, 1])

    result = run_backtest(
        _bars(index),
        pd.DataFrame(index=index),
        lambda _obs: next(actions),
        obs_window=1,
        cost_model=CostModel(spread_points=0.0, point_size=0.01),
        hold_on_zero=True,
        block_overnight=False,
    )

    terminal = result["trades"].loc[result["trades"]["type"] == "eod_close"].iloc[-1]
    expected_close = index[-1] + pd.Timedelta(minutes=1)
    assert terminal["fill_time"] == expected_close
    assert result["equity"].index[-1] == expected_close
    assert result["equity"].iloc[-1] == result["account"].equity
    assert result["equity_valuation_times"].is_monotonic_increasing
