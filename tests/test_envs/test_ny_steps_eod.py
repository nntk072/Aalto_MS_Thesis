"""NY-only env steps, observation window, overnight replay, EOD modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_rl.data.session import add_session_id, add_session_labels
from quant_rl.envs.trading_env import TradingEnv


def _ohlc_index(idx: pd.DatetimeIndex, close: np.ndarray[Any, Any]) -> pd.DataFrame:
    bars = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tickvol": 50,
            "volume": 2000,
            "vol": 0,
            "spread": 0.6,
            "gap_flag": False,
        },
        index=idx,
    )
    bars.index.name = "datetime"
    bars = add_session_labels(bars)
    bars = add_session_id(bars)
    return bars


def _two_session_frame(obs_window: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx1 = pd.date_range("2025-01-06 15:00", periods=180, freq="1min", tz="Etc/GMT-3")
    idx2 = pd.date_range("2025-01-07 15:00", periods=180, freq="1min", tz="Etc/GMT-3")
    idx = pd.DatetimeIndex(idx1.append(idx2))
    close = 20000.0 + np.arange(len(idx), dtype=float) * 0.1
    bars = _ohlc_index(idx, close)
    features = pd.DataFrame(
        {
            "london_high": 20500.0,
            "london_low": 19500.0,
            "asian_high": 20300.0,
            "asian_low": 19700.0,
            "volume_spike": 1.0,
            "last_swing_high": close + 50,
            "last_swing_low": close - 50,
            "atr_5": np.full(len(idx), 2.0),
        },
        index=idx,
    )
    return bars, features


def _make_env(**kwargs: Any) -> TradingEnv:
    bars, features = _two_session_frame()
    kwargs.setdefault(
        "eod_risk",
        {"max_age_hours": 1000.0, "stale_bars": 10_000, "max_loss_usd": 1e12},
    )
    return TradingEnv(bars=bars, features=features, obs_window=10, **kwargs)


@pytest.mark.unit
def test_env_steps_only_ny_bars() -> None:
    env = _make_env()
    env.reset()
    times = []
    for _ in range(20):
        times.append(env.bars.index[env.step_idx])
        env.step(0)
    for ts in times:
        t = pd.Timestamp(ts).tz_convert("Etc/GMT-3").time()
        assert t >= pd.Timestamp("2000-01-01 16:30").time()
        assert t <= pd.Timestamp("2000-01-01 23:00").time()


@pytest.mark.unit
def test_ny_open_observation_contains_pre_ny_bars() -> None:
    env = _make_env()
    env.reset()
    first_ny = env.bars.index[env.step_idx]
    assert pd.Timestamp(first_ny).strftime("%H:%M") == "16:30"
    start = env.step_idx - env.obs_window
    oldest = env.bars.index[start]
    assert pd.Timestamp(oldest).strftime("%H:%M") == "16:20"


@pytest.mark.unit
def test_ny_open_window_not_previous_ny() -> None:
    env = _make_env()
    env.reset()
    ny_times = env.bars.index[env._ny_indices]
    day2_open = [t for t in ny_times if pd.Timestamp(t).day == 7][0]
    abs_i = int(np.flatnonzero(env.bars.index == day2_open)[0])
    env.step_idx = abs_i
    env._ny_pos = int(np.where(env._ny_indices == abs_i)[0][0])
    start = env.step_idx - env.obs_window
    oldest = pd.Timestamp(env.bars.index[start])
    latest = pd.Timestamp(env.bars.index[env.step_idx])
    assert oldest.day == latest.day
    assert oldest.hour == 16 and oldest.minute == 20


@pytest.mark.unit
def test_observation_window_indices() -> None:
    env = _make_env()
    env.reset()
    assert pd.Timestamp(env.bars.index[env.step_idx]).strftime("%H:%M") == "16:30"
    oldest = env.bars.index[env.step_idx - 10]
    assert pd.Timestamp(oldest).strftime("%H:%M") == "16:20"
    env.step(0)
    assert pd.Timestamp(env.bars.index[env.step_idx]).strftime("%H:%M") == "16:31"


@pytest.mark.unit
def test_block_overnight_true_force_close() -> None:
    env = _make_env(block_overnight=True)
    env.reset()
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    env.step_idx = int(last_d1)
    env._ny_pos = int(np.where(env._ny_indices == last_d1)[0][0])
    env.position = env.broker.open_position(env.account, (20000.0, 20000.6), 0.1, 1)
    assert env.position is not None
    env.step(0)
    assert env.position is None
    assert any(t.get("reason") == "session_end" for t in env.trade_log)


@pytest.mark.unit
def test_block_overnight_false_no_force_close() -> None:
    env = _make_env(block_overnight=False)
    env.reset()
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    env.step_idx = int(last_d1)
    env._ny_pos = int(np.where(env._ny_indices == last_d1)[0][0])
    env.position = env.broker.open_position(env.account, (20000.0, 20000.6), 0.1, 1)
    assert env.position is not None
    env.position.sl_price = 1.0
    env.position.tp_price = 1e9
    env.position.entry_timestamp = env.bars.index[env.step_idx]
    env.position.entry_atr = 2.0
    env.step(0)
    assert env.position is not None


@pytest.mark.unit
def test_position_survives_session_boundary() -> None:
    env = _make_env(block_overnight=False)
    env.reset()
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    env.step_idx = int(last_d1)
    env._ny_pos = int(np.where(env._ny_indices == last_d1)[0][0])
    pos = env.broker.open_position(env.account, (20000.0, 20000.6), 0.1, 1)
    assert pos is not None
    pos.sl_price = 1.0
    pos.tp_price = 1e9
    pos.entry_timestamp = env.bars.index[env.step_idx]
    pos.entry_atr = 2.0
    env.position = pos
    env.step(0)
    assert env.position is not None
    assert env.position.entry_price == pos.entry_price


@pytest.mark.unit
def test_skipped_bars_replay_hits_sl() -> None:
    env = _make_env(block_overnight=False)
    env.reset()
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    env.step_idx = int(last_d1)
    env._ny_pos = int(np.where(env._ny_indices == last_d1)[0][0])
    skip = int(last_d1) + 1
    env.bars.loc[env.bars.index[skip], "low"] = 100.0
    pos = env.broker.open_position(env.account, (20000.0, 20000.6), 0.1, 1)
    assert pos is not None
    pos.sl_price = 150.0
    pos.tp_price = 1e12
    pos.entry_timestamp = env.bars.index[env.step_idx]
    pos.entry_atr = 2.0
    env.position = pos
    env.step(0)
    assert env.position is None
    assert any(t.get("type") == "stop_close" for t in env.trade_log)


@pytest.mark.unit
def test_fill_latency_last_ny_bar() -> None:
    env = _make_env(block_overnight=True, fill_latency_bars=0)
    env.reset()
    last_d1 = int([i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1])
    fill_idx = last_d1 + 1
    fill_ts = pd.Timestamp(env.bars.index[fill_idx])
    next_ny = int(env._ny_indices[int(np.where(env._ny_indices == last_d1)[0][0]) + 1])
    next_ny_ts = pd.Timestamp(env.bars.index[next_ny])
    assert fill_ts != next_ny_ts
    assert fill_ts < next_ny_ts


@pytest.mark.unit
def test_eod_episode_continuation() -> None:
    env = _make_env(block_overnight=False, max_episode_steps=500)
    obs, _ = env.reset()
    done = truncated = False
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    while env.step_idx <= last_d1 and not (done or truncated):
        obs, _, done, truncated, _ = env.step(0)
    assert not done
    assert pd.Timestamp(env.bars.index[env.step_idx]).day == 7


@pytest.mark.unit
def test_eod_age_uses_timestamp() -> None:
    env = _make_env(
        block_overnight=False,
        eod_risk={"max_age_hours": 0.01, "stale_bars": 10_000, "max_loss_usd": 1e9},
    )
    env.reset()
    env.position = env.broker.open_position(env.account, (20000.0, 20000.6), 0.1, 1)
    assert env.position is not None
    env.position.entry_timestamp = env.bars.index[env.step_idx] - pd.Timedelta(hours=2)
    env.position.entry_atr = 2.0
    env.position.sl_price = 1.0
    env.step(0)
    assert any(t.get("reason") == "max_age" for t in env.trade_log)


@pytest.mark.unit
def test_ftmo_daily_reset_on_ny_jump() -> None:
    env = _make_env(block_overnight=True, episodic=False)
    env.reset()
    env.step(0)
    env.account.daily_loss = 123.0
    last_d1 = [i for i in env._ny_indices if pd.Timestamp(env.bars.index[i]).day == 6][-1]
    env.step_idx = int(last_d1)
    env._ny_pos = int(np.where(env._ny_indices == last_d1)[0][0])
    env.step(0)
    env.step(0)
    assert env.account.daily_loss == pytest.approx(0.0)


@pytest.mark.unit
def test_engine_force_closes_last_ny_bar() -> None:
    from quant_rl.backtest.engine import run_backtest

    bars, features = _two_session_frame()
    opened = {"done": False}

    def policy(_obs: object) -> int:
        if not opened["done"]:
            opened["done"] = True
            return 1
        return 0

    result = run_backtest(
        bars,
        features,
        policy,
        obs_window=10,
        hold_on_zero=True,
        block_overnight=True,
    )
    trades = result["trades"]
    assert len(trades) > 0
    assert (trades["type"] == "eod_close").any()
    assert (trades["reason"] == "session_end").any()
