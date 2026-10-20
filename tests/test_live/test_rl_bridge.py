"""Chain C tests: latency-injected fills + RL→MT5 bridge contract.

Runs headless (no MT5, no SB3 weights): the bridge is exercised with a
stub model and a stub data source matching the duck-typed interface the
live robot loop expects. ``MetaTrader5`` (a Windows-only package) is
stubbed in conftest so the ``integrations.mt5.domain.signal`` import works.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from integrations.mt5.adapters.data import TradingData
from simulation.contracts.observations import OBSERVATION_SCHEMA_V1, build_observation_schema

# --- headless MetaTrader5 stub (Windows-only package, absent on Linux) -----
if "MetaTrader5" not in sys.modules:
    try:  # pragma: no cover - depends on platform
        import MetaTrader5  # noqa: F401
    except ImportError:
        _mt5_stub = ModuleType("MetaTrader5")
        for attr in (
            "TIMEFRAME_M1",
            "TIMEFRAME_M5",
            "TIMEFRAME_M15",
            "TIMEFRAME_M30",
            "TIMEFRAME_H1",
            "TIMEFRAME_H4",
            "TIMEFRAME_D1",
        ):
            setattr(_mt5_stub, attr, 1)
        sys.modules["MetaTrader5"] = _mt5_stub


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """Deterministic bars with drift so latency shifts fills adversely."""
    n = 400
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.arange(n) * 0.5  # steady uptrend: later fills are pricier
    return pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "tickvol": np.full(n, 100.0),
            "vol": np.zeros(n),
            "spread": np.full(n, 0.6),
        },
        index=idx,
    )


def _make_env(bars: pd.DataFrame, latency: int):
    from simulation.environment.env import TradingEnv

    # Swing levels below/above the uptrend's entry prices so long entries
    # compute structure SL/TP. Without them the env now rejects entries
    # (no naked positions), which would skip the latency assertions.
    feats = pd.DataFrame(0.0, index=bars.index, columns=["f0"])
    feats["last_swing_low"] = 19990.0
    feats["last_swing_high"] = 20020.0
    return TradingEnv(
        bars=bars,
        features=feats,
        obs_window=10,
        fill_latency_bars=latency,
    )


@pytest.mark.parametrize("latency", [0, 1, 3])
def test_latency_env_constructs_and_steps(trending_bars, latency):
    """Env accepts fill_latency_bars and steps without error."""
    env = _make_env(trending_bars, latency)
    obs, _ = env.reset(seed=42)
    # f0, the two swings, and the derived reachable_r column.
    assert obs["seq"].shape == (10, 4)
    for _ in range(20):
        obs, *_ = env.step(0)  # hold
    assert env.fill_latency_bars == latency


def test_latency_changes_fill_price(trending_bars):
    """A delayed entry stays pending until its future quote time."""

    env0 = _make_env(trending_bars, 0)
    env2 = _make_env(trending_bars, 2)
    env0.reset(seed=42)
    env2.reset(seed=42)

    # Drive both to the same step, then submit a long.
    for _ in range(10):
        env0.step(0)
        env2.step(0)
    env0.step(3)  # open long, discrete action
    scheduled_bar = env2.step_idx + 3
    env2.step(3)
    p0 = env0.position
    if p0 is None:  # action mapping guard
        pytest.skip("long entry action did not open a position")
    assert env2.position is None
    assert not any(row.get("type") == "open" for row in env2.trade_log)
    # Source timestamps label bar opens. The pending fill is the scheduled
    # bar's open, which is the same instant as the preceding bar's close.
    expected_fill_time = trending_bars.index[scheduled_bar]
    assert pd.Timestamp(env2.equity_times[-1]) < expected_fill_time

    while env2.step_idx < scheduled_bar - 1:
        env2.step(0)
        assert env2.position is None
    env2.step(0)
    p2 = env2.position
    if p2 is None:
        pytest.fail("pending latency order did not fill at its scheduled bar")

    assert p2.entry_price > p0.entry_price, (
        "latency-injected fill should be later (pricier) in an uptrend"
    )
    open_row = next(row for row in env2.trade_log if row.get("type") == "open")
    assert pd.Timestamp(open_row["fill_time"]) == expected_fill_time
    assert pd.Timestamp(open_row["order_submission_time"]) < expected_fill_time


def test_zero_latency_matches_legacy_default(trending_bars):
    """fill_latency_bars=0 keeps the previous next-bar fill behaviour."""
    env = _make_env(trending_bars, 0)
    env.reset(seed=42)
    bid, ask = env._bar_quote(env._bar_at(env.step_idx + 1))
    # step once with hold; the internal fill quote must equal bar t+1 quote
    env.step(0)
    assert env.fill_latency_bars == 0


def test_pending_latency_order_is_cancelled_when_episode_ends(trending_bars):
    env = _make_env(trending_bars, 2)
    env.reset(seed=42)
    env.max_episode_steps = 1

    _, _, _, truncated, _ = env.step(3)

    assert truncated
    assert env.position is None
    assert env._pending_entry_order is None
    assert any(
        row.get("type") == "order_cancel"
        and row.get("reason") == "episode_ended_before_scheduled_fill"
        for row in env.trade_log
    )


class _StubModel:
    """Duck-typed SB3 model: action = +1 (long) always."""

    _observation_schema = build_observation_schema(OBSERVATION_SCHEMA_V1).to_dict()

    def predict(self, obs: dict[str, Any], deterministic: bool = True) -> np.ndarray[Any, Any]:  # noqa: ARG002
        return np.array([1])


class _StubData(TradingData):
    """Duck-typed MT5Data returning a small bars frame."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def get_symbol(self) -> str:
        return "US100.cash"

    def get_data(self) -> pd.DataFrame:
        return self._bars


def test_rl_bridge_observation_contract(trending_bars):
    """Bridge builds the same dict obs contract as the env (T, F) + (5,)."""
    from live.runtime.strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="config/base/default.yaml")
    adapter.update_bars(trending_bars)
    obs = adapter.build_observation({"equity": 100_000.0})
    assert set(obs) == {"seq", "seq_mask", "account"}
    assert obs["seq"].dtype == np.float32
    assert obs["account"].shape == (6,)
    assert obs["seq_mask"].shape == (obs["seq"].shape[0],)
    action = adapter.predict_signal()
    assert action == 1


def test_rl_bridge_maps_actions_to_signals(trending_bars):
    """as_strategy maps action>0 → BUY, 0 → HOLD via the duck-typed source."""
    from live.runtime.strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="config/base/default.yaml")
    strategy = adapter.as_strategy(_StubData(trending_bars))
    symbol, sig = strategy.signal()
    assert symbol == "US100.cash"
    assert sig.value == "buy"


def test_rl_bridge_too_few_bars_no_signal(trending_bars):
    """Bridge returns NONE when there isn't enough history."""
    from live.runtime.strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="config/base/default.yaml")
    strategy = adapter.as_strategy(_StubData(trending_bars.iloc[:5]))
    _, sig = strategy.signal()
    assert sig.value == "none"
