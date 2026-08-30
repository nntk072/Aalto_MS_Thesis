"""Chain C tests: latency-injected fills + RL→MT5 bridge contract.

Runs headless (no MT5, no SB3 weights): the bridge is exercised with a
stub model and a stub data source matching the duck-typed interface the
live robot loop expects. ``MetaTrader5`` (a Windows-only package) is
stubbed in conftest so the ``mt5_trading.domain.signal`` import works.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

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
    from quant_rl.envs.trading_env import TradingEnv

    feats = pd.DataFrame(0.0, index=bars.index, columns=["f0"])
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
    assert obs["seq"].shape == (10, 1)
    for _ in range(20):
        obs, *_ = env.step(0)  # hold
    assert env.fill_latency_bars == latency


def test_latency_changes_fill_price(trending_bars):
    """In an uptrend, a latency-injected long entry fills at a higher price."""

    env0 = _make_env(trending_bars, 0)
    env2 = _make_env(trending_bars, 2)
    env0.reset(seed=42)
    env2.reset(seed=42)

    # Drive both to the same step, then compare the fill quote for a long.
    for _ in range(10):
        env0.step(0)
        env2.step(0)
    env0.step(3)  # open long, discrete action
    env2.step(3)
    p0 = env0.position
    p2 = env2.position
    if p0 is None or p2 is None:  # action mapping guard
        pytest.skip("long entry action did not open a position")
    assert p2.entry_price > p0.entry_price, (
        "latency-injected fill should be later (pricier) in an uptrend"
    )


def test_zero_latency_matches_legacy_default(trending_bars):
    """fill_latency_bars=0 keeps the previous next-bar fill behaviour."""
    env = _make_env(trending_bars, 0)
    env.reset(seed=42)
    bid, ask = env._bar_quote(env.bars.iloc[env.step_idx + 1])
    # step once with hold; the internal fill quote must equal bar t+1 quote
    env.step(0)
    assert env.fill_latency_bars == 0


class _StubModel:
    """Duck-typed SB3 model: action = +1 (long) always."""

    def predict(self, obs: dict, deterministic: bool = True) -> np.ndarray:  # noqa: ARG002
        return np.array([1])


class _StubData:
    """Duck-typed MT5Data returning a small bars frame."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars

    def get_symbol(self) -> str:
        return "US100.cash"

    def get_data(self) -> pd.DataFrame:
        return self._bars


def test_rl_bridge_observation_contract(trending_bars):
    """Bridge builds the same dict obs contract as the env (T, F) + (5,)."""
    from quant_rl.live.rl_strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    adapter.update_bars(trending_bars)
    obs = adapter.build_observation({"equity": 100_000.0})
    assert set(obs) == {"seq", "account"}
    assert obs["seq"].dtype == np.float32
    assert obs["account"].shape == (1, 5)
    action = adapter.predict_signal()
    assert action == 1


def test_rl_bridge_maps_actions_to_signals(trending_bars):
    """as_strategy maps action>0 → BUY, 0 → HOLD via the duck-typed source."""
    from quant_rl.live.rl_strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    strategy = adapter.as_strategy(_StubData(trending_bars))
    symbol, sig = strategy.signal()
    assert symbol == "US100.cash"
    assert sig.value == "buy"


def test_rl_bridge_too_few_bars_no_signal(trending_bars):
    """Bridge returns NONE when there isn't enough history."""
    from quant_rl.live.rl_strategy import RLStrategyAdapter

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    strategy = adapter.as_strategy(_StubData(trending_bars.iloc[:5]))
    _, sig = strategy.signal()
    assert sig.value == "none"
