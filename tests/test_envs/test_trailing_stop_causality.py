"""Integration checks for confirmed structure swings driving trailing stops."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from features.technical.structure import structure_levels
from simulation.environment.env import TradingEnv
from trading.execution.broker import Position


@pytest.mark.parametrize(
    ("direction", "field", "reference", "initial_stop"),
    [
        (1, "last_swing_low", 101.0, 95.0),
        (-1, "last_swing_high", 99.0, 105.0),
        (1, "last_swing_low", float("nan"), 95.0),
        (-1, "last_swing_high", float("inf"), 105.0),
    ],
)
def test_trailing_stop_ignores_invalid_or_wrong_side_references(
    direction: int, field: str, reference: float, initial_stop: float
) -> None:
    env: Any = SimpleNamespace(
        sl_buffer_pts=0.1,
        _finite=lambda row, key: row.get(key),
    )
    pos: Any = SimpleNamespace(
        sl_mode="trailing",
        direction=direction,
        sl_price=initial_stop,
    )
    TradingEnv._apply_trailing_stop(env, pos, cast(Any, {field: reference, "close": 100.0}))
    assert pos.sl_price == pytest.approx(initial_stop)


def test_trailing_stop_uses_swing_only_from_its_confirmation_bar() -> None:
    index = pd.date_range("2026-01-05 14:00", periods=10, freq="1min", tz="UTC")
    close = np.asarray([100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0])
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "session": "ny",
            "session_id": 1,
        },
        index=index,
    )
    # A later gap through the moved stop exercises the fill behavior after
    # the causally confirmed update has taken effect.
    bars.loc[index[5], ["open", "high", "low", "close"]] = [96.5, 97.0, 96.0, 96.5]
    structure = structure_levels(bars, swing_period=2, atr_mult=0.0)
    features = structure[["last_swing_high", "last_swing_low"]].copy()
    features["close"] = close
    env = TradingEnv(bars=bars, features=features, obs_window=1, sl_buffer_pts=0.1)
    env.reset(seed=7)
    pos = Position(
        direction=1,
        size=0.7,
        entry_price=100.0,
        margin_used=0.0,
        sl_price=95.0,
        sl_initial_price=95.0,
        stop_distance=5.0,
        planned_risk_usd=350.0,
        sl_mode="trailing",
        position_id="causal:trail",
    )

    # The pivot low occurs at bar 2 and is confirmed at bar 4 (right=2).
    assert np.isnan(features["last_swing_low"].iloc[3])
    assert features["last_swing_low"].iloc[4] == pytest.approx(97.5)

    TradingEnv._apply_trailing_stop(
        env,
        pos,
        env._feature_row_at(3),
        bar_idx=3,
        event_time=env._bar_time(3),
    )
    assert pos.sl_price == pytest.approx(95.0)
    TradingEnv._apply_trailing_stop(
        env,
        pos,
        env._feature_row_at(4),
        bar_idx=4,
        event_time=env._bar_time(4),
    )
    assert pos.sl_price == pytest.approx(97.4)
    update = next(row for row in env.trade_log if row.get("type") == "sl_update")
    assert update["position_id"] == pos.position_id
    assert update["reference"] == "last_swing_low"
    assert update["reference_price"] == pytest.approx(97.5)
    assert update["time"] == env._bar_time(4)
    assert update["effective_time"] == env._bar_time(5)
    # Management may tighten the live stop, but the original risk used for
    # sizing and R-multiple attribution remains anchored to entry.
    assert pos.sl_initial_price == pytest.approx(95.0)
    assert pos.stop_distance == pytest.approx(5.0)
    assert pos.planned_risk_usd == pytest.approx(350.0)

    pos.initial_size = 1.0
    pos.filled_size = 0.3
    pos.position_id = "causal:partial"
    env.position = pos
    env.trade_log.append({"type": "open", "position_id": pos.position_id})
    env.step_idx = 5
    env._check_sl_tp(env._bar_at(5), fill_bid=96.5, fill_ask=96.5, bar_idx=5)
    assert env.position is None
    exit_fill = next(row for row in env.trade_log if row.get("fill_id"))
    assert exit_fill["price"] <= 96.5
    assert exit_fill["position_complete"] is True
