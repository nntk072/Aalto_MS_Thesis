"""EarlyAbortCallback stops learn() from the same decision as the equity gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_callback_returns_false_when_the_window_fails() -> None:
    pytest.importorskip("stable_baselines3")
    from quant_rl.train.callbacks import EarlyAbortCallback

    cb = EarlyAbortCallback(min_timesteps=100, window=2)
    cb.model = cast(
        Any, SimpleNamespace(num_timesteps=500, logger=SimpleNamespace(name_to_value={}))
    )
    failing = {
        "end_equity": 90_000.0,
        "start_equity": 100_000.0,
        "slope": -1.0,
        "max_peak_trailing_dd": 0.2,
        "n_trades": 2,
        "breach_reason": "max_loss",
        "reward_sum": 1.0,
    }
    cb.locals = {
        "dones": np.array([True, True]),
        "infos": [{"episode_equity": failing}, {"episode_equity": failing}],
    }
    assert cb.on_step() is False
    assert cb.reason is not None
    assert cb.reason.startswith("equity_gate:")


def test_callback_stops_on_a_nonfinite_rollout_metric() -> None:
    pytest.importorskip("stable_baselines3")
    from quant_rl.train.callbacks import EarlyAbortCallback

    cb = EarlyAbortCallback(min_timesteps=2_000_000, window=4)
    cb.model = cast(
        Any,
        SimpleNamespace(
            num_timesteps=10,
            logger=SimpleNamespace(name_to_value={"train/loss": float("nan")}),
        ),
    )
    cb.locals = {"dones": np.array([False]), "infos": [{}]}
    cb._on_rollout_end()
    assert cb._on_step() is False
    assert cb.reason == "nonfinite"
