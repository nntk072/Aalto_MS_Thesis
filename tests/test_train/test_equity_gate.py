"""Train-year equity gate: net rise, not a green trade on every bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.train.equity_gate import assess_train_equity, equity_slope


@pytest.mark.unit
def test_slope_positive_when_equity_trends_up() -> None:
    eq = pd.Series([100_000.0, 99_500.0, 100_400.0, 101_000.0])
    assert equity_slope(eq) > 0.0


@pytest.mark.unit
def test_gate_allows_a_losing_day_when_the_year_rises() -> None:
    # +1000 then -500, still above the start, still trading.
    eq = pd.Series([100_000.0, 101_000.0, 100_500.0, 101_200.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=False)
    assert result["ok"] is True
    assert result["reason"] == "ok"


@pytest.mark.unit
def test_gate_fails_when_the_account_dies_on_the_cap() -> None:
    eq = pd.Series([100_000.0, 101_800.0, 89_845.0, 89_845.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=True)
    assert result["ok"] is False
    assert "max_loss_stopped_trading" in result["reason"]
    assert "end_not_above_start" in result["reason"]
    assert "peak_trailing_dd" in result["reason"]
    assert result["max_peak_trailing_dd"] > 0.10
    assert equity_slope(eq) < 0.0


@pytest.mark.unit
def test_episode_equity_recorded_when_the_episode_ends() -> None:
    from tests.test_envs.test_ny_steps_eod import _make_env

    env = _make_env(max_episode_steps=1)
    env.reset()
    _obs, _reward, done, truncated, info = env.step(0)
    assert done or truncated
    ep = info["episode_equity"]
    assert ep["start_equity"] == pytest.approx(env.initial_balance)
    assert ep["end_equity"] == pytest.approx(env.account.equity)
    assert ep["n_trades"] == 0
    assert "reward_sum" in ep
    assert np.isfinite(ep["reward_sum"])


@pytest.mark.unit
def test_peak_dd_flattens_at_the_quote_and_leaves_the_stop() -> None:
    from tests.test_envs.test_ny_steps_eod import _make_env

    env = _make_env(peak_trailing_dd_limit=0.10)
    env.reset()
    pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
    assert pos is not None
    pos.entry_price = 100.0
    pos.sl_price = 90.0
    pos.tp_price = 130.0
    env.position = pos
    env.account.peak_equity = 110_000.0
    env.account.balance = 98_000.0
    env.account.equity = 98_000.0
    blocked = env._apply_peak_trailing_dd(100.0, 100.6, env.bars.index[env.step_idx])
    assert blocked is True
    assert env.position is None
    close = env.trade_log[-1]
    assert close["reason"] == "peak_dd_behavior"
    assert close["price"] == pytest.approx(100.0)
    assert close["price"] != pytest.approx(90.0)
