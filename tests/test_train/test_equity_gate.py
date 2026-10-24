"""Train-year equity gate: net rise, not a green trade on every bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning.rl.callbacks import EpisodeEquityCallback
from learning.rl.stopping import decide_early_abort
from training.callbacks.equity import assess_train_equity, equity_slope


@pytest.mark.unit
def test_slope_positive_when_equity_trends_up() -> None:
    eq = pd.Series([100_000.0, 99_500.0, 100_400.0, 101_000.0])
    assert equity_slope(eq) > 0.0


@pytest.mark.unit
def test_gate_allows_a_losing_day_when_risk_limits_hold() -> None:
    # +1000 then -500, still above the start, still trading.
    eq = pd.Series([100_000.0, 101_000.0, 100_500.0, 101_200.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=False)
    assert result["ok"] is True
    assert result["reason"] == "ok"


@pytest.mark.unit
def test_net_loss_and_negative_slope_are_metrics_not_gate_failures() -> None:
    eq = pd.Series([100_000.0, 99_500.0, 99_200.0, 98_900.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=False)
    assert result["ok"] is True
    assert result["equity_delta"] == pytest.approx(-1_100.0)
    assert result["slope"] < 0.0
    assert result["reason"] == "ok"


@pytest.mark.unit
def test_peak_drawdown_over_ten_percent_is_reported_but_not_a_gate() -> None:
    eq = pd.Series([100_000.0, 120_000.0, 100_000.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=False)
    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["max_peak_trailing_dd"] == pytest.approx(1 / 6)
    assert result["peak_trailing_dd_limit"] == 0.0


@pytest.mark.unit
def test_configured_peak_drawdown_limit_reports_policy_miss() -> None:
    eq = pd.Series([100_000.0, 120_000.0, 107_500.0])
    result = assess_train_equity(
        eq,
        initial=100_000.0,
        breached=False,
        peak_trailing_dd_limit=0.10,
    )
    assert result["ok"] is False
    assert result["reason"] == "peak_trailing_dd"
    assert result["peak_dd_policy_enabled"] is True
    assert result["peak_dd_policy_ok"] is False
    assert result["hard_risk_ok"] is True
    assert result["max_peak_trailing_dd"] == pytest.approx(0.10416666666666667)
    assert result["peak_trailing_dd_limit"] == 0.10


@pytest.mark.unit
def test_gate_fails_when_the_account_dies_on_the_cap() -> None:
    eq = pd.Series([100_000.0, 101_800.0, 89_845.0, 89_845.0])
    result = assess_train_equity(eq, initial=100_000.0, breached=True, peak_trailing_dd_limit=0.10)
    assert result["ok"] is False
    assert "max_loss_stopped_trading" in result["reason"]
    assert "peak_trailing_dd" in result["reason"]
    assert "end_not_above_start" not in result["reason"]
    assert "slope_not_positive" not in result["reason"]
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
    assert ep["episode_steps"] == 1
    assert ep["completed_data_split"] is False
    assert ep["episode_end_reason"] == "max_episode_steps"
    assert ep["peak_trailing_dd_limit"] == pytest.approx(env.peak_trailing_dd_limit)
    assert "reward_sum" in ep
    assert np.isfinite(ep["reward_sum"])
    assert np.isfinite(ep["slope"])
    assert np.isfinite(ep["max_peak_trailing_dd"])


@pytest.mark.unit
def test_episode_equity_callback_persists_full_split_coverage(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    callback = EpisodeEquityCallback(tmp_path / "episode_equity.csv")
    monkeypatch.setattr(callback, "model", SimpleNamespace(num_timesteps=99_504), raising=False)
    monkeypatch.setattr(
        callback,
        "locals",
        {
            "dones": np.asarray([True]),
            "infos": [
                {
                    "episode_equity": {
                        "end_equity": 100_100.0,
                        "start_equity": 100_000.0,
                        "reward_sum": 0.1,
                        "n_trades": 3,
                        "episode_steps": 99_504,
                        "completed_data_split": True,
                        "episode_end_reason": "data_end",
                        "breach_reason": "",
                    }
                }
            ],
        },
        raising=False,
    )

    assert callback._on_step() is True
    assert (tmp_path / "episode_equity.csv").is_file()
    callback._on_training_end()
    recorded = pd.read_csv(tmp_path / "episode_equity.csv").iloc[0]
    assert recorded["environment_index"] == 0
    assert recorded["episode_steps"] == 99_504
    assert bool(recorded["completed_data_split"])
    assert recorded["episode_end_reason"] == "data_end"


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


def _failing_episode(*, trades: int = 2) -> dict[str, float | int | str]:
    return {
        "end_equity": 90_000.0,
        "start_equity": 100_000.0,
        "slope": -1.0,
        "max_peak_trailing_dd": 0.20,
        "n_trades": trades,
        "breach_reason": "max_loss",
    }


def _rising_episode() -> dict[str, float | int | str]:
    return {
        "end_equity": 101_000.0,
        "start_equity": 100_000.0,
        "slope": 1.0,
        "max_peak_trailing_dd": 0.02,
        "n_trades": 3,
        "breach_reason": "",
    }


@pytest.mark.unit
def test_early_abort_waits_out_the_warmup() -> None:
    episodes = [_failing_episode() for _ in range(4)]
    assert (
        decide_early_abort(
            episodes,
            num_timesteps=4_999_999,
            min_timesteps=5_000_000,
            window=4,
            nonfinite=False,
        )
        is None
    )


@pytest.mark.unit
def test_early_abort_keeps_going_when_equity_does_not_rise() -> None:
    assert (
        decide_early_abort(
            [_failing_episode() for _ in range(4)],
            num_timesteps=5_000_000,
            min_timesteps=5_000_000,
            window=4,
            nonfinite=False,
            stop_on_equity_gate=False,
        )
        is None
    )


@pytest.mark.unit
def test_early_abort_stops_when_the_window_fails_survival_gate() -> None:
    reason = decide_early_abort(
        [_failing_episode() for _ in range(4)],
        num_timesteps=5_000_000,
        min_timesteps=5_000_000,
        window=4,
        nonfinite=False,
    )
    assert reason is not None
    assert reason.startswith("equity_gate:")


@pytest.mark.unit
def test_early_abort_does_not_stop_for_peak_drawdown_alone() -> None:
    episodes = [
        {
            "end_equity": 100_000.0,
            "start_equity": 100_000.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.20,
            "peak_trailing_dd_limit": 0.0,
            "n_trades": 2,
            "breach_reason": "",
        }
        for _ in range(4)
    ]
    assert (
        decide_early_abort(
            episodes,
            num_timesteps=5_000_000,
            min_timesteps=5_000_000,
            window=4,
            nonfinite=False,
        )
        is None
    )


@pytest.mark.unit
def test_early_abort_does_not_stop_on_configured_peak_drawdown_miss() -> None:
    episodes = [
        {
            "end_equity": 100_000.0,
            "start_equity": 100_000.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.20,
            "peak_trailing_dd_limit": 0.10,
            "n_trades": 2,
            "breach_reason": "",
        }
        for _ in range(4)
    ]
    reason = decide_early_abort(
        episodes,
        num_timesteps=5_000_000,
        min_timesteps=5_000_000,
        window=4,
        nonfinite=False,
        stop_on_equity_gate=True,
    )
    assert reason is None


@pytest.mark.unit
def test_early_abort_keeps_going_when_the_window_never_trades() -> None:
    idle = [
        {
            "end_equity": 100_000.0,
            "start_equity": 100_000.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.0,
            "n_trades": 0,
            "breach_reason": "",
        }
        for _ in range(4)
    ]
    reason = decide_early_abort(
        idle,
        num_timesteps=5_000_000,
        min_timesteps=5_000_000,
        window=4,
        nonfinite=False,
    )
    assert reason is None


@pytest.mark.unit
def test_no_trade_early_abort_remains_an_explicit_opt_in() -> None:
    idle = [
        {
            "end_equity": 100_000.0,
            "start_equity": 100_000.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.0,
            "n_trades": 0,
            "breach_reason": "",
        }
        for _ in range(4)
    ]
    reason = decide_early_abort(
        idle,
        num_timesteps=5_000_000,
        min_timesteps=5_000_000,
        window=4,
        nonfinite=False,
        stop_on_no_trades=True,
    )
    assert reason == "no_trades"


@pytest.mark.unit
def test_early_abort_keeps_going_when_one_episode_rises() -> None:
    episodes = [_failing_episode(), _failing_episode(), _rising_episode(), _failing_episode()]
    assert (
        decide_early_abort(
            episodes,
            num_timesteps=5_000_000,
            min_timesteps=5_000_000,
            window=4,
            nonfinite=False,
        )
        is None
    )


@pytest.mark.unit
def test_early_abort_stops_immediately_on_nonfinite() -> None:
    assert (
        decide_early_abort(
            [],
            num_timesteps=1,
            min_timesteps=5_000_000,
            window=4,
            nonfinite=True,
        )
        == "nonfinite"
    )
