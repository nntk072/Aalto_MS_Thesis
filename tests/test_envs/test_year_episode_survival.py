"""Year-episode, session-scoped seq, and FTMO terminal-flag tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_rl.backtest.account import AccountState
from quant_rl.data.session import add_session_id, add_session_labels
from quant_rl.envs.trading_env import TradingEnv
from quant_rl.train.train_rl import _max_episode_steps


def _ohlc(idx: pd.DatetimeIndex, close: np.ndarray[Any, Any]) -> pd.DataFrame:
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


def _multi_day_env(
    *,
    n_days: int = 3,
    bars_per_day: int = 120,
    obs_window: int = 60,
    episodic: bool = True,
    max_episode_steps: int | None = None,
    guardrail_kwargs: dict[str, float] | None = None,
) -> TradingEnv:
    chunks: list[pd.DatetimeIndex] = []
    for d in range(n_days):
        day = 6 + d
        chunks.append(
            pd.date_range(
                f"2025-01-{day:02d} 15:00",
                periods=bars_per_day,
                freq="1min",
                tz="Etc/GMT-3",
            )
        )
    idx = chunks[0]
    for c in chunks[1:]:
        idx = pd.DatetimeIndex(idx.append(c))
    idx = pd.DatetimeIndex(idx)
    close = 20000.0 + np.arange(len(idx), dtype=float) * 0.05
    bars = _ohlc(idx, close)
    # Tag each calendar day's bars with a distinct marker feature so we can
    # detect yesterday leaking into today's seq window.
    day_ids = np.zeros(len(idx), dtype=np.float32)
    for d in range(n_days):
        sid = int(bars["session_id"].iloc[d * bars_per_day])
        day_ids[bars["session_id"].to_numpy() == sid] = float(d + 1)
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
            "day_marker": day_ids,
        },
        index=idx,
    )
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=obs_window,
        episodic=episodic,
        max_episode_steps=max_episode_steps,
        guardrail_kwargs=guardrail_kwargs,
        eod_risk={"max_age_hours": 1000.0, "stale_bars": 10_000, "max_loss_usd": 1e12},
    )


@pytest.mark.unit
def test_max_episode_steps_null_reaches_last_session() -> None:
    env = _multi_day_env(n_days=2, bars_per_day=90, obs_window=10, max_episode_steps=None)
    env.reset()
    done = truncated = False
    steps = 0
    last_sid = None
    while not (done or truncated) and steps < 50_000:
        last_sid = int(env.bars["session_id"].iloc[min(env.step_idx, len(env.bars) - 1)])
        _, _, done, truncated, _ = env.step(0)
        steps += 1
    assert truncated is True
    assert done is False
    assert last_sid == int(env.bars["session_id"].iloc[-1])
    assert steps > 50  # walked well past a single short cap


@pytest.mark.unit
def test_session_scoped_seq_pads_at_ny_open() -> None:
    env = _multi_day_env(n_days=2, bars_per_day=100, obs_window=60)
    env.reset()
    # Force pointer to the first NY bar of session 2 so the window must pad.
    sids = env.bars["session_id"].to_numpy()
    uniq = pd.unique(sids)
    assert len(uniq) >= 2
    day2_start = int(np.flatnonzero(sids == uniq[1])[0])
    env.step_idx = day2_start
    env._ny_pos = int(np.searchsorted(env._ny_indices, day2_start, side="left"))
    obs = env._get_observation()
    assert obs["seq"].shape[0] == 60
    marker_col = list(env._obs_features.columns).index("day_marker")
    markers = obs["seq"][:, marker_col]
    # At exact session open, seq is empty → fully padded zeros.
    assert np.all(markers == 0.0)


@pytest.mark.unit
def test_session_scoped_seq_excludes_yesterday() -> None:
    env = _multi_day_env(n_days=2, bars_per_day=100, obs_window=60)
    env.reset()
    # Advance into day-2 NY far enough that an absolute 60-bar window
    # would otherwise include day-1 bars.
    for _ in range(80):
        env.step(0)
    obs = env._get_observation()
    marker_col = list(env._obs_features.columns).index("day_marker")
    markers = obs["seq"][:, marker_col]
    nonzero = markers[markers != 0.0]
    assert nonzero.size > 0
    assert set(np.unique(nonzero).tolist()) == {2.0}


@pytest.mark.unit
def test_hard_max_loss_terminates_not_truncates() -> None:
    env = _multi_day_env(
        n_days=2,
        bars_per_day=80,
        obs_window=10,
        episodic=True,
        max_episode_steps=None,
        guardrail_kwargs={
            "daily_loss_limit": 5_000.0,
            "max_loss_limit": 10_000.0,
            "soft_daily_loss_limit": 2_000.0,
            "soft_max_loss_limit": 5_000.0,
            "trailing_dd_limit": 0.0,
        },
    )
    env.reset()
    # Simulate absolute max-loss breach without opening trades.
    env.account = AccountState(initial_balance=100_000.0)
    env.account.equity = 89_000.0
    env.account.balance = 89_000.0
    env.account._update_loss_from_initial()
    assert env.account.loss_from_initial() >= 10_000.0

    _, reward, done, truncated, _ = env.step(0)
    assert done is True
    assert truncated is False
    assert reward == pytest.approx(-10.0)
    assert env.breach_events and env.breach_events[0]["reason"] == "max_drawdown"


@pytest.mark.unit
def test_hard_daily_terminates_year_episode() -> None:
    env = _multi_day_env(
        n_days=2,
        bars_per_day=80,
        obs_window=10,
        episodic=True,
        max_episode_steps=None,
        guardrail_kwargs={
            "daily_loss_limit": 5_000.0,
            "max_loss_limit": 10_000.0,
            "soft_daily_loss_limit": 2_000.0,
        },
    )
    env.reset()
    env.account.daily_loss = 5_500.0
    _, reward, done, truncated, _ = env.step(0)
    assert done is True
    assert truncated is False
    assert reward == pytest.approx(-10.0)
    assert env.breach_events[0]["reason"] == "daily_loss"


@pytest.mark.unit
def test_trailing_dd_terminates_year_episode() -> None:
    env = _multi_day_env(
        n_days=2,
        bars_per_day=80,
        obs_window=10,
        episodic=True,
        max_episode_steps=None,
        guardrail_kwargs={
            "daily_loss_limit": 5_000.0,
            "max_loss_limit": 10_000.0,
            "trailing_dd_limit": 0.07,
            "soft_trailing_dd_limit": 0.04,
        },
    )
    obs, _ = env.reset()
    assert obs["account"].shape == (6,)
    env.account.update_equity(10_000.0)
    env.account.balance = 102_300.0
    env.account.update_equity(0.0)
    _, reward, done, truncated, _ = env.step(0)
    assert done is True
    assert truncated is False
    assert reward == pytest.approx(-10.0)
    assert env.breach_events[0]["reason"] == "trailing_dd"


@pytest.mark.unit
def test_max_episode_steps_parser_none() -> None:
    class _E:
        def get(self, key: str, default: Any = None) -> Any:
            return None

    class _C:
        env = _E()

    assert _max_episode_steps(_C()) is None


@pytest.mark.unit
def test_eval_year_fail_latches_once() -> None:
    """FTMO max-loss in eval blocks the rest of the year without re-logging."""
    env = _multi_day_env(
        n_days=3,
        bars_per_day=80,
        obs_window=10,
        episodic=False,
        max_episode_steps=None,
        guardrail_kwargs={
            "daily_loss_limit": 5_000.0,
            "max_loss_limit": 10_000.0,
            "trailing_dd_limit": 0.0,
        },
    )
    env.reset()
    env.account.balance = 89_000.0
    env.account.update_equity(0.0)
    _, _, done, truncated, _ = env.step(0)
    assert done is False
    assert truncated is False
    assert env.breach_events[0]["reason"] == "max_drawdown"
    n_first = len(env.breach_events)
    for _ in range(120):
        _, _, done, truncated, _ = env.step(0)
        if done or truncated:
            break
    assert len(env.breach_events) == n_first
    assert env._year_failed is True


@pytest.mark.unit
def test_shared_obs_memmap_matches_private_copy_across_session(tmp_path: Any) -> None:
    """A read-only memmap must match a private matrix, including a new session."""
    private = _multi_day_env(n_days=2, bars_per_day=100, obs_window=60)
    array = np.ascontiguousarray(private._obs_features.to_numpy(dtype=np.float32))
    path = tmp_path / "obs_features.npy"
    np.save(path, array)
    shared = TradingEnv(
        bars=private.bars,
        features=private.features,
        obs_window=60,
        eod_risk={"max_age_hours": 1000.0, "stale_bars": 10_000, "max_loss_usd": 1e12},
        obs_features_mmap=str(path),
    )
    assert shared._obs_features_arr.flags.writeable is False
    private.reset()
    shared.reset()
    for _ in range(150):
        obs_private, _, done_p, trunc_p, _ = private.step(0)
        obs_shared, _, done_s, trunc_s, _ = shared.step(0)
        np.testing.assert_array_equal(obs_private["seq"], obs_shared["seq"])
        np.testing.assert_array_equal(obs_private["seq_mask"], obs_shared["seq_mask"])
        assert done_p == done_s
        assert trunc_p == trunc_s
        if done_p or trunc_p:
            break
    assert private.step_idx > 100
