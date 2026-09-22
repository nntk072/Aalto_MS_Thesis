"""Stage A observation columns and train/live observation parity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_rl.envs.trading_env import TradingEnv
from quant_rl.features.build import ensure_london_atr_distances, select_obs_columns
from quant_rl.features.indicators import ny_session_clock
from quant_rl.live.rl_strategy import RLStrategyAdapter


class _StubModel:
    def predict(self, obs: dict[str, Any], deterministic: bool = True) -> np.ndarray[Any, Any]:  # noqa: ARG002
        return np.array([1])


def test_london_distances_use_explicit_atr_signs() -> None:
    feat = pd.DataFrame(
        {
            "london_high": [110.0, 110.0],
            "london_low": [90.0, 90.0],
            "atr_5": [2.0, 2.0],
        }
    )
    close = pd.Series([100.0, 104.0])
    out = ensure_london_atr_distances(feat, close)
    assert out["price_to_london_high_atr"].iloc[0] == (110.0 - 100.0) / 2.0
    assert out["price_to_london_low_atr"].iloc[0] == (100.0 - 90.0) / 2.0
    assert out["price_to_london_high_atr"].iloc[1] == (110.0 - 104.0) / 2.0
    assert out["price_to_london_low_atr"].iloc[1] == (104.0 - 90.0) / 2.0


def test_london_distances_skipped_when_pd_context_present() -> None:
    feat = pd.DataFrame(
        {
            "london_high": [110.0],
            "london_low": [90.0],
            "atr_5": [2.0],
            "ctx_london_high_dist_atr": [1.5],
            "ctx_london_low_dist_atr": [0.5],
        }
    )
    out = ensure_london_atr_distances(feat, pd.Series([100.0]))
    assert "price_to_london_high_atr" not in out.columns
    assert "price_to_london_low_atr" not in out.columns


def test_session_clock_clips_outside_the_session() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-06 10:00",
            "2025-01-06 16:30",
            "2025-01-06 23:00",
            "2025-01-06 23:30",
        ]
    )
    clock = ny_session_clock(index, "16:30", "23:00")
    assert clock["ny_session_sin"].between(-1.0, 1.0).all()
    assert clock["ny_session_cos"].between(-1.0, 1.0).all()
    assert clock["ny_minutes_to_close"].between(0.0, 1.0).all()
    assert clock["ny_minutes_to_close"].iloc[0] == 1.0
    assert clock["ny_minutes_to_close"].iloc[1] == 1.0
    assert clock["ny_minutes_to_close"].iloc[2] == 0.0
    assert clock["ny_minutes_to_close"].iloc[3] == 0.0
    assert clock["ny_session_sin"].iloc[1] == 0.0


def test_train_live_observation_parity() -> None:
    n = 40
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = 20000.0 + np.linspace(0.0, 10.0, n)
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
            "session_id": 0,
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "volume_spike": 1.0,
            "M5_last_swing_high": close + 5,
            "signal": np.linspace(-1.0, 1.0, n),
        },
        index=idx,
    )
    env = TradingEnv(bars=bars, features=features, obs_window=10, initial_balance=100_000.0)
    env.account.update_equity(-250.0)
    env_obs = env._get_observation()

    adapter = RLStrategyAdapter(model=_StubModel(), config_path="quant_rl/config/default.yaml")
    adapter._features = features
    adapter.obs_window = 10
    adapter._initial_balance = 100_000.0
    step = int(env.step_idx)
    live_obs = adapter.build_observation(
        {
            "equity": float(env.account.equity),
            "open_pnl": 0.0,
            "position_direction": 0.0,
            "dist_to_sl": 0.0,
            "trailing_dd": float(env.account.trailing_drawdown_pct()),
            "close": float(bars["close"].iloc[step]),
        }
    )

    assert set(env_obs) >= {"seq", "seq_mask", "account"}
    assert set(live_obs) == {"seq", "seq_mask", "account"}
    assert env_obs["seq"].shape == live_obs["seq"].shape
    assert env_obs["seq_mask"].shape == live_obs["seq_mask"].shape
    assert env_obs["account"].shape == live_obs["account"].shape == (6,)
    assert list(env._obs_features.columns) == list(select_obs_columns(features).columns)
    assert "M5_last_swing_high" not in env._obs_features.columns
    np.testing.assert_allclose(env_obs["account"], live_obs["account"])
