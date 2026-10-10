"""Exact behavior checks for the memory-mapped environment path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.test_envs.test_trading_env_golden_traces import (
    _make_deterministic_bars,
    _make_features,
)


@pytest.mark.parametrize("strategy_actions", [False, True])
def test_bundle_env_step_trace_matches_frames(tmp_path: Path, strategy_actions: bool) -> None:
    """The same seeded actions produce exactly the same observations and trace."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=120)
    features = _make_features(bars)
    features["atr_5"] = np.linspace(1.0, 4.0, len(bars))
    features["ema_21"] = features["london_high"]
    common: dict[str, Any] = {
        "obs_window": 10,
        "strategy_actions": strategy_actions,
        "strategy": BaselineStrategy(),
    }
    frames = TradingEnv(bars=bars, features=features, **common)
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        OmegaConf.create(
            {
                "env": {"strategy_actions": strategy_actions, "obs_window": 10},
                "data": {"raw_dir": str(tmp_path), "m1_files": {}, "tick_files": {}},
            }
        ),
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="equivalence-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    shared = TradingEnv.from_bundle(bundle_path, **common)
    obs_frames, _ = frames.reset(seed=41)
    obs_shared, _ = shared.reset(seed=41)
    ended = False
    for action in [0] * 200:
        for key in obs_frames:
            np.testing.assert_array_equal(obs_frames[key], obs_shared[key])
        left = frames.step(action)
        right = shared.step(action)
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        assert left[1:4] == right[1:4]
        assert left[4] == right[4]
        obs_frames, obs_shared = left[0], right[0]
        if left[2] or left[3]:
            ended = True
            break

    assert ended
    assert shared.bars is None
    assert shared.features is None


def test_bundle_env_without_session_id_matches_frames(tmp_path: Path) -> None:
    """Missing source session IDs retain the legacy absolute-window behavior."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=96).drop(columns="session_id")
    features = _make_features(bars)
    cfg = OmegaConf.create(
        {"env": {"strategy_actions": False, "obs_window": 10}, "data": {"raw_dir": str(tmp_path)}}
    )
    frames = TradingEnv(bars=bars, features=features, obs_window=10)
    path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="missing-session-id-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    shared = TradingEnv.from_bundle(path, obs_window=10)
    left, _ = frames.reset(seed=19)
    right, _ = shared.reset(seed=19)
    for _ in range(20):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        left_step = frames.step(0)
        right_step = shared.step(0)
        np.testing.assert_array_equal(left_step[0]["seq"], right_step[0]["seq"])
        assert left_step[1:4] == right_step[1:4]
        assert left_step[4] == right_step[4]
        left, right = left_step[0], right_step[0]


def test_bundle_env_uses_read_only_ndarray_views(tmp_path: Path) -> None:
    """The hot path uses plain ndarray views backed by shared read-only mmaps."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.shared_bundle import load_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=64)
    features = _make_features(bars)
    cfg = OmegaConf.create(
        {"env": {"strategy_actions": False, "obs_window": 10}, "data": {"raw_dir": str(tmp_path)}}
    )
    path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="ndarray-view-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    bundle = load_bundle(path)
    env = TradingEnv(bars=None, features=None, bundle_data=bundle, obs_window=10)
    assert isinstance(env._features_arr, np.ndarray)
    assert not isinstance(env._features_arr, np.memmap)
    assert not env._features_arr.flags.writeable
    assert np.shares_memory(env._features_arr, bundle["features_num"])


def _assert_same_equity_times(left: list[pd.Timestamp], right: list[pd.Timestamp]) -> None:
    assert len(left) == len(right), f"equity_times length differs: {len(left)} vs {len(right)}"
    for i, (a, b) in enumerate(zip(left, right)):
        assert a.tzinfo is not None, f"equity_times[{i}] is naive (frames path): {a}"
        assert a.utcoffset() == pd.Timedelta(hours=3), (
            f"equity_times[{i}] utcoffset={a.utcoffset()}"
        )
        assert b.tzinfo is not None, f"equity_times[{i}] is naive (bundle path): {b}"
        assert b.utcoffset() == pd.Timedelta(hours=3), (
            f"equity_times[{i}] utcoffset={b.utcoffset()}"
        )
        assert a.tz_convert("UTC").value == b.tz_convert("UTC").value, (
            f"equity_times[{i}] differs: {a} vs {b}"
        )


def _assert_same_trade_log(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    assert len(left) == len(right), f"trade_log length differs: {len(left)} vs {len(right)}"
    for i, (la, rb) in enumerate(zip(left, right)):
        for key, value in la.items():
            if key == "time":
                continue
            other = rb.get(key)
            both_nan = (
                isinstance(value, float)
                and isinstance(other, float)
                and np.isnan(value)
                and np.isnan(other)
            )
            assert both_nan or value == other, f"trade_log[{i}].{key} differs: {value} vs {other}"
        la_time = la.get("time")
        rb_time = rb.get("time")
        assert la_time is not None and rb_time is not None
        assert la_time.tzinfo is not None, f"trade_log[{i}].time is naive in frames"
        assert la_time.utcoffset() == pd.Timedelta(hours=3), (
            f"trade_log[{i}].time utcoffset={la_time.utcoffset()}"
        )
        assert rb_time is not None and rb_time.tzinfo is not None, (
            f"trade_log[{i}].time is naive in bundle"
        )
        assert rb_time.utcoffset() == pd.Timedelta(hours=3), (
            f"trade_log[{i}].time utcoffset={rb_time.utcoffset()}"
        )
        assert la_time.tz_convert("UTC").value == rb_time.tz_convert("UTC").value, (
            f"trade_log[{i}].time differs: {la_time} vs {rb_time}"
        )


def test_equity_times_and_trade_log_match_frames_and_bundle(tmp_path: Path) -> None:
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=120)
    features = _make_features(bars)
    common: dict[str, Any] = {
        "obs_window": 10,
        "strategy": BaselineStrategy(),
        "strategy_actions": False,
    }
    frames = TradingEnv(bars=bars, features=features, **common)
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        OmegaConf.create(
            {
                "env": {"strategy_actions": False, "obs_window": 10},
                "data": {"raw_dir": str(tmp_path), "m1_files": {}, "tick_files": {}},
            }
        ),
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="equity-time-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    shared = TradingEnv.from_bundle(bundle_path, **common)

    def run_episode(env: TradingEnv) -> dict[str, Any]:
        env.reset(seed=42)
        for action in range(300):
            _, _, done, trunc, _ = env.step(action)
            if done or trunc:
                break
        return {
            "equity_times": list(env.equity_times),
            "trade_log": [dict(record) for record in env.trade_log],
        }

    left = run_episode(frames)
    right = run_episode(shared)

    _assert_same_equity_times(left["equity_times"], right["equity_times"])
    _assert_same_trade_log(left["trade_log"], right["trade_log"])
