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
@pytest.mark.parametrize("action_pattern", ["hold", "mixed"])
def test_bundle_env_step_trace_matches_frames(
    tmp_path: Path, strategy_actions: bool, action_pattern: str
) -> None:
    """The same seeded actions produce exactly the same observations and trace."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=120)
    features = _make_features(bars)
    features["atr_5"] = np.linspace(1.0, 4.0, len(bars))
    features["ema_21"] = features["london_high"]
    # The bundle and frame paths must both exclude absolute event metadata
    # from policy observations while retaining relative state features.
    index_ns = np.asarray(bars.index, dtype="datetime64[ns]").astype(np.int64).astype(np.float64)
    features["po3_ifvg_time_ns"] = index_ns
    features["M5_sweep_high_pivot_time_ns"] = index_ns
    features["po3_ifvg_age"] = np.arange(len(bars), dtype=np.float64)
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
    assert "po3_ifvg_time_ns" not in frames._obs_columns
    assert "M5_sweep_high_pivot_time_ns" not in frames._obs_columns
    assert "po3_ifvg_age" in frames._obs_columns
    assert shared._obs_columns == frames._obs_columns
    obs_frames, _ = frames.reset(seed=41)
    obs_shared, _ = shared.reset(seed=41)
    ended = False
    actions: list[Any]
    if strategy_actions:
        width = frames.action_schema.width
        if action_pattern == "hold":
            actions = [np.zeros(width, dtype=np.float32) for _ in range(200)]
        else:
            base = np.linspace(-1.0, 1.0, width, dtype=np.float32)
            actions = [base if i % 2 else -base for i in range(200)]
    else:
        actions = [0] * 200 if action_pattern == "hold" else [i % 20 for i in range(200)]
    for action in actions:
        for key in obs_frames:
            np.testing.assert_array_equal(obs_frames[key], obs_shared[key])
        left = frames.step(action)
        right = shared.step(action)
        for key in left[0]:
            np.testing.assert_array_equal(left[0][key], right[0][key])
        assert left[1:4] == right[1:4]
        for key, value in left[4].items():
            other = right[4][key]
            if isinstance(value, dict):
                assert isinstance(other, dict)
                for subkey, subvalue in value.items():
                    subother = other[subkey]
                    if isinstance(subvalue, (float, np.floating)) and np.isnan(subvalue):
                        assert isinstance(subother, (float, np.floating)) and np.isnan(subother)
                    else:
                        assert subvalue == subother, (key, subkey)
            else:
                assert value == other, key
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


def test_account_state_v2_bundle_observation_matches_frames(tmp_path: Path) -> None:
    """Versioned account coordinates remain identical on the shared-bundle path."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    common: dict[str, Any] = {
        "obs_window": 10,
        "observation_schema_version": "account-state-v2",
        "strategy": BaselineStrategy(),
    }
    frames = TradingEnv(bars=bars, features=features, **common)
    path = build_env_bundle(
        tmp_path,
        bars,
        features,
        OmegaConf.create(
            {
                "env": {"strategy_actions": False, "obs_window": 10},
                "data": {"raw_dir": str(tmp_path)},
            }
        ),
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="state-v2-parity",
        tickbook=None,
        pre_ny_by_date=None,
    )
    shared = TradingEnv.from_bundle(path, **common)
    frame_obs, _ = frames.reset(seed=22)
    shared_obs, _ = shared.reset(seed=22)
    for _ in range(12):
        np.testing.assert_array_equal(frame_obs["account"], shared_obs["account"])
        frame_obs = frames.step(0)[0]
        shared_obs = shared.step(0)[0]


@pytest.mark.parametrize("entry_state_observation", [False, True])
def test_entry_fsm_transitions_match_frames_and_bundle(
    tmp_path: Path, entry_state_observation: bool
) -> None:
    """FSM transitions and optional policy visibility agree on both data paths."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import PO3IFVGStrategy
    from quant_rl.envs.trading_env import TradingEnv
    from tests.test_envs.test_entry_state_machine import _market_fixture

    bars, features = _market_fixture()
    strategy = PO3IFVGStrategy(enforce_gate=True)
    common: dict[str, Any] = {
        "obs_window": 10,
        "strategy": strategy,
        "strategy_actions": True,
        "entry_state_machine": True,
        "entry_state_observation": entry_state_observation,
        "candidate_max_age_bars": 5,
        "open_manipulation_bars": 0,
        "entry_cooldown_bars": 0,
        "min_sl_atr_mult": 0.0,
        "min_sl_points": 0.0,
        "allow_agent_sl_mode": False,
        "allow_ema_exit": False,
    }
    frames = TradingEnv(bars=bars, features=features, **common)
    cfg = OmegaConf.create(
        {
            "env": {
                "strategy_actions": True,
                "entry_state_machine": True,
                "entry_state_observation": entry_state_observation,
                "obs_window": 10,
            },
            "data": {"raw_dir": str(tmp_path)},
        }
    )
    path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=strategy.raw_columns,
        feature_config_hash=f"entry-fsm-parity-hidden-{not entry_state_observation}",
        tickbook=None,
        pre_ny_by_date=None,
    )
    shared = TradingEnv.from_bundle(path, **common)
    frame_obs, _ = frames.reset(seed=23)
    shared_obs, _ = shared.reset(seed=23)
    action = np.array([0.0, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)

    for expected_state in ("CANDIDATE", "ARMED", "IN_POSITION"):
        for key in frame_obs:
            np.testing.assert_array_equal(frame_obs[key], shared_obs[key])
        frame_step = frames.step(action)
        shared_step = shared.step(action)
        assert frame_step[1:4] == shared_step[1:4]
        assert frame_step[4]["entry_state"] == expected_state
        assert shared_step[4]["entry_state"] == expected_state
        assert frames.entry_state_machine is not None
        assert shared.entry_state_machine is not None
        assert frames.entry_state_machine.state == shared.entry_state_machine.state
        assert frames.entry_state_machine.state == expected_state
        frame_obs, shared_obs = frame_step[0], shared_step[0]
        if entry_state_observation:
            np.testing.assert_array_equal(frame_obs["account"], shared_obs["account"])
            assert frame_obs["account"].shape[0] == 10
        else:
            assert frame_obs["account"].shape[0] == 6
            assert shared_obs["account"].shape[0] == 6


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


def test_bundle_worker_preserves_strict_eligibility_and_simplex_flags(tmp_path: Path) -> None:
    """Worker reconstruction keeps configuration that changes research behavior."""
    from quant_rl.config import load_config
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.env_spec import EnvSpec
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.worker import make_bundle_env
    from quant_rl.train.train_rl import make_env

    bars = _make_deterministic_bars(n=120)
    features = _make_features(bars)
    cfg = load_config()
    cfg.data.raw_dir = str(tmp_path)
    cfg.data.m1_files = {}
    cfg.data.tick_files = {}
    cfg.env.obs_window = 10
    cfg.env.strategy_actions = True
    cfg.env.strict_research_eligibility = True
    cfg.env.allow_agent_tp_mode = True
    cfg.env.allow_multi_tp = True
    cfg.env.allow_simplex = False
    cfg.strategy.name = "baseline"

    frames = make_env(
        bars,
        features,
        cfg,
        algo="ppo",
        reward="dsr",
        episodic=False,
    )
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="worker-env-flags-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    spec = EnvSpec.from_cfg(
        cfg,
        algo="ppo",
        reward="dsr",
        arch="tcn",
        bundle_dir=str(bundle_path),
        episodic=False,
    )
    shared = make_bundle_env(spec)

    assert frames.strict_research_eligibility is True
    assert shared.strict_research_eligibility is True
    assert frames.allow_simplex is False
    assert shared.allow_simplex is False
    assert frames.action_schema.to_dict() == shared.action_schema.to_dict()
    frames_obs, _ = frames.reset(seed=41)
    shared_obs, _ = shared.reset(seed=41)
    for key in frames_obs:
        np.testing.assert_array_equal(frames_obs[key], shared_obs[key])


def test_legacy_sequence_columns_match_between_frame_and_bundle(tmp_path: Path) -> None:
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv

    bars = _make_deterministic_bars(n=120)
    features = _make_features(bars)
    features["po3_ifvg_time_ns"] = np.asarray(bars.index, dtype="datetime64[ns]").astype(np.int64)
    strategy = BaselineStrategy()
    frame_env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=10,
        episodic=False,
        strategy=strategy,
        legacy_sequence_columns=True,
    )
    bundle_dir = tmp_path / "legacy_bundle"
    bundle_cfg = OmegaConf.create(
        {
            "env": {"strategy_actions": False, "obs_window": 10},
            "data": {"raw_dir": str(tmp_path), "m1_files": {}, "tick_files": {}},
        }
    )
    current_bundle_path = build_env_bundle(
        bundle_dir,
        bars,
        features,
        bundle_cfg,
        raw_columns=strategy.raw_columns,
        feature_config_hash="legacy-observation-columns-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    bundle_path = build_env_bundle(
        bundle_dir,
        bars,
        features,
        bundle_cfg,
        raw_columns=strategy.raw_columns,
        feature_config_hash="legacy-observation-columns-v1",
        tickbook=None,
        pre_ny_by_date=None,
        legacy_sequence_columns=True,
    )
    bundle_env = TradingEnv.from_bundle(
        bundle_path,
        obs_window=10,
        episodic=False,
        strategy=BaselineStrategy(),
    )

    assert "po3_ifvg_time_ns" in frame_env._obs_columns
    assert current_bundle_path != bundle_path
    assert bundle_env._obs_columns == frame_env._obs_columns
    assert bundle_env._legacy_sequence_columns is True
    frame_obs, _ = frame_env.reset(seed=11)
    bundle_obs, _ = bundle_env.reset(seed=11)
    np.testing.assert_array_equal(frame_obs["seq"], bundle_obs["seq"])
