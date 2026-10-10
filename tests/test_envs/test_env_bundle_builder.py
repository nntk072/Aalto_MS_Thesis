"""Bundle key and source-array construction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tests.test_envs.test_trading_env_golden_traces import _make_deterministic_bars, _make_features


def test_bundle_key_separates_different_input_slices(tmp_path: Path) -> None:
    """A shorter evaluation prefix cannot reuse the full training bundle."""
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy

    bars = _make_deterministic_bars(n=64)
    features = _make_features(bars)
    config = OmegaConf.create(
        {"env": {"strategy_actions": False, "obs_window": 10}, "data": {"raw_dir": str(tmp_path)}}
    )
    full = build_env_bundle(
        tmp_path,
        bars,
        features,
        config,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="slice-key-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    prefix = build_env_bundle(
        tmp_path,
        bars.iloc[:32],
        features.iloc[:32],
        config,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="slice-key-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    assert full != prefix


def test_bundled_observation_matrix_matches_frames_path(tmp_path: Path) -> None:
    """The bundle observation matrix is byte-identical to the frame conversion."""
    from quant_rl.config import load_config
    from quant_rl.data.ticks import TickBook
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.shared_bundle import load_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.features.build import attach_reachable_r, select_obs_columns

    bars = _make_deterministic_bars(n=64)
    bars["session"] = np.where(np.arange(len(bars)) % 2, "ny", "off")
    features = _make_features(bars)
    features["atr_5"] = ["invalid" if index == 0 else str(index + 1) for index in range(len(bars))]
    features["ema_21"] = np.linspace(19900.0, 20100.0, len(bars))
    tick_ts = bars.index.to_numpy(dtype="datetime64[ns]").view(np.int64).copy()
    tick_bid = np.linspace(20000.0, 20001.0, len(bars), dtype=np.float32)
    tick_ask = tick_bid + np.float32(0.5)
    tickbook = TickBook(tick_ts, tick_bid, tick_ask)
    pre_ny = {bars.index[0].date(): np.arange(6, dtype=np.float32).reshape(3, 2)}
    cfg = load_config()
    cfg.env.strategy_actions = False
    raw_columns: list[str] = list(BaselineStrategy().raw_columns)
    expected_frame = attach_reachable_r(select_obs_columns(features, raw_columns), bars, features)
    expected = np.ascontiguousarray(expected_frame.to_numpy(dtype=np.float32))
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=raw_columns,
        feature_config_hash="fixture-v1",
        tickbook=tickbook,
        pre_ny_by_date=pre_ny,
    )
    bundle = load_bundle(bundle_path)
    actual = bundle["obs_features"]

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(
        bundle["bar_ohlc"], bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    )
    np.testing.assert_array_equal(bundle["bar_spread"], bars["spread"].to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(
        bundle["bar_time_ns"], bars.index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    )
    np.testing.assert_array_equal(bundle["bar_session_id"], bars["session_id"].to_numpy())
    np.testing.assert_array_equal(bundle["bar_session"], (np.arange(len(bars)) + 1) % 2)
    np.testing.assert_array_equal(
        bundle["bar_day_ord"],
        [day.toordinal() for day in pd.DatetimeIndex(bars.index).date],
    )
    np.testing.assert_array_equal(
        bundle["atr_5"], pd.to_numeric(features["atr_5"], errors="coerce")
    )
    np.testing.assert_array_equal(bundle["ema_21"], features["ema_21"])
    np.testing.assert_array_equal(bundle["tick_ts"], tick_ts)
    np.testing.assert_array_equal(bundle["tick_bid"], tick_bid)
    np.testing.assert_array_equal(bundle["tick_ask"], tick_ask)
    np.testing.assert_array_equal(bundle["pre_ny_seq"][0], next(iter(pre_ny.values())))
    assert bundle.manifest["labels"]["session"] == ["ny", "off"]


def test_bundle_stores_volume_and_export_includes_it(tmp_path: Path) -> None:
    """Volume and tickvol survive the bundle round trip for chart exports."""
    from quant_rl.envs.env_bundle_builder import bars_for_export, build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy

    bars = _make_deterministic_bars(n=64)
    features = _make_features(bars)
    config = OmegaConf.create(
        {"env": {"strategy_actions": False, "obs_window": 10}, "data": {"raw_dir": str(tmp_path)}}
    )
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        config,
        raw_columns=BaselineStrategy().raw_columns,
        feature_config_hash="volume-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    exported = bars_for_export(bundle_path)
    assert "volume" in exported.columns
    assert "tickvol" in exported.columns
    np.testing.assert_array_equal(
        exported["volume"].to_numpy(), bars["volume"].to_numpy(dtype=np.float64)
    )
    np.testing.assert_array_equal(
        exported["tickvol"].to_numpy(), bars["tickvol"].to_numpy(dtype=np.float64)
    )
