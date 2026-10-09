"""Tests for atomic, immutable shared environment data bundles."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_envs.test_trading_env_golden_traces import (
    _make_deterministic_bars,
    _make_features,
)


def _fixture_bundle_inputs() -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Return deterministic source arrays and an observation-relevant config."""
    bars = _make_deterministic_bars(n=32)
    features = _make_features(bars)
    arrays = {
        "bar_ohlc": bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64),
        "bar_time_ns": bars.index.to_numpy(dtype="datetime64[ns]").view(np.int64).copy(),
        "features_num": features.to_numpy(dtype=np.float64),
    }
    config = {
        "env": {"obs_window": 10},
        "strategy": {"raw_columns": ["last_swing_high", "last_swing_low"]},
        "features": {"hash": "fixture-v1"},
        "data": {"primary": "US100.cash", "split": {"train_end": "2025-01-31"}},
        "logging": {"level": "INFO"},
    }
    return arrays, config


def _concurrent_build(cache_dir: str, result_queue: Any) -> None:
    """Build one identical bundle in a child process."""
    from quant_rl.envs.shared_bundle import build_bundle

    arrays, config = _fixture_bundle_inputs()
    result_queue.put(
        str(
            build_bundle(
                cache_dir,
                arrays=arrays,
                config=config,
                source_fingerprints={"bars": {"size": 32}},
            )
        )
    )


def test_bundle_round_trip_matches_source(tmp_path: Path) -> None:
    """Every saved array reloads byte-for-byte equal to its input."""
    from quant_rl.envs.shared_bundle import build_bundle, load_bundle

    arrays, config = _fixture_bundle_inputs()
    bundle_path = build_bundle(
        tmp_path,
        arrays=arrays,
        config=config,
        source_fingerprints={"bars": {"size": 32}},
        column_names={"features_num": ["london_high", "london_low"]},
    )
    loaded = load_bundle(bundle_path)

    assert set(loaded.arrays) == set(arrays)
    for name, source in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], source)
    assert loaded.manifest["column_names"]["features_num"] == [
        "london_high",
        "london_low",
    ]


def test_build_is_idempotent(tmp_path: Path) -> None:
    """A second build reuses the completed directory without rewriting files."""
    from quant_rl.envs.shared_bundle import build_bundle

    arrays, config = _fixture_bundle_inputs()
    first = build_bundle(
        tmp_path,
        arrays=arrays,
        config=config,
        source_fingerprints={"bars": {"size": 32}},
    )
    mtimes = {path.name: path.stat().st_mtime_ns for path in first.iterdir()}
    second = build_bundle(
        tmp_path,
        arrays=arrays,
        config=config,
        source_fingerprints={"bars": {"size": 32}},
    )

    assert second == first
    assert {path.name: path.stat().st_mtime_ns for path in second.iterdir()} == mtimes


def test_build_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed array write leaves neither a final bundle nor temporary files."""
    import quant_rl.envs.shared_bundle as shared_bundle

    arrays, config = _fixture_bundle_inputs()
    key = shared_bundle.bundle_key(config, {"bars": {"size": 32}})
    original_writer = shared_bundle._write_array
    writes = 0

    def fail_writer(path: Path, array: np.ndarray[Any, Any]) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated write failure")
        original_writer(path, array)

    monkeypatch.setattr(shared_bundle, "_write_array", fail_writer)
    with pytest.raises(OSError, match="simulated write failure"):
        shared_bundle.build_bundle(
            tmp_path,
            arrays=arrays,
            config=config,
            source_fingerprints={"bars": {"size": 32}},
        )

    bundle_root = tmp_path / "shared_env"
    assert not (bundle_root / key).exists()
    assert list(bundle_root.glob(f"{key}.tmp-*")) == []


def test_concurrent_builds_produce_one_bundle(tmp_path: Path) -> None:
    """Concurrent builders serialize on the key lock and return one bundle."""
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_concurrent_build, args=(str(tmp_path), results)) for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    paths = [Path(results.get(timeout=10)) for _ in workers]

    assert paths[0] == paths[1]
    assert len(list((tmp_path / "shared_env").glob("*/manifest.json"))) == 1


def test_key_uses_observation_relevant_config_only() -> None:
    """Observation settings affect the key while logging settings do not."""
    from quant_rl.envs.shared_bundle import bundle_key

    _, config = _fixture_bundle_inputs()
    source_fingerprints = {"bars": {"size": 32}}
    changed_logging = {**config, "logging": {"level": "DEBUG"}}
    changed_window = {**config, "env": {"obs_window": 20}}

    assert bundle_key(config, source_fingerprints) == bundle_key(
        changed_logging, source_fingerprints
    )
    assert bundle_key(config, source_fingerprints) != bundle_key(
        changed_window, source_fingerprints
    )


def test_bundle_arrays_are_read_only(tmp_path: Path) -> None:
    """Every mmap loaded from the bundle is read-only."""
    from quant_rl.envs.shared_bundle import build_bundle, load_bundle

    arrays, config = _fixture_bundle_inputs()
    bundle_path = build_bundle(
        tmp_path,
        arrays=arrays,
        config=config,
        source_fingerprints={"bars": {"size": 32}},
    )
    loaded = load_bundle(bundle_path)

    assert all(array.flags.writeable is False for array in loaded.arrays.values())
