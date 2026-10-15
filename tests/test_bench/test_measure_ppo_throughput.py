from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scripts.bench.measure_ppo_throughput import (
    _apply_profile_overrides,
    _json_default,
    _prepare_bundle,
    _process_tree_cpu_seconds,
)


def test_profile_overrides_are_explicit_and_preserve_production_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 64},
            "ppo": {"n_steps": 8192, "batch_size": 4096},
        }
    )

    _apply_profile_overrides(cfg, n_envs=None, rollout_steps_per_env=None)

    assert cfg.env.n_envs == 64
    assert cfg.ppo.n_steps == 8192
    assert cfg.ppo.batch_size == 4096


def test_rollout_override_is_global_steps_times_logical_envs() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 32},
            "ppo": {"n_steps": 8192},
        }
    )

    _apply_profile_overrides(cfg, n_envs=None, rollout_steps_per_env=128)

    assert cfg.ppo.n_steps == 4096


def test_numpy_scalars_are_json_serializable() -> None:
    assert _json_default(np.int64(7)) == 7
    assert _json_default(np.float32(1.5)) == 1.5


def test_process_tree_cpu_measurement_is_nonnegative() -> None:
    assert _process_tree_cpu_seconds() >= 0.0


def test_variant_config_uses_production_strategy_merge_and_cli_override(
    tmp_path: Path,
) -> None:
    cfg, bundle_path, algo, arch = _prepare_bundle(
        tmp_path,
        variant="ladder_a0_flat",
        overrides=[
            "ppo.n_epochs=7",
            "env.vector_backend=sharded_bundle",
            "env.vector_shards=4",
        ],
    )

    assert bundle_path == tmp_path
    assert algo == "ppo"
    assert arch == "tcn"
    assert cfg.strategy.name == "baseline"
    assert cfg.ppo.n_epochs == 7
    assert cfg.env.vector_backend == "sharded_bundle"
    assert cfg.env.vector_shards == 4
    assert cfg.ppo.total_timesteps > 0
