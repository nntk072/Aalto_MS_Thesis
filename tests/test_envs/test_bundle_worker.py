"""Bundle worker construction and payload tests."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_envs.test_trading_env_golden_traces import _make_deterministic_bars, _make_features


def test_worker_factory_matches_default_make_env(tmp_path: Path) -> None:
    """The compact spec factory retains the current training env defaults."""
    from quant_rl.config import load_config
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.env_spec import EnvSpec
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.worker import make_bundle_env
    from quant_rl.train.train_rl import make_env

    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    cfg = load_config()
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=list(BaselineStrategy().raw_columns),
        feature_config_hash="worker-default-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    spec = EnvSpec.from_cfg(
        cfg,
        algo="ppo",
        reward="dsr",
        arch="tcn",
        bundle_dir=str(bundle_path),
    )
    left = make_env(bars, features, cfg, algo="ppo", reward="dsr", arch="tcn")
    right = make_bundle_env(spec)
    obs_left, _ = left.reset(seed=15)
    obs_right, _ = right.reset(seed=15)
    for key in obs_left:
        np.testing.assert_array_equal(obs_left[key], obs_right[key])
    for _ in range(60):
        left_step = left.step(0)
        right_step = right.step(0)
        for key in left_step[0]:
            np.testing.assert_array_equal(left_step[0][key], right_step[0][key])
        assert left_step[1:4] == right_step[1:4]
        assert left_step[4] == right_step[4]


def test_worker_import_does_not_load_torch_or_sb3() -> None:
    """Bundle worker imports remain independent of the training stack."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import quant_rl.envs.worker, sys; "
            "assert 'torch' not in sys.modules; "
            "assert 'stable_baselines3' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_worker_factory_matches_strategy_actions(tmp_path: Path) -> None:
    """The strategy-action risk ranges and strategy flags survive serialization."""
    from quant_rl.config import load_config
    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.env_spec import EnvSpec
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.worker import make_bundle_env
    from quant_rl.train.train_rl import make_env

    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    features["atr_5"] = np.linspace(1.0, 3.0, len(bars))
    cfg = load_config()
    cfg.env.strategy_actions = True
    cfg.strategy.name = "baseline"
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=list(BaselineStrategy().raw_columns),
        feature_config_hash="worker-strategy-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    spec = EnvSpec.from_cfg(
        cfg,
        algo="ppo",
        reward="dsr",
        arch="tcn",
        bundle_dir=str(bundle_path),
    )
    frames = make_env(bars, features, cfg, algo="ppo", reward="dsr", arch="tcn")
    shared = make_bundle_env(spec)
    assert frames.action_space == shared.action_space
    assert frames.risk_frac_range == shared.risk_frac_range
    assert frames.rr_ratio_range == shared.rr_ratio_range
    assert frames._max_tp_dist is not None
    assert shared._max_tp_dist is not None
    np.testing.assert_array_equal(frames._max_tp_dist, shared._max_tp_dist)


@pytest.mark.parametrize("name", ["po3_ifvg", "distribution", "unrecognized"])
def test_worker_recreates_named_strategy(name: str) -> None:
    """Each strategy name reconstructs its existing class and reward type."""
    from quant_rl.envs.distribution_reward import DistributionReward
    from quant_rl.envs.po3_reward import PO3Reward
    from quant_rl.envs.strategies import BaselineStrategy, DistributionStrategy, PO3IFVGStrategy
    from quant_rl.envs.worker import _strategy

    strategy, reward, weight = _strategy(
        {
            "strategy": {
                "name": name,
                "manipulation_filter": "follow",
                "ifvg": {"require_price_retest": False},
                "reward": {"strategy_weight": 0.25},
            }
        },
        True,
    )
    assert weight == (0.0 if name == "unrecognized" else 0.25)
    if name == "po3_ifvg":
        assert isinstance(strategy, PO3IFVGStrategy)
        assert isinstance(reward, PO3Reward)
        assert strategy.require_price_retest is False
    elif name == "distribution":
        assert isinstance(strategy, DistributionStrategy)
        assert isinstance(reward, DistributionReward)
    else:
        assert isinstance(strategy, BaselineStrategy)
        assert reward is None


def test_env_spec_pickle_is_small(tmp_path: Path) -> None:
    """The worker payload contains only a bundle path and scalar config."""
    from quant_rl.envs.env_spec import EnvSpec

    spec = EnvSpec.from_cfg(
        {"env": {"obs_window": 60}, "account": {"initial_balance": 100_000}},
        algo="ppo",
        reward="dsr",
        arch="tcn",
        bundle_dir=str(tmp_path),
    )
    assert len(pickle.dumps(spec)) < 1_000_000
