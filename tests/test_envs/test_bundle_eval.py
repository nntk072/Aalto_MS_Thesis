"""Evaluation using a prebuilt bundle environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from tests.test_envs.test_trading_env_golden_traces import _make_deterministic_bars, _make_features


def test_evaluate_model_accepts_bundle_environment(tmp_path: Path) -> None:
    """Final evaluations can walk a bundle env without retaining source frames."""
    import pandas as pd

    from quant_rl.envs.env_bundle_builder import build_env_bundle
    from quant_rl.envs.strategies import BaselineStrategy
    from quant_rl.envs.trading_env import TradingEnv
    from quant_rl.eval.rollout import evaluate_model

    class HoldModel:
        @staticmethod
        def predict(observation: Any, deterministic: bool = True) -> tuple[int, None]:
            del observation, deterministic
            return 0, None

    bars = _make_deterministic_bars(n=96)
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
        feature_config_hash="eval-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    frames_result = evaluate_model(HoldModel(), bars, features, obs_window=10)
    bundle_env = TradingEnv.from_bundle(path, obs_window=10, episodic=False)
    bundle_result = evaluate_model(
        HoldModel(), pd.DataFrame(), pd.DataFrame(), obs_window=10, env=bundle_env
    )
    np.testing.assert_array_equal(frames_result["equity"], bundle_result["equity"])
    pd.testing.assert_frame_equal(frames_result["trades"], bundle_result["trades"])
    assert frames_result["reward_sum"] == bundle_result["reward_sum"]
    assert frames_result["n_steps"] == bundle_result["n_steps"]
