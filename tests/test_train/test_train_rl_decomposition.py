"""Tests for the decomposed train_rl.py helper functions.

Verifies that parse_train_args, _setup_rngs, _load_merged_config, and
_build_training_log behave correctly after the main() decomposition.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from quant_rl.train.train_rl import (
    _build_training_log,
    _setup_rngs,
    parse_train_args,
)


class TestParseTrainArgs:
    """Tests for parse_train_args()."""

    def test_default_args(self) -> None:
        """Default args match the expected baseline config."""
        with patch("sys.argv", ["train_rl"]):
            args = parse_train_args()
        assert args.seed == 42
        assert args.algo == "ppo"
        assert args.arch == "tcn"
        assert args.reward == "dsr"
        assert args.strategy == "baseline"
        assert args.mvp is False
        assert args.force is False
        assert args.use_vae is False
        assert args.wandb is False
        assert args.walk_forward is False

    def test_custom_seed(self) -> None:
        with patch("sys.argv", ["train_rl", "--seed", "123"]):
            args = parse_train_args()
        assert args.seed == 123

    def test_mvp_flag(self) -> None:
        with patch("sys.argv", ["train_rl", "--mvp"]):
            args = parse_train_args()
        assert args.mvp is True

    def test_sac_algo(self) -> None:
        with patch("sys.argv", ["train_rl", "--algo", "sac"]):
            args = parse_train_args()
        assert args.algo == "sac"

    def test_sweep_reward(self) -> None:
        with patch("sys.argv", ["train_rl", "--reward", "sweep"]):
            args = parse_train_args()
        assert args.reward == "sweep"


class TestSetupRngs:
    """Tests for _setup_rngs()."""

    def test_numpy_seed_reproducible(self) -> None:
        """Same seed produces same random sequence."""
        _setup_rngs(42)
        a = np.random.random(5)
        _setup_rngs(42)
        b = np.random.random(5)
        np.testing.assert_array_equal(a, b)


class TestBuildTrainingLog:
    """Tests for _build_training_log()."""

    def test_log_contains_required_keys(self) -> None:
        """Training log has all required fields."""
        test_m = type(
            "Metrics",
            (),
            {
                "sharpe": 1.5,
                "max_drawdown": 0.05,
                "n_trades": 10,
                "total_return_pct": 0.02,
            },
        )()
        test_result = {"n_breach_sessions": 0}
        log = _build_training_log(
            seed=42,
            mvp=True,
            algo="ppo",
            arch="tcn",
            reward="dsr",
            timesteps=8192,
            train_bars=11700,
            test_bars=49925,
            test_m=test_m,
            test_result=test_result,
        )
        assert log["seed"] == 42
        assert log["mvp"] is True
        assert log["algo"] == "ppo"
        assert log["arch"] == "tcn"
        assert log["reward"] == "dsr"
        assert log["timesteps"] == 8192
        assert log["train_bars"] == 11700
        assert log["test_bars"] == 49925
        assert log["test_sharpe"] == 1.5
        assert log["test_max_dd"] == 0.05
        assert log["test_trades"] == 10
        assert log["test_return"] == 0.02
        assert log["test_breaches"] == 0
        assert "timestamp" in log

    def test_log_breaches_default_zero(self) -> None:
        """Missing n_breach_sessions defaults to 0."""
        test_m = type(
            "Metrics",
            (),
            {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "total_return_pct": 0.0,
            },
        )()
        log = _build_training_log(
            seed=0,
            mvp=False,
            algo="sac",
            arch="gru",
            reward="sweep",
            timesteps=1000,
            train_bars=100,
            test_bars=200,
            test_m=test_m,
            test_result={},  # no n_breach_sessions
        )
        assert log["test_breaches"] == 0
