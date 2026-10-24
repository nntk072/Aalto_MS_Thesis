"""Tests for the decomposed train_rl.py helper functions.

Verifies that parse_train_args, _setup_rngs, load_merged_config, and
_build_training_log behave correctly after the main() decomposition.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from evaluation.metrics import PerformanceMetrics
from training.artifacts.manifests import _action_space_width, _build_training_log
from training.artifacts.metrics import _write_evaluation_artifacts
from training.callbacks.activity import assess_train_activity
from training.configuration.cli import load_merged_config, parse_train_args
from training.runner.reporting_runtime import finalize_training_run
from training.runner.seed import _setup_rngs
from training.runner.seed_runtime import SeedRunner
from training.setup.environments import _training_episodic, cost_model
from training.setup.models import _initialize_value_head


def test_training_episode_mode_defaults_to_full_split() -> None:
    assert _training_episodic(SimpleNamespace(training={}, env={})) is False
    assert (
        _training_episodic(SimpleNamespace(training={"episode_mode": "full_split"}, env={}))
        is False
    )


def test_bundle_viability_factory_uses_spec_after_training_frames_are_released() -> None:
    from simulation.environment.worker import make_bundle_env

    env_spec = SimpleNamespace(bundle_dir="shared/train")

    factory = SeedRunner._bundle_eval_factory(env_spec)

    assert factory.func is make_bundle_env
    assert factory.args == (env_spec,)


def test_training_episode_mode_can_preserve_legacy_risk_terminated_behavior() -> None:
    cfg = SimpleNamespace(
        training={"episode_mode": "risk_terminated"},
        env={"max_episode_steps": 100},
    )
    assert _training_episodic(cfg) is True


def test_training_episode_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="training.episode_mode"):
        _training_episodic(SimpleNamespace(training={"episode_mode": "random_start"}, env={}))


def test_full_split_training_rejects_artificial_episode_cap() -> None:
    cfg = SimpleNamespace(
        training={"episode_mode": "full_split"},
        env={"max_episode_steps": 100},
    )
    with pytest.raises(ValueError, match="env.max_episode_steps=null"):
        _training_episodic(cfg)


def test_action_space_width_handles_legacy_discrete_space() -> None:
    assert _action_space_width(SimpleNamespace(action_space=SimpleNamespace(shape=(), n=20))) == 20
    assert _action_space_width(SimpleNamespace(action_space=SimpleNamespace(shape=(5,)))) == 5
    assert _action_space_width(SimpleNamespace(action_space=SimpleNamespace(shape=()))) is None


def test_evaluation_artifacts_separate_train_and_report_only_test(tmp_path: Path) -> None:
    train_metrics = PerformanceMetrics(
        total_return_pct=-1.0,
        total_pnl=-1000.0,
        max_drawdown=0.04,
        sharpe=-0.5,
        profit_factor=0.8,
        win_rate=0.4,
        n_trades=10,
    )
    test_metrics = PerformanceMetrics(
        total_return_pct=-6.76,
        total_pnl=-6760.0,
        max_drawdown=0.087,
        sharpe=-4.25,
        profit_factor=0.75,
        win_rate=0.459,
        n_trades=122,
    )

    _write_evaluation_artifacts(
        run_dir=tmp_path,
        seed=42,
        completed_timesteps=106_496,
        train_result={
            "policy_action_mode": "stochastic",
            "action_rng_seed": 42,
            "n_sessions": 100,
            "days_traded": 80,
            "completed_data_split": True,
            "episode_end_reason": "data_end",
        },
        test_result={
            "policy_action_mode": "stochastic",
            "action_rng_seed": 42,
            "n_sessions": 130,
            "days_traded": 32,
            "completed_data_split": False,
            "episode_end_reason": "max_episode_steps",
            "fail_time": "2026-02-16",
        },
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_gate={"ok": False, "breached": True, "reason": "max_loss_stopped_trading"},
        train_activity_gate=assess_train_activity(train_metrics.n_trades),
        account_currency="USD",
    )

    evaluation_dir = tmp_path / "evaluations"
    train = json.loads((evaluation_dir / "train_in_sample.json").read_text())
    test = json.loads((evaluation_dir / "test_report_only.json").read_text())
    summary = (evaluation_dir / "summary.md").read_text()

    assert train["role"] == "in_sample_training_evaluation"
    assert train["policy_action_mode"] == "stochastic"
    assert train["action_rng_seed"] == 42
    assert train["metrics"]["total_pnl"] == -1000.0
    assert train["equity_gate"]["ok"] is False
    assert train["activity_gate"]["status"] == "ACTIVE"
    assert train["zero_trade_diagnosis"] is None
    assert train["peak_dd_policy_fit"]["status"] == "DISABLED"
    assert test["role"] == "report_only_test_evaluation"
    assert test["policy_action_mode"] == "stochastic"
    assert test["action_rng_seed"] == 42
    assert test["metrics"]["total_pnl"] == -6760.0
    assert test["oos_role"].startswith("previously_inspected_holdout")
    assert train["completed_data_split"] is True
    assert train["episode_end_reason"] == "data_end"
    assert test["completed_data_split"] is False
    assert test["episode_end_reason"] == "max_episode_steps"
    assert "| Test (report-only; previously inspected) | -6.76% | -6760.00 USD |" in summary
    assert "Both final replays use stochastic policy actions, matching PPO training." in summary
    assert "Peak-DD training policy: DISABLED" in summary
    assert "Hard-risk survival: FAIL" in summary
    assert json.loads((evaluation_dir / "summary.json").read_text())["training_admissible"] is False


def test_peak_dd_policy_miss_does_not_fail_completed_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.runner.reporting_runtime as reporting_runtime

    monkeypatch.setattr(reporting_runtime, "_build_training_log", lambda **_kwargs: {})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("effective: true\n")
    feature_cache = tmp_path / "features.parquet"
    Path(f"{feature_cache}.hash").write_text("feature-hash\n")
    result = {
        "seed": 50,
        "test_m": object(),
        "test_result": {},
        "train_gate": {
            "ok": False,
            "end_equity": 102_000.0,
            "slope": 0.0,
            "max_peak_trailing_dd": 0.101,
            "peak_trailing_dd_limit": 0.10,
            "breached": False,
            "reason": "peak_trailing_dd",
        },
        "train_activity_gate": assess_train_activity(2),
        "train_result": {
            "entry_state_diagnostics": {"entry_state_machine": True, "candidate_created": 2},
            "entry_diag": {},
        },
        "abort_reason": None,
        "model_num_timesteps": 2_007_040,
        "best_checkpoint_status": "ACTIVE",
        "best_checkpoint_n_trades": 2,
        "deterministic_viability_evaluations": [],
        "phase_timings_seconds": {},
        "run_manifest": "manifest.json",
    }
    cfg = SimpleNamespace(env={}, strategy={})
    args = SimpleNamespace(
        strategy="baseline",
        mvp=False,
        seed=50,
        algo="ppo",
        arch="tcn",
        reward="dsr",
        wandb=False,
        allow_locked_oos_for_selection=False,
    )

    training_log = finalize_training_run(
        cfg=cfg,
        args=args,
        is_multi_seed=False,
        seed_results=[result],
        timesteps=2_000_000,
        train_bar_count=100,
        test_bar_count=50,
        run_dir=run_dir,
        feat_cache=feature_cache,
        feature_columns=["close"],
        seeds=[50],
    )

    assert training_log["training_completed"] is True
    assert training_log["training_admissible"] is False
    assert training_log["overall_status"] == "COMPLETED_POLICY_MISS_RETRAIN_REQUIRED"
    assert training_log["policy_fit_miss_seeds"] == [50]
    assert training_log["hard_survival_failure_seeds"] == []
    assert training_log["hard_risk_ok_by_seed"] == {"50": True}
    assert training_log["retraining_recommended"] is True


def test_zero_train_trades_fail_closed_after_a_survival_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import training.runner.reporting_runtime as reporting_runtime

    monkeypatch.setattr(reporting_runtime, "_build_training_log", lambda **_kwargs: {})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("effective: true\n")
    feature_cache = tmp_path / "features.parquet"
    Path(f"{feature_cache}.hash").write_text("feature-hash\n")
    result = {
        "seed": 50,
        "test_m": object(),
        "test_result": {},
        "train_gate": {"ok": True, "end_equity": 100_000.0, "slope": 0.0, "reason": ""},
        "train_activity_gate": assess_train_activity(0),
        "train_result": {
            "entry_state_diagnostics": {"entry_state_machine": True, "candidate_created": 0},
            "entry_diag": {},
        },
        "abort_reason": None,
        "model_num_timesteps": 20_000_000,
        "best_checkpoint_status": "NO_ACTIVE_CHECKPOINT",
        "best_checkpoint_n_trades": 0,
        "deterministic_viability_evaluations": [
            {"timestep": 5_000_000, "deterministic_n_trades": 0}
        ],
        "phase_timings_seconds": {},
        "run_manifest": "manifest.json",
    }
    cfg = SimpleNamespace(env={}, strategy={})
    args = SimpleNamespace(
        strategy="baseline",
        mvp=False,
        seed=50,
        algo="ppo",
        arch="tcn",
        reward="dsr",
        wandb=False,
        allow_locked_oos_for_selection=False,
    )

    with pytest.raises(SystemExit, match="TRAIN_ACTIVITY_FAILED"):
        finalize_training_run(
            cfg=cfg,
            args=args,
            is_multi_seed=False,
            seed_results=[result],
            timesteps=20_000_000,
            train_bar_count=100,
            test_bar_count=50,
            run_dir=run_dir,
            feat_cache=feature_cache,
            feature_columns=["close"],
            seeds=[50],
        )

    training_log = json.loads((run_dir / "training_log.json").read_text())
    assert training_log["train_survival_ok"] is True
    assert training_log["train_activity_ok"] is False
    assert training_log["overall_status"] == "FAIL_INACTIVE"
    assert training_log["training_admissible"] is False
    assert training_log["zero_trade_diagnosis"] == "FAIL_ENTRY_PATH"
    assert (
        training_log["deterministic_viability_evaluations_by_seed"]["50"]
        == result["deterministic_viability_evaluations"]
    )


def test_resuming_preserves_loaded_value_head_while_fresh_runs_zero_it() -> None:
    model = SimpleNamespace(policy=SimpleNamespace(value_net=torch.nn.Linear(2, 1)))
    with torch.no_grad():
        model.policy.value_net.weight.fill_(3.0)
        model.policy.value_net.bias.fill_(4.0)

    _initialize_value_head(model, resumed=True)
    assert torch.all(model.policy.value_net.weight == 3.0)
    assert torch.all(model.policy.value_net.bias == 4.0)

    _initialize_value_head(model, resumed=False)
    assert torch.all(model.policy.value_net.weight == 0.0)
    assert torch.all(model.policy.value_net.bias == 0.0)


class TestParseTrainArgs:
    """Tests for parse_train_args()."""

    def test_default_args(self) -> None:
        """Default args match the expected baseline config."""
        with patch("sys.argv", ["train_rl"]):
            args = parse_train_args()
        assert args.seed == 42
        assert args.algo == "ppo"
        # Leave the architecture unset so config/quality-variant resolution
        # can select it; train_rl falls back to tcn after config is loaded.
        assert args.arch is None
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
            action_schema={
                "version": "overlay-action-v2",
                "key": "6_dir",
                "width": 6,
                "coordinates": ["direction", "intensity", "stop", "risk", "target", "exit"],
            },
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
        assert log["strategy"] == "baseline"
        assert log["strategy_actions"] is False
        assert log["action_schema"]["coordinates"][0] == "direction"
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

    def test_entry_funnels_keep_training_and_test_diagnostics_separate(self) -> None:
        test_m = type(
            "Metrics",
            (),
            {"sharpe": 0.0, "max_drawdown": 0.0, "n_trades": 0, "total_return_pct": 0.0},
        )()
        log = _build_training_log(
            seed=50,
            mvp=False,
            algo="ppo",
            arch="tcn",
            reward="dsr",
            timesteps=20_000_000,
            train_bars=100,
            test_bars=50,
            test_m=test_m,
            train_result={
                "entry_diag": {"opened": 3, "rejected_gate": 7},
                "entry_state_diagnostics": {"candidate_created": 11, "entered": 3},
            },
            test_result={
                "entry_diag": {"opened": 5, "rejected_gate": 13},
                "entry_state_diagnostics": {"candidate_created": 22, "entered": 5},
            },
        )

        funnels = log["entry_funnel_by_split"]
        assert funnels["train"]["entry_opened"] == 3
        assert funnels["train"]["candidate_created"] == 11
        assert funnels["test"]["entry_opened"] == 5
        assert funnels["test"]["candidate_created"] == 22
        assert "entry_opened" not in log


class TestLoadMergedConfig:
    """Tests for load_merged_config() override ordering (SL Phase 0.1)."""

    def test_overrides_win_over_strategy_overlay(self) -> None:
        """CLI overrides apply after the Idea 1 variant merge.

        ``config/2_po3_ifvg.yaml`` pins ``env.reward_mode: pnl``; an
        explicit ``env.reward_mode=rr`` override must survive the merge
        instead of being silently reverted to pnl.
        """
        import argparse

        ns = argparse.Namespace(config=None, strategy="po3_ifvg", overrides=["env.reward_mode=rr"])
        cfg = load_merged_config(ns)
        assert str(cfg.env.reward_mode) == "rr"

    def test_baseline_leaves_overrides_intact(self) -> None:
        import argparse

        ns = argparse.Namespace(config=None, strategy="baseline", overrides=["env.reward_mode=rr"])
        cfg = load_merged_config(ns)
        assert str(cfg.env.reward_mode) == "rr"


class TestCostModel:
    """Tests for cost_model() YAML wiring (SL Phase 0.5)."""

    def test_cost_model_matches_default_yaml(self) -> None:
        from configuration import load_config

        cfg = load_config([])
        model = cost_model(cfg)
        assert model.slippage_points == float(cfg.costs.slippage_points) == 5.0
        assert model.spread_points == float(cfg.costs.spread_us100)
        assert float(cfg.risk.min_sl_points) >= 2 * model.slippage_points
