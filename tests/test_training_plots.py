"""Smoke tests for training-progress plots."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from quant_rl.eval.training_plots import (
    plot_entropy_explvar,
    plot_kl_clip,
    plot_learning_curve,
    plot_learning_rate,
    plot_losses,
    save_training_plots,
)


@pytest.fixture()
def training_log(tmp_path) -> pd.DataFrame:
    """Synthetic training log with all expected SB3 columns."""
    n = 20
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "timestep": np.arange(1, n + 1) * 2048,
            "rollout/ep_rew_mean": rng.normal(0, 1, n).cumsum(),
            "rollout/ep_len_mean": rng.uniform(200, 1000, n),
            "train/policy_gradient_loss": rng.uniform(-0.05, 0.05, n),
            "train/value_loss": rng.uniform(0, 10, n),
            "train/entropy_loss": rng.uniform(-2, -0.5, n),
            "train/approx_kl": rng.uniform(0, 0.02, n),
            "train/clip_fraction": rng.uniform(0, 0.5, n),
            "train/explained_variance": rng.uniform(-0.1, 1.0, n),
            "train/learning_rate": np.full(n, 3e-4),
        }
    )
    path = tmp_path / "training_log.csv"
    df.to_csv(path, index=False)
    return df


def test_plot_learning_curve(training_log, tmp_path):
    plot_learning_curve(training_log, out_path=tmp_path / "lc.png")
    assert (tmp_path / "lc.png").stat().st_size > 0


def test_plot_losses(training_log, tmp_path):
    plot_losses(training_log, out_path=tmp_path / "losses.png")
    assert (tmp_path / "losses.png").stat().st_size > 0


def test_plot_entropy_explvar(training_log, tmp_path):
    plot_entropy_explvar(training_log, out_path=tmp_path / "entropy.png")
    assert (tmp_path / "entropy.png").stat().st_size > 0


def test_plot_kl_clip(training_log, tmp_path):
    plot_kl_clip(training_log, out_path=tmp_path / "kl.png")
    assert (tmp_path / "kl.png").stat().st_size > 0


def test_plot_learning_rate(training_log, tmp_path):
    plot_learning_rate(training_log, out_path=tmp_path / "lr.png")
    assert (tmp_path / "lr.png").stat().st_size > 0


def test_save_training_plots(tmp_path):
    """save_training_plots should generate all 5 PNGs."""
    n = 15
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "timestep": np.arange(1, n + 1) * 2048,
            "rollout/ep_rew_mean": rng.normal(0, 1, n),
            "train/policy_gradient_loss": rng.normal(0, 0.01, n),
            "train/value_loss": rng.uniform(0, 5, n),
            "train/entropy_loss": rng.uniform(-2, -0.5, n),
            "train/approx_kl": rng.uniform(0, 0.01, n),
            "train/clip_fraction": rng.uniform(0, 0.3, n),
            "train/explained_variance": rng.uniform(0, 1, n),
            "train/learning_rate": np.full(n, 3e-4),
        }
    )
    log_path = tmp_path / "training_log.csv"
    df.to_csv(log_path, index=False)

    save_training_plots(log_path, out_dir=tmp_path, save_html=False)

    for name in ["learning_curve", "losses", "entropy_explvar", "kl_clip", "lr"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 0, f"Missing {name}.png"


def test_save_training_plots_empty_log(tmp_path):
    """save_training_plots should not crash on missing/empty log."""
    missing = tmp_path / "nonexistent.csv"
    save_training_plots(missing, out_dir=tmp_path, save_html=False)
    # Should not raise, should create no files
    assert not any(tmp_path.glob("*.png"))
