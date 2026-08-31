"""CLI smoke tests for the W9 evaluation scripts.

Runs each script's ``main()`` against a tiny synthetic CSV fixture and
asserts it exits cleanly and writes a well-formed report.  Training-based
scripts run with the smallest possible budgets so the whole module stays
fast; they catch integration breakage (imports, CLI wiring, split logic,
JSON shape) that unit tests can't.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.compare_encoders import main as compare_encoders_main
from scripts.report_g3 import main as report_g3_main
from scripts.run_baseline_eval import main as run_baseline_eval_main
from scripts.train_lstm_baseline import main as train_lstm_main
from scripts.train_rl import main as train_rl_main

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = str(REPO_ROOT / "quant_rl" / "config" / "default.yaml")

# Module-level: these smoke tests invoke real (tiny-budget) training runs.
# Marked slow so the default `-m 'not slow'` inner loop stays fast; CI still
# runs the full suite. See [tool.pytest.ini_options] markers in pyproject.toml.
pytestmark = pytest.mark.slow


def _make_bars_csv(path: Path, n_bars: int = 3000) -> Path:
    """Write a tiny synthetic bars CSV spanning the default train/test split.

    3000 hourly bars from 2025-10-01 reach into early 2026, so the default
    ``--train-end 2025-12-31`` / ``--test-start 2026-01-01`` split yields
    non-empty in-sample and held-out slices.
    """
    idx = pd.date_range("2025-10-01 00:00", periods=n_bars, freq="1h")
    rng = np.random.default_rng(3)
    close = 20_000.0 + np.cumsum(rng.normal(0.05, 2.0, n_bars))
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.integers(1000, 5000, n_bars).astype(float),
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_csv(path)
    return path


@pytest.fixture(scope="module")
def bars_csv(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    yield _make_bars_csv(tmp_path_factory.mktemp("smoke") / "bars.csv")


@pytest.fixture(autouse=True)
def _chdir_repo_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each script from an isolated cwd so artifacts land in tmp_path."""
    monkeypatch.chdir(tmp_path)


@pytest.mark.integration
class TestBaselineAndReportScripts:
    def test_run_baseline_eval(self, bars_csv: Path, tmp_path: Path) -> None:
        # Arrange
        out_dir = tmp_path / "runs"
        sys.argv = [
            "run_baseline_eval.py",
            "--bars-csv",
            str(bars_csv),
            "--out-dir",
            str(out_dir),
        ]

        # Act
        run_baseline_eval_main()

        # Assert
        for name in ("buy_and_hold", "ema_macd_rsi", "breakout"):
            report_path = out_dir / f"baseline_{name}" / "metrics.json"
            assert report_path.exists(), f"missing {report_path}"
            report = json.loads(report_path.read_text())
            assert report["run_name"] == f"baseline_{name}"
            oos = report["out_of_sample"]
            assert np.isfinite(oos["sharpe"])
            assert oos["n_trades"] >= 0
            assert "sweep_delay" in oos

    def test_report_g3_reads_oos_block(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange — fabricate a run dir in the train_rl.py report shape
        runs_dir = tmp_path / "rl_runs" / "ppo_gru_vae1_seed42"
        runs_dir.mkdir(parents=True)
        (runs_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "run_name": "ppo_gru_vae1_seed42",
                    "in_sample": {"sharpe": 5.0, "max_drawdown": 0.01, "breach_count": 0},
                    "out_of_sample": {"sharpe": 1.5, "max_drawdown": 0.05, "breach_count": 0},
                }
            )
        )
        sys.argv = [
            "report_g3.py",
            "--runs-dir",
            str(runs_dir.parent),
            "--sharpe-threshold",
            "1.0",
        ]

        # Act / Assert: must not raise, and must read the OOS numbers (1.5,
        # not the in-sample 5.0 — an in-sample pass must not leak into G3)
        report_g3_main()
        out = capsys.readouterr().out
        assert "out-of-sample" in out
        assert "1.500" in out


@pytest.mark.integration
class TestTrainingScriptSmoke:
    def test_train_lstm_baseline(self, bars_csv: Path, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "lstm_sweep.pt"
        sys.argv = [
            "train_lstm_baseline.py",
            "--bars-csv",
            str(bars_csv),
            "--epochs",
            "1",
            "--window",
            "24",
            "--horizon",
            "6",
            "--out",
            str(out),
        ]

        # Act
        train_lstm_main()

        # Assert
        assert out.exists()
        metrics_path = tmp_path / "lstm_sweep_metrics.json"
        assert metrics_path.exists()
        report = json.loads(metrics_path.read_text())
        assert 0.0 <= report["val_acc"] <= 1.0
        assert np.isfinite(report["out_of_sample"]["sharpe"])

    def test_train_rl(self, bars_csv: Path, tmp_path: Path) -> None:
        """Smoke-test PPO training with the default config (discrete actions)."""
        # Arrange — smallest viable training budget
        out_dir = tmp_path / "rl_runs"
        sys.argv = [
            "train_rl.py",
            "--bars-csv",
            str(bars_csv),
            "--steps",
            "100",
            "--config",
            CONFIG,
            "--out-dir",
            str(out_dir),
            "--run-name",
            "smoke",
        ]

        # Act
        train_rl_main()

        # Assert
        report = json.loads((out_dir / "smoke" / "metrics.json").read_text())
        assert report["split"]["test_bars"] > 0
        assert np.isfinite(report["in_sample"]["sharpe"])
        assert np.isfinite(report["out_of_sample"]["sharpe"])

    def test_compare_encoders(self, bars_csv: Path, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "encoders.json"
        sys.argv = [
            "compare_encoders.py",
            "--bars-csv",
            str(bars_csv),
            "--steps",
            "100",
            "--config",
            CONFIG,
            "--archs",
            "gru",
            "--out",
            str(out),
        ]

        # Act
        compare_encoders_main()

        # Assert
        results = json.loads(out.read_text())
        assert "gru" in results
        assert np.isfinite(results["gru"]["out_of_sample"]["sharpe"])


def test_smoke_fixture_spans_default_split(tmp_path: Path) -> None:
    """Guard: the synthetic CSV must produce non-empty train and test splits."""
    from quant_rl.data.split import split_train_test

    bars = pd.read_csv(_make_bars_csv(tmp_path / "bars.csv"), index_col=0, parse_dates=True)
    features = bars.select_dtypes(include=["number"])
    train_bars, test_bars, _, _ = split_train_test(bars, features)
    assert not train_bars.empty
    assert not test_bars.empty


def test_repo_root_is_package_parent() -> None:
    assert (REPO_ROOT / "quant_rl").is_dir()
