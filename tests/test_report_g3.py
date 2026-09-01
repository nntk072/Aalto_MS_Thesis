"""Unit tests for scripts/report_g3.py dual-schema run-report reading.

Master Roadmap Stage 3: report_g3 must read BOTH
  - ``metrics.json`` (``in_sample`` / ``out_of_sample`` blocks) written by
    ``scripts/train_rl.py``, and
  - the flat ``training_log.json`` written by the canonical
    ``quant_rl/train/train_rl.py`` (where the ``test_*`` keys *are* the
    out-of-sample block).

Uses tiny synthetic JSON fixtures — no training, fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from scripts.report_g3 import main as report_g3_main

FLAT_LOG = {
    "seed": 42,
    "mvp": True,
    "algo": "ppo",
    "arch": "tcn",
    "reward": "dsr",
    "timesteps": 8192,
    "test_sharpe": -1.0341740027280075,
    "test_max_dd": 0.025355224396087284,
    "test_trades": 88,
    "test_return": -0.21790967373631429,
    "test_breaches": 0,
    "timestamp": "2026-09-01T05:51:52.561116",
}


def _write(tmp_path: Path, run_name: str, report: dict[str, object], kind: str) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{kind}.json").write_text(json.dumps(report))
    return tmp_path


@pytest.fixture(autouse=True)
def _argv_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "argv", raising=False)


def _run_report_g3(runs_dir: Path, capsys: pytest.CaptureFixture[str], threshold: float = 1.0):
    sys.argv = ["report_g3.py", "--runs-dir", str(runs_dir), "--sharpe-threshold", str(threshold)]
    report_g3_main()
    return capsys.readouterr().out


def test_reads_flat_training_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Canonical entrypoint output (training_log.json) must be read as the
    out-of-sample block — no 'no metrics.json found' failure."""
    runs_dir = _write(tmp_path, "20260901_054947_rl_train_seed42", FLAT_LOG, "training_log")

    out = _run_report_g3(runs_dir, capsys)

    assert "out-of-sample" in out
    assert "-1.034" in out
    assert "ppo" in out and "tcn" in out


def test_reads_metrics_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """scripts/train_rl.py rich schema must keep working (in-sample must NOT
    leak into the G3 verdict)."""
    runs_dir = _write(
        tmp_path,
        "ppo_gru_vae1_seed42",
        {
            "run_name": "ppo_gru_vae1_seed42",
            "in_sample": {"sharpe": 5.0, "max_drawdown": 0.01, "breach_count": 0},
            "out_of_sample": {"sharpe": 1.5, "max_drawdown": 0.05, "breach_count": 0},
        },
        "metrics",
    )

    out = _run_report_g3(runs_dir, capsys, threshold=1.0)

    assert "out-of-sample" in out
    assert "1.500" in out


def test_prefers_metrics_json_when_both_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run dir holding both files must be read via the richer metrics.json."""
    run_dir = tmp_path / "both"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_name": "ppo_gru_vae1_seed42",
                "out_of_sample": {"sharpe": 1.5, "max_drawdown": 0.05, "breach_count": 0},
            }
        )
    )
    flat = dict(FLAT_LOG, test_sharpe=-1.034, algo="sac", arch="gru")
    (run_dir / "training_log.json").write_text(json.dumps(flat))

    out = _run_report_g3(tmp_path, capsys, threshold=1.0)

    assert "1.500" in out  # from metrics.json
    assert "-1.034" not in out  # the flat log must be ignored


def test_empty_runs_dir_raises(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An empty --runs-dir must fail loudly, not silently pass G3."""
    sys.argv = [
        "report_g3.py",
        "--runs-dir",
        str(tmp_path / "nonexistent"),
        "--sharpe-threshold",
        "1.0",
    ]
    with pytest.raises(SystemExit, match="training_log.json"):
        report_g3_main()
