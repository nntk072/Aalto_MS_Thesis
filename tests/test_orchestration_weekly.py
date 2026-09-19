"""Focused lifecycle tests for the weekly thesis orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from quant_rl.orchestration.weekly import BatchSpec, WeeklyOrchestrator


def test_weekly_layout_and_reports(tmp_path: Path) -> None:
    def runner(run):
        (run.run_dir / "metrics.json").write_text('{"sharpe": 1.0}\n')
        return {"status": "completed", "metrics": {"sharpe": 1.0}}

    week = WeeklyOrchestrator(tmp_path, runner=runner).run_week(
        3, [BatchSpec(seeds=(1, 2), steps=8)]
    )
    assert (week / "batches/batch_1/report.md").is_file()
    assert (week / "report.md").is_file()
    manifest = json.loads((week / "manifest.json").read_text())
    runs = manifest["batches"][0]["runs"]
    assert len(runs) == 6
    assert all(run["run_id"].startswith("run_") for run in runs)
    assert all((week / "batches/batch_1" / run["run_id"] / "result.json").is_file() for run in runs)
    assert "baseline" in (week / "report.md").read_text()


def test_failures_are_preserved_and_retried(tmp_path: Path) -> None:
    attempts: dict[str, int] = {}

    def runner(run):
        attempts[run.strategy] = attempts.get(run.strategy, 0) + 1
        if attempts[run.strategy] == 1:
            raise RuntimeError("synthetic failure")
        return {"status": "completed"}

    week = WeeklyOrchestrator(tmp_path, runner=runner).run_week(
        4, [BatchSpec(strategies=("baseline",), seeds=(7,))], retries=1
    )
    runs = json.loads((week / "batches/batch_1/manifest.json").read_text())["runs"]
    assert [run["status"] for run in runs] == ["failed", "completed"]
    assert runs[1]["retry_of"] == runs[0]["run_id"]
    assert "failed" in (week / "batches/batch_1/report.md").read_text()


def test_comparability_metadata_is_stable(tmp_path: Path) -> None:
    week = WeeklyOrchestrator(tmp_path, runner=lambda run: {"status": "completed"}).run_week(
        5, [BatchSpec(strategies=("baseline", "distribution"), seeds=(42,), dataset_hash="abc")]
    )
    manifest = json.loads((week / "batches/batch_1/manifest.json").read_text())
    assert manifest["comparability"]["dataset_hash"] == "abc"
    assert manifest["comparability_key"]
    assert manifest["incomparable_reasons"] == []
