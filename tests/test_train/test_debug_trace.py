"""Flight recorder incidents for exceptions, non-finite values, and checks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from quant_rl.train.debug_trace import FlightRecorder

pytestmark = pytest.mark.unit


def _healthy(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "explained_variance": 0.5,
        "std": 0.7,
        "reward_per_1k": {"pnl": -0.01, "strategy": -0.001},
        "shape_on_hold": 0,
        "soft_daily_repeat": 0,
        "soft_year_repeat": 0,
        "corr_val_open_pnl": 0.9,
        "corr_boot_open_pnl": 0.9,
    }
    row.update(overrides)
    return row


def _incidents(directory: Path) -> list[dict[str, object]]:
    path = directory / "debug_incidents.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_exception_writes_traceback_and_previous_snapshots(tmp_path: Path) -> None:
    recorder = FlightRecorder(tmp_path, ring_size=5)
    recorder.record(_healthy(explained_variance=0.4))
    recorder.record(_healthy(explained_variance=0.6))
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        recorder.capture("train exception", _healthy(explained_variance=0.8))
    rows = _incidents(tmp_path)
    assert len(rows) == 1
    assert rows[0]["reason"] == "train exception"
    assert "RuntimeError" in str(rows[0]["traceback"])
    history = rows[0]["history"]
    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0]["explained_variance"] == pytest.approx(0.4)
    assert history[1]["explained_variance"] == pytest.approx(0.6)


def test_nonfinite_value_writes_an_incident(tmp_path: Path) -> None:
    recorder = FlightRecorder(tmp_path)
    recorder.record(_healthy(explained_variance=float("nan")))
    rows = _incidents(tmp_path)
    assert len(rows) == 1
    assert str(rows[0]["reason"]).startswith("non-finite")


def test_extra_check_writes_an_incident(tmp_path: Path) -> None:
    def flag(records: Sequence[Mapping[str, object]]) -> str:
        return "custom" if records and records[-1].get("flag") else ""

    recorder = FlightRecorder(tmp_path, extra_checks=[flag])
    recorder.record(_healthy(flag=1))
    rows = _incidents(tmp_path)
    assert len(rows) == 1
    assert rows[0]["reason"] == "custom"


def test_healthy_record_and_the_first_update_write_no_incident(tmp_path: Path) -> None:
    recorder = FlightRecorder(tmp_path)
    recorder.record(_healthy(explained_variance=0.0, corr_val_open_pnl=0.0, corr_boot_open_pnl=0.0))
    for _ in range(4):
        recorder.record(_healthy(explained_variance=0.1))
    assert _incidents(tmp_path) == []
    assert (tmp_path / "debug.jsonl").exists()
