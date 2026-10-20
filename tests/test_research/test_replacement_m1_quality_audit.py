from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/research/audits/audit_replacement_m1_quality.py"
)
SPEC = importlib.util.spec_from_file_location("replacement_m1_quality_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _row(date: str, time: str, *, low: float = 1.0, close: float = 1.5) -> dict[str, Any]:
    return {
        "<DATE>": date,
        "<TIME>": time,
        "<OPEN>": 1.25,
        "<HIGH>": 2.0,
        "<LOW>": low,
        "<CLOSE>": close,
        "<TICKVOL>": 3,
        "<VOL>": 0,
        "<SPREAD>": 0.1,
    }


def _write_fixture(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_inventory_counts_inclusive_train_and_continuous_report_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    source = tmp_path / "US100.cash_M1_fixture.csv"
    _write_fixture(
        source,
        [
            _row("2024.12.29", "23:59:00"),
            _row("2024.12.30", "00:00:00"),
            _row("2025.12.31", "23:59:00"),
            _row("2026.01.01", "00:00:00"),
            _row("2026.10.16", "23:59:00"),
            _row("2026.10.17", "00:00:00"),
        ],
    )

    result = AUDIT.inventory(source, train_end="2025-12-31", test_start="2026-01-01")

    assert result["rows_before_train_start"] == 1
    assert result["train_rows_inclusive_through_2025_12_31"] == 3
    assert result["report_only_rows_2026_01_01_through_2026_10_16"] == 2
    assert result["rows_after_report_only_end"] == 1
    assert result["timestamp_parse_failures"] == 0
    assert result["duplicate_timestamps"] == 0
    assert result["out_of_order_adjacent_timestamps"] == 0


def test_inventory_reports_anomalies_without_classifying_wall_clock_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    source = tmp_path / "US500.cash_M1_fixture.csv"
    rows = [
        _row("2025.12.31", "12:00:00"),
        _row("2025.12.31", "13:01:00"),
        _row("2025.12.31", "12:00:00", low=1.6),
    ]
    rows[1]["<OPEN>"] = float("inf")
    _write_fixture(source, rows)

    result = AUDIT.inventory(source, train_end="2025-12-31", test_start="2026-01-01")

    assert result["duplicate_timestamps"] == 1
    assert result["out_of_order_adjacent_timestamps"] == 1
    assert result["rows_with_nonfinite_or_missing_numeric_values"] == 1
    assert result["invalid_ohlc_order_rows_with_finite_numeric_values"] == 1
    assert result["observed_adjacent_wall_clock_gaps_seconds"]["greater_than_60"] == 1
    assert "missing_bar_count" not in result
