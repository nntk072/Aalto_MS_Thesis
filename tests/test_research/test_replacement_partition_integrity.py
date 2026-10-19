from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/audit_replacement_partition_integrity.py"
)
SPEC = importlib.util.spec_from_file_location("replacement_partition_integrity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<DATE>\t<TIME>\t<BID>\t<ASK>\n"
        + "".join(f"{date}\t{time}\t1\t2\n" for date, time in rows),
        encoding="utf-8",
    )


def test_audit_partition_requires_exact_ordered_split_and_expected_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    source = tmp_path / "US100.cash_M1_fixture.csv"
    training = tmp_path / "data/training/US100.cash_M1_fixture.csv"
    testing = tmp_path / "data/testing/US100.cash_M1_fixture.csv"
    rows = [
        ("2024.12.30", "01:05:00"),
        ("2025.12.31", "23:49:00"),
        ("2026.01.02", "01:05:00"),
        ("2026.10.16", "23:49:00"),
    ]
    _write(source, rows)
    _write(training, rows[:2])
    _write(testing, rows[2:])

    result = AUDIT.audit_partition(source, training, testing)

    assert result["ordered_partitions_reconstruct_source"] is True
    assert result["source"]["row_count"] == 4
    assert result["training"]["row_count"] == 2
    assert result["training"]["first_date"] == "2024-12-30"
    assert result["training"]["last_date"] == "2025-12-31"
    assert result["testing"]["row_count"] == 2
    assert result["testing"]["first_date"] == "2026-01-02"
    assert result["testing"]["last_date"] == "2026-10-16"


def test_audit_partition_rejects_reordered_or_missing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AUDIT, "ROOT", tmp_path)
    source = tmp_path / "US500.cash_M1_fixture.csv"
    training = tmp_path / "data/training/US500.cash_M1_fixture.csv"
    testing = tmp_path / "data/testing/US500.cash_M1_fixture.csv"
    rows = [
        ("2024.12.30", "01:05:00"),
        ("2025.12.31", "23:49:00"),
        ("2026.01.02", "01:05:00"),
    ]
    _write(source, rows)
    _write(training, rows[:1])
    _write(testing, [rows[2], rows[1]])

    with pytest.raises(ValueError, match="ordered partition bytes"):
        AUDIT.audit_partition(source, training, testing)
