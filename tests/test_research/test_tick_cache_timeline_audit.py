"""Read-only tick-cache interval and identity audit contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scripts.research.audits.audit_tick_cache_timeline import audit_caches


def test_tick_cache_audit_reports_intervals_and_identical_logical_data(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ts_ns": [1_735_689_600_000_000_000, 1_735_689_660_000_000_000],
            "bid": [100.0, 101.0],
            "ask": [100.2, 101.2],
        }
    )
    first = tmp_path / "US100.cash_ticks_v2.parquet"
    second = tmp_path / "US100.cash_ticks_v3.parquet"
    frame.to_parquet(first)
    frame.to_parquet(second)

    report = audit_caches([first, second])

    assert report["entries"][0]["rows"] == 2
    assert report["entries"][0]["strictly_increasing_timestamps"] is True
    assert report["entries"][0]["first_tick_utc"] == "2025-01-01T00:00:00+00:00"
    assert report["entries"][0]["last_tick_utc"] == "2025-01-01T00:01:00+00:00"
    comparison = report["comparisons_to_first_path_per_symbol"]["US100.cash"][0]
    assert comparison["same_rows"] is True
    assert comparison["same_logical_content"] is True
    assert comparison["same_first_tick"] is True
    assert comparison["same_last_tick"] is True


def test_tick_cache_audit_distinguishes_cache_with_earlier_coverage(tmp_path: Path) -> None:
    base = 1_735_689_600_000_000_000
    late = tmp_path / "US100.cash_ticks_late.parquet"
    early = tmp_path / "US100.cash_ticks_early.parquet"
    pd.DataFrame({"ts_ns": [base], "bid": [100.0], "ask": [100.2]}).to_parquet(late)
    pd.DataFrame(
        {"ts_ns": [base - 60_000_000_000, base], "bid": [99.0, 100.0], "ask": [99.2, 100.2]}
    ).to_parquet(early)

    report = audit_caches([late, early])

    assert report["entries"][1]["first_tick_utc"] < report["entries"][0]["first_tick_utc"]
    assert (
        report["comparisons_to_first_path_per_symbol"]["US100.cash"][0]["same_logical_content"]
        is False
    )
