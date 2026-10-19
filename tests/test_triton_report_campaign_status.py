from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.triton.report_campaign_status import (
    elapsed_for_job,
    estimate_remaining_seconds,
    recent_fps,
)


def test_running_attempt_elapsed_ignores_prior_attempt_end_time() -> None:
    timezone = ZoneInfo("Europe/Helsinki")
    now = dt.datetime.fromisoformat("2026-10-19T12:31:44+00:00").astimezone(timezone)

    assert (
        elapsed_for_job(
            "2026-10-19T12:31:24+00:00",
            "2026-10-19T12:22:21+00:00",
            "running",
            now,
            timezone,
        )
        == "0:00:20"
    )


def test_recent_fps_uses_median_of_last_five_samples(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        "\n".join(f"| fps | {value} |" for value in (100, 200, 300, 400, 500, 600)),
        encoding="utf-8",
    )

    assert recent_fps(log) == (400.0, 5)


def test_recent_fps_reports_available_sample_count(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("| fps | 750 |\n", encoding="utf-8")

    assert recent_fps(log) == (750.0, 1)


def test_recent_fps_returns_none_when_no_samples_exist(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("training has not started\n", encoding="utf-8")

    assert recent_fps(log) is None


def test_estimated_remaining_uses_target_steps_and_recent_fps() -> None:
    assert estimate_remaining_seconds(19_000_000, 20_000_000, 100.0) == 10_000


def test_estimated_remaining_is_unavailable_without_valid_training_rate() -> None:
    assert estimate_remaining_seconds(None, 20_000_000, 100.0) is None
    assert estimate_remaining_seconds(20_000_000, 20_000_000, 100.0) is None
    assert estimate_remaining_seconds(1, 20_000_000, 0.0) is None
