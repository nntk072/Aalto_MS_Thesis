#!/usr/bin/env python3
"""Rebuild and transactionally publish full-period bars and validated tick caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configuration import load_config
from configuration.paths import repository_root
from market_data.loading.pipeline import BAR_CACHE_VERSION, build_tick_books, run_pipeline
from market_data.loading.ticks import TICK_CACHE_VERSION

ROOT = repository_root(__file__)
RUN = "20261017_job21020179_rerun1"
TARGET_CACHE = ROOT / "cache" / "us100_us500_20241230_20261016"
BACKUP = ROOT / "cache" / "backups" / f"full-period-rebuild-{RUN}"
REPORT = (
    ROOT
    / "docs"
    / "thesis"
    / "evidence"
    / "replacement_data_20261017"
    / (f"full_period_data_cache_rebuild_{RUN}.json")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if BACKUP.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite prior rebuild evidence")
    cfg = load_config()
    stage = TARGET_CACHE.parent / f".{TARGET_CACHE.name}.{RUN}.stage"
    if stage.exists():
        raise FileExistsError(f"refusing to reuse stage directory: {stage}")
    stage.mkdir(parents=True)
    cfg.data.cache_dir = str(stage)

    market = run_pipeline(cfg, force=True)
    books = build_tick_books(cfg, force=True)
    expected: list[Path] = []
    periods: dict[str, Any] = {}
    for symbol in cfg.data.symbols:
        periods[symbol] = {}
        for timeframe, frame in market[symbol].items():
            if frame.empty:
                raise RuntimeError(f"empty rebuilt bar cache: {symbol}/{timeframe}")
            first, last = pd.Timestamp(frame.index[0]), pd.Timestamp(frame.index[-1])
            if first.date().isoformat() > "2024-12-30" or last.date().isoformat() < "2026-10-16":
                raise RuntimeError(
                    f"bar coverage misses requested full range: {symbol}/{timeframe}"
                )
            if not any(
                "2026-07-01" <= pd.Timestamp(ts).date().isoformat() <= "2026-10-16"
                for ts in frame.index
            ):
                raise RuntimeError(f"July-October rows absent: {symbol}/{timeframe}")
            periods[symbol][timeframe] = {
                "rows": len(frame),
                "first_timestamp": first.isoformat(),
                "last_timestamp": last.isoformat(),
            }
            expected.append(stage / f"{symbol}_{timeframe}_{BAR_CACHE_VERSION}.parquet")

        book = books.get(symbol)
        if book is None or len(book) == 0:
            raise RuntimeError(f"missing rebuilt tick book: {symbol}")
        tick_dates = pd.to_datetime(book._ts[[0, -1]], unit="ns", utc=True).tz_convert(
            str(cfg.data.tz)
        )
        if (
            tick_dates[0].date().isoformat() > "2024-12-30"
            or tick_dates[-1].date().isoformat() < "2026-10-16"
        ):
            raise RuntimeError(f"tick coverage misses requested full range: {symbol}")
        periods[symbol]["ticks"] = {
            "rows": len(book),
            "first_timestamp": tick_dates[0].isoformat(),
            "last_timestamp": tick_dates[-1].isoformat(),
        }
        expected.append(stage / f"{symbol}_ticks_{TICK_CACHE_VERSION}.parquet")

    if len(expected) != 18 or any(not p.is_file() or p.stat().st_size == 0 for p in expected):
        raise RuntimeError("rebuilt cache set is incomplete")

    TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True)
    targets = [TARGET_CACHE / p.name for p in expected]
    prior = [p for p in targets if p.is_file()]
    for path in prior:
        shutil.copy2(path, BACKUP / path.name)
    published: list[Path] = []
    try:
        for source, target in zip(expected, targets, strict=True):
            os.replace(source, target)
            published.append(target)
        result = {
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "train_end_inclusive": "2025-12-31",
            "report_only_test_start": "2026-01-01",
            "full_source_end": "2026-10-16",
            "july_october_included_in_full_cache": True,
            "bar_cache_version": BAR_CACHE_VERSION,
            "tick_cache_version": TICK_CACHE_VERSION,
            "cache_dir": TARGET_CACHE.relative_to(ROOT).as_posix(),
            "backup_dir": BACKUP.relative_to(ROOT).as_posix(),
            "raw_files": {
                symbol: {
                    "m1": str(
                        (ROOT / str(cfg.data.raw_dir) / str(cfg.data.m1_files[symbol])).relative_to(
                            ROOT
                        )
                    ),
                    "ticks": str(
                        (
                            ROOT / str(cfg.data.raw_dir) / str(cfg.data.tick_files[symbol])
                        ).relative_to(ROOT)
                    ),
                }
                for symbol in cfg.data.symbols
            },
            "coverage": periods,
            "artifacts": {
                target.relative_to(ROOT).as_posix(): {
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
                for target in targets
            },
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{REPORT.name}.", dir=REPORT.parent)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        os.replace(tmp_name, REPORT)
    except Exception:
        prior_names = {p.name for p in prior}
        for target in reversed(published):
            saved = BACKUP / target.name
            if target.name in prior_names and saved.is_file():
                shutil.copy2(saved, target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps({"status": "passed", "report": str(REPORT), "cache_files": len(targets)}))


if __name__ == "__main__":
    main()
