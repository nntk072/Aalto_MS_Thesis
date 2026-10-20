#!/usr/bin/env python3
"""Stream raw replacement tick quality and M1-minute coverage through Oct 16."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configuration.paths import repository_root

ROOT = repository_root(__file__)
CONFIG_PATH = ROOT / "config/base/default.yaml"
COVERAGE_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/"
    "tick_execution_coverage_schedule_freeze_final.json"
)
TICK_AUDIT_HELPER = ROOT / "docs/thesis/audit_tick_quality_current_campaign_2026-10-17.py"
TICKS_SOURCE = ROOT / "market_data/loading/ticks.py"
DEFAULT_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/replacement_tick_quality_20261018.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location(
        "replacement_tick_quality_helper", TICK_AUDIT_HELPER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tick audit helper: {TICK_AUDIT_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    cfg = OmegaConf.load(CONFIG_PATH)
    if not isinstance(cfg, DictConfig):
        raise TypeError("default configuration must be a mapping")
    coverage = json.loads(COVERAGE_REPORT.read_text(encoding="utf-8"))
    if coverage.get("split") != {
        "test_start_inclusive": str(cfg.data.split.test_start),
        "train_end_inclusive": str(cfg.data.split.train_end),
    }:
        raise RuntimeError("schedule coverage report split differs from current configuration")
    helper = _load_helper()
    symbols: dict[str, Any] = {}
    data_dir = (ROOT / str(cfg.data.raw_dir)).resolve()

    for symbol in cfg.data.symbols:
        symbol = str(symbol)
        reference = coverage.get("symbols", {}).get(symbol)
        if not isinstance(reference, dict):
            raise ValueError(f"coverage report lacks symbol {symbol}")
        tick_record = reference.get("tick_source")
        m1_record = reference.get("m1_source")
        if not isinstance(tick_record, dict) or not isinstance(m1_record, dict):
            raise ValueError(f"coverage report lacks raw-source hashes for {symbol}")
        tick_path = (data_dir / str(cfg.data.tick_files[symbol])).resolve()
        m1_path = (data_dir / str(cfg.data.m1_files[symbol])).resolve()
        tick_path.relative_to(ROOT)
        m1_path.relative_to(ROOT)
        if not tick_path.is_file() or not m1_path.is_file():
            raise FileNotFoundError(f"missing configured source file for {symbol}")
        if tick_record.get("path") != str(tick_path) or m1_record.get("path") != str(m1_path):
            raise RuntimeError(
                f"configured source path differs from coverage evidence for {symbol}"
            )

        expected_tick_hash = tick_record.get("sha256")
        expected_m1_hash = m1_record.get("sha256")
        if not isinstance(expected_tick_hash, str) or not isinstance(expected_m1_hash, str):
            raise ValueError(f"coverage report has malformed source hashes for {symbol}")

        tick_result = helper.scan_tick_file(tick_path, expected_tick_hash)
        observed_quote_minutes = tick_result.pop("_observed_m1_minutes")
        m1_minutes = helper.load_m1_minutes(m1_path)
        actual_m1_hash = sha256(m1_path)
        if actual_m1_hash != expected_m1_hash:
            raise RuntimeError(f"M1 source hash changed since coverage audit: {symbol}")
        if tick_result["sha256"] != expected_tick_hash:
            raise RuntimeError(f"tick source hash changed since coverage audit: {symbol}")

        train_end = pd.Timestamp(str(cfg.data.split.train_end)).normalize()
        train_end_minute = int(
            (train_end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)).value // 60_000_000_000
        )
        test_start = pd.Timestamp(str(cfg.data.split.test_start))
        test_start_minute = int(test_start.value // 60_000_000_000)
        train_minutes = {minute for minute in m1_minutes if minute <= train_end_minute}
        test_minutes = {minute for minute in m1_minutes if minute >= test_start_minute}
        covered = m1_minutes.intersection(observed_quote_minutes)
        train_covered = train_minutes.intersection(observed_quote_minutes)
        test_covered = test_minutes.intersection(observed_quote_minutes)
        tick_result.update(
            {
                "m1_source_path": str(m1_path),
                "m1_source_sha256": actual_m1_hash,
                "m1_source_matches_coverage_report": True,
                "m1_unique_minutes": len(m1_minutes),
                "m1_minutes_with_at_least_one_valid_raw_quote": len(covered),
                "m1_minutes_without_valid_raw_quotes": len(m1_minutes) - len(covered),
                "m1_minute_valid_quote_coverage_fraction": (
                    len(covered) / len(m1_minutes) if m1_minutes else None
                ),
                "training_end_inclusive": str(cfg.data.split.train_end),
                "training_m1_unique_minutes": len(train_minutes),
                "training_m1_minutes_with_valid_raw_quote": len(train_covered),
                "training_m1_minutes_without_valid_raw_quotes": len(train_minutes)
                - len(train_covered),
                "training_valid_quote_coverage_fraction": (
                    len(train_covered) / len(train_minutes) if train_minutes else None
                ),
                "report_only_test_start_inclusive": str(cfg.data.split.test_start),
                "report_only_test_end_inclusive": "2026-10-16",
                "report_only_test_m1_unique_minutes": len(test_minutes),
                "report_only_test_m1_minutes_with_valid_raw_quote": len(test_covered),
                "report_only_test_m1_minutes_without_valid_raw_quotes": len(test_minutes)
                - len(test_covered),
                "report_only_test_valid_quote_coverage_fraction": (
                    len(test_covered) / len(test_minutes) if test_minutes else None
                ),
            }
        )
        symbols[symbol] = tick_result

    source_paths = [
        Path(__file__).resolve(),
        TICK_AUDIT_HELPER,
        TICKS_SOURCE,
        CONFIG_PATH,
        COVERAGE_REPORT,
    ]
    all_hashes_match = all(
        item["campaign_hash_match"] and item["m1_source_matches_coverage_report"]
        for item in symbols.values()
    )
    return {
        "schema": "aalto.replacement_tick_quality_audit.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "configuration": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(CONFIG_PATH),
            "timestamp_clock": str(cfg.data.timestamp_clock),
            "train_end_inclusive": str(cfg.data.split.train_end),
            "report_only_test_start_inclusive": str(cfg.data.split.test_start),
            "report_only_test_end_inclusive": "2026-10-16",
        },
        "source_identity": {
            "user_attested_source": "FTMO Demo server",
            "underlying_liquidity_provider": "unknown",
            "training_use": "user-authorized",
            "publication_or_export_terms": "not provided",
        },
        "coverage_reference": {
            "path": COVERAGE_REPORT.relative_to(ROOT).as_posix(),
            "sha256": sha256(COVERAGE_REPORT),
        },
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "symbols": symbols,
        "input_integrity_disposition": "verified" if all_hashes_match else "mismatch",
        "quote_quality_disposition": "descriptive_only_not_acceptance_gated",
        "coverage_definition": (
            "A configured M1 bar-start minute is covered when at least one raw tick in that minute "
            "has a valid timestamp and finite positive bid/ask with ask >= bid."
        ),
        "scope_limitations": [
            "This scans raw bid/ask/timestamp rows and M1-minute coverage; it does not quantify stale-quote execution impact or prove fill parity.",
            "No feed-specific quality thresholds or authoritative FTMO CFD schedule are available; observed quote properties are descriptive, not pass/fail acceptance criteria.",
            "Equal stored-millisecond timestamps do not reveal true sub-millisecond event order; the separate tie-policy report covers deterministic source-order resolution.",
            "The continuous 2026-01-01 through 2026-10-16 interval is exposed exploratory/report-only evidence, not a fresh holdout.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "report": str(output),
                "input_integrity_disposition": json.loads(payload)["input_integrity_disposition"],
                "quote_quality_disposition": "descriptive_only_not_acceptance_gated",
                "elapsed_seconds": json.loads(payload)["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
