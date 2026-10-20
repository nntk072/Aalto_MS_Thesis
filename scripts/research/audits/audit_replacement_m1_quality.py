#!/usr/bin/env python3
"""Inventory replacement M1 source-file integrity through the report-only end."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from configuration.paths import repository_root

ROOT = repository_root(__file__)
DEFAULT_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/replacement_m1_quality_20261018.json"
)
NAMED_CACHE_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/"
    "named_feature_cache_rebuild_18variant_20261018_job21020179_clockfix.json"
)
DATA_REBUILD_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/"
    "full_period_data_cache_rebuild_20261017_job21020179_rerun1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path, *, train_end: str, test_start: str) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t")
    required = [
        "<DATE>",
        "<TIME>",
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
        "<TICKVOL>",
        "<VOL>",
        "<SPREAD>",
    ]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required MT5 M1 columns: {missing}")

    timestamps = pd.to_datetime(
        frame["<DATE>"].astype(str) + " " + frame["<TIME>"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    numeric_columns = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(numeric_values).all(axis=1)
    o, h, low, c = (
        numeric[name].to_numpy(dtype=np.float64)
        for name in ("<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>")
    )
    valid_ohlc = (h >= low) & (h >= o) & (h >= c) & (low <= o) & (low <= c)
    gaps = timestamps.diff().dt.total_seconds().dropna()
    dates = timestamps.dt.date
    train_end_date = pd.Timestamp(train_end).date()
    test_start_date = pd.Timestamp(test_start).date()
    valid_timestamps = timestamps.dropna()
    if valid_timestamps.empty:
        raise ValueError(f"no parseable timestamps in {path}")

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "symbol": path.name.split("_M1_")[0],
        "row_count": int(len(frame)),
        "timestamp_parse_failures": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "out_of_order_adjacent_timestamps": int((gaps < 0).sum()),
        "first_timestamp_broker_wall_clock": valid_timestamps.iloc[0].isoformat(),
        "last_timestamp_broker_wall_clock": valid_timestamps.iloc[-1].isoformat(),
        "rows_before_train_start": int((dates < pd.Timestamp("2024-12-30").date()).sum()),
        "train_rows_inclusive_through_2025_12_31": int((dates <= train_end_date).sum()),
        "report_only_rows_2026_01_01_through_2026_10_16": int(
            ((dates >= test_start_date) & (dates <= pd.Timestamp("2026-10-16").date())).sum()
        ),
        "rows_after_report_only_end": int((dates > pd.Timestamp("2026-10-16").date()).sum()),
        "rows_with_nonfinite_or_missing_numeric_values": int((~finite_rows).sum()),
        "invalid_ohlc_order_rows_with_finite_numeric_values": int(
            (finite_rows & ~valid_ohlc).sum()
        ),
        "negative_tick_volume_rows": int((numeric["<TICKVOL>"] < 0).sum()),
        "negative_real_volume_rows": int((numeric["<VOL>"] < 0).sum()),
        "negative_spread_rows": int((numeric["<SPREAD>"] < 0).sum()),
        "zero_real_volume_rows": int((numeric["<VOL>"] == 0).sum()),
        "observed_adjacent_wall_clock_gaps_seconds": {
            "greater_than_60": int((gaps > 60).sum()),
            "greater_than_300": int((gaps > 300).sum()),
            "greater_than_3600": int((gaps > 3600).sum()),
            "largest_10": sorted((int(value) for value in gaps if pd.notna(value)), reverse=True)[
                :10
            ],
        },
    }


def build_report() -> dict[str, Any]:
    cfg = OmegaConf.load(ROOT / "config/base/default.yaml")
    train_end = str(cfg.data.split.train_end)
    test_start = str(cfg.data.split.test_start)
    named_report = json.loads(NAMED_CACHE_REPORT.read_text(encoding="utf-8"))
    rebuild_report = json.loads(DATA_REBUILD_REPORT.read_text(encoding="utf-8"))
    if named_report.get("status") != "passed" or rebuild_report.get("status") != "passed":
        raise RuntimeError("replacement cache source reports must both have passed")

    files: dict[str, Any] = {}
    for symbol, name in cfg.data.m1_files.items():
        path = (ROOT / str(cfg.data.raw_dir) / str(name)).resolve()
        row = inventory(path, train_end=train_end, test_start=test_start)
        expected_hash = named_report.get("raw_m1_sha256", {}).get(str(symbol))
        if not isinstance(expected_hash, str):
            raise ValueError(f"named-cache report lacks raw M1 hash for {symbol}")
        row["matches_named_cache_rebuild_source_hash"] = row["sha256"] == expected_hash
        if not row["matches_named_cache_rebuild_source_hash"]:
            raise RuntimeError(f"raw M1 source changed since named-cache rebuild: {symbol}")
        files[str(symbol)] = row

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema": "aalto.replacement_m1_quality_audit.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_head": head,
        "audit_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
        "source_identity": {
            "user_attested_source": "FTMO Demo server",
            "underlying_liquidity_provider": "unknown",
            "training_use": "user-authorized",
            "publication_or_export_terms": "not provided",
            "timezone_assumption": {
                "wall_clock_carrier": str(cfg.data.tz),
                "timestamp_clock_model": str(cfg.data.timestamp_clock),
                "account_specific_export_path_independently_verified": False,
            },
        },
        "data_role": {
            "training_start": "2024-12-30",
            "train_end_inclusive": train_end,
            "report_only_test_start": test_start,
            "report_only_test_end_inclusive": "2026-10-16",
            "test_interval": "continuous; exploratory/report-only",
            "prior_exposure": "January-July 2026 outcomes were previously inspected",
            "confirmatory_or_fresh_holdout": False,
        },
        "input_reports": {
            "named_feature_cache_report": {
                "path": NAMED_CACHE_REPORT.relative_to(ROOT).as_posix(),
                "sha256": sha256(NAMED_CACHE_REPORT),
            },
            "full_period_cache_report": {
                "path": DATA_REBUILD_REPORT.relative_to(ROOT).as_posix(),
                "sha256": sha256(DATA_REBUILD_REPORT),
            },
        },
        "configuration": {
            "config_path": "config/base/default.yaml",
            "config_sha256": sha256(ROOT / "config/base/default.yaml"),
            "symbols": [str(symbol) for symbol in cfg.data.symbols],
            "split": {"train_end_inclusive": train_end, "test_start_inclusive": test_start},
        },
        "m1_files": files,
        "summary": {
            "file_count": len(files),
            "raw_source_hashes_match_feature_rebuild": all(
                bool(row["matches_named_cache_rebuild_source_hash"]) for row in files.values()
            ),
            "zero_timestamp_parse_failures": all(
                row["timestamp_parse_failures"] == 0 for row in files.values()
            ),
            "zero_duplicates": all(row["duplicate_timestamps"] == 0 for row in files.values()),
            "zero_out_of_order": all(
                row["out_of_order_adjacent_timestamps"] == 0 for row in files.values()
            ),
            "zero_nonfinite_or_missing_numeric": all(
                row["rows_with_nonfinite_or_missing_numeric_values"] == 0 for row in files.values()
            ),
            "zero_invalid_ohlc": all(
                row["invalid_ohlc_order_rows_with_finite_numeric_values"] == 0
                for row in files.values()
            ),
        },
        "interpretation_limits": [
            "Elapsed timestamp gaps are observed wall-clock discontinuities; no authoritative FTMO CFD trading calendar was provided to classify them as missing bars.",
            "The scan checks the raw M1 bar files only. Tick quality, feature causality, fold-local fitting, MTF alignment, cache behavior, and model performance require their separate evidence.",
            "M1 CFD VOL is not asserted to be exchange traded volume; the zero-volume count is reported as an export property.",
            "The complete 2026-01-01 through 2026-10-16 interval is exposed exploratory/report-only evidence, not a fresh holdout.",
        ],
        "evidence_role": "bounded replacement M1 source integrity and row coverage inventory; not data-rights, feed-authenticity, schedule-validity, or strategy evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.{datetime.now(UTC).timestamp():.0f}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "passed", "report": str(output)}))


if __name__ == "__main__":
    main()
