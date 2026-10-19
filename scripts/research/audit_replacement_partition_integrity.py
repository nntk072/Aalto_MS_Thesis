#!/usr/bin/env python3
"""Verify ordered replacement-data train/test partitions against raw exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT / "docs/thesis/evidence/replacement_data_20261017/data_partition_integrity_20261018.json"
)
SYMBOLS = ("US100.cash", "US500.cash")
TRAIN_END = date(2025, 12, 31)
TEST_START = date(2026, 1, 1)
_BLOCK_SIZE = 8 * 1024 * 1024


def _parse_date(line: bytes, path: Path) -> str:
    try:
        raw = line.decode("utf-8").split("\t", maxsplit=1)[0]
        parsed = datetime.strptime(raw, "%Y.%m.%d").date()
    except (UnicodeDecodeError, ValueError, IndexError) as exc:
        raise ValueError(f"cannot parse first CSV field as MT5 date in {path}: {line!r}") from exc
    return parsed.isoformat()


def _scan(path: Path, *, combined: Any | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    row_count = 0
    first_row: bytes | None = None
    last_tail = b""
    body_bytes = 0
    last_byte = b""
    with path.open("rb") as stream:
        header = stream.readline()
        if not header:
            raise ValueError(f"empty export: {path}")
        digest.update(header)
        if combined is not None:
            combined.update(header)
        for block in iter(lambda: stream.read(_BLOCK_SIZE), b""):
            digest.update(block)
            if combined is not None:
                combined.update(block)
            body_bytes += len(block)
            row_count += block.count(b"\n")
            if first_row is None:
                first_row = block.split(b"\n", maxsplit=1)[0]
                if not first_row and b"\n" in block:
                    first_row = block.split(b"\n", maxsplit=2)[1]
            last_tail = (last_tail + block)[-4096:]
            last_byte = block[-1:]
    if body_bytes and last_byte != b"\n":
        row_count += 1
    if row_count == 0 or first_row is None:
        raise ValueError(f"export has no data rows: {path}")
    last_row = last_tail.rstrip(b"\r\n").rsplit(b"\n", maxsplit=1)[-1]
    return {
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "row_count": row_count,
        "first_date": _parse_date(first_row, path),
        "last_date": _parse_date(last_row, path),
        "header": header,
    }


def audit_partition(source: Path, training: Path, testing: Path) -> dict[str, Any]:
    """Require source bytes to equal train bytes followed by test rows and header."""
    if not source.is_file() or not training.is_file() or not testing.is_file():
        raise FileNotFoundError(f"missing source or partition for {source.name}")

    source_record = _scan(source)
    ordered = hashlib.sha256()
    training_record = _scan(training, combined=ordered)

    test_digest = hashlib.sha256()
    with testing.open("rb") as stream:
        test_header = stream.readline()
        if test_header != training_record["header"] or test_header != source_record["header"]:
            raise ValueError(f"partition headers do not match source: {source.name}")
        test_digest.update(test_header)
        test_rows = 0
        test_body_bytes = 0
        test_first: bytes | None = None
        test_last_tail = b""
        test_last_byte = b""
        for block in iter(lambda: stream.read(_BLOCK_SIZE), b""):
            test_digest.update(block)
            ordered.update(block)
            test_body_bytes += len(block)
            test_rows += block.count(b"\n")
            if test_first is None:
                test_first = block.split(b"\n", maxsplit=1)[0]
                if not test_first and b"\n" in block:
                    test_first = block.split(b"\n", maxsplit=2)[1]
            test_last_tail = (test_last_tail + block)[-4096:]
            test_last_byte = block[-1:]
        if test_body_bytes and test_last_byte != b"\n":
            test_rows += 1

    if test_rows == 0 or test_first is None:
        raise ValueError(f"testing partition has no data rows: {testing}")
    test_last = test_last_tail.rstrip(b"\r\n").rsplit(b"\n", maxsplit=1)[-1]

    train_last_date = date.fromisoformat(training_record["last_date"])
    test_first_date = date.fromisoformat(_parse_date(test_first, testing))
    if training_record["first_date"] != "2024-12-30":
        raise ValueError(f"training partition does not begin on 2024-12-30: {source.name}")
    if train_last_date > TRAIN_END:
        raise ValueError(f"training partition crosses 2025-12-31: {source.name}")
    if test_first_date < TEST_START:
        raise ValueError(f"testing partition starts before 2026-01-01: {source.name}")
    test_last_date = date.fromisoformat(_parse_date(test_last, testing))
    if test_last_date > date(2026, 10, 16):
        raise ValueError(f"testing partition extends beyond 2026-10-16: {source.name}")
    if training_record["row_count"] + test_rows != source_record["row_count"]:
        raise ValueError(f"partition row counts do not sum to source: {source.name}")
    if ordered.hexdigest() != source_record["sha256"]:
        raise ValueError(f"ordered partition bytes do not reconstruct source: {source.name}")

    test_record = {
        "path": testing.resolve().relative_to(ROOT).as_posix(),
        "bytes": testing.stat().st_size,
        "sha256": test_digest.hexdigest(),
        "row_count": test_rows,
        "first_date": _parse_date(test_first, testing),
        "last_date": test_last_date.isoformat(),
    }
    training_record.pop("header")
    source_record.pop("header")
    return {
        "source": source_record,
        "training": training_record,
        "testing": test_record,
        "ordered_partitions_reconstruct_source": True,
    }


def build_report() -> dict[str, Any]:
    source_root = ROOT / "data"
    records: dict[str, Any] = {}
    for symbol in SYMBOLS:
        sources = sorted(source_root.glob(f"{symbol}_*.csv"))
        if not sources:
            raise FileNotFoundError(f"no replacement exports found for {symbol}")
        for source in sources:
            key = source.name
            records[key] = audit_partition(
                source,
                ROOT / "data/training" / key,
                ROOT / "data/testing" / key,
            )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    script = Path(__file__).resolve()
    return {
        "schema": "aalto.replacement_partition_integrity.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_head": head,
        "audit_script": script.relative_to(ROOT).as_posix(),
        "audit_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "source_identity": {
            "user_attested_source": "FTMO Demo server, MetaTrader 5",
            "liquidity_provider": "unknown",
            "training_use": "user-authorized internal model training",
        },
        "date_contract": {
            "training_start_inclusive": "2024-12-30",
            "training_end_inclusive": TRAIN_END.isoformat(),
            "testing_start_inclusive": TEST_START.isoformat(),
            "report_only_test_end_inclusive": "2026-10-16",
            "test_role": "continuous exploratory/report-only; previously inspected",
        },
        "exports": records,
        "summary": {
            "export_count": len(records),
            "all_ordered_partitions_reconstruct_source": all(
                record["ordered_partitions_reconstruct_source"] for record in records.values()
            ),
        },
        "limitations": [
            "The audit proves byte identity and ordered partition reconstruction, not feed authenticity, broker calendar validity, or permission to redistribute raw CSV data.",
            "The raw US100 D1 bar export is absent; the configured pipeline constructs US100 D1 from M1.",
        ],
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
