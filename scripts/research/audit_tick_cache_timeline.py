#!/usr/bin/env python3
"""Compare tick-cache coverage intervals and logical content without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_hash(*arrays: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def audit_tick_cache(path: Path) -> dict[str, Any]:
    """Read a cache and report its interval, integrity summaries, and content hash."""
    frame = pd.read_parquet(path, columns=["ts_ns", "bid", "ask"])
    missing = {"ts_ns", "bid", "ask"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    timestamps = frame["ts_ns"].to_numpy(dtype=np.int64)
    bid = frame["bid"].to_numpy(dtype=np.float64)
    ask = frame["ask"].to_numpy(dtype=np.float64)
    if timestamps.size == 0:
        raise ValueError(f"{path} contains no ticks")
    finite = np.isfinite(bid) & np.isfinite(ask)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
        "rows": int(timestamps.size),
        "first_tick_utc": pd.Timestamp(int(timestamps.min()), unit="ns", tz="UTC").isoformat(),
        "last_tick_utc": pd.Timestamp(int(timestamps.max()), unit="ns", tz="UTC").isoformat(),
        "strictly_increasing_timestamps": bool(np.all(np.diff(timestamps) > 0)),
        "finite_bid_ask_rows": int(finite.sum()),
        "positive_bid_ask_rows": int(((bid > 0) & (ask > 0)).sum()),
        "uncrossed_rows": int((bid <= ask).sum()),
        "logical_sha256": _logical_hash(timestamps, bid.astype(np.float32), ask.astype(np.float32)),
    }


def audit_caches(paths: list[Path]) -> dict[str, Any]:
    """Audit requested paths and identify identical logical caches per symbol."""
    entries = [audit_tick_cache(path) for path in paths]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        name = Path(str(entry["path"])).name
        symbol = name.split("_ticks", maxsplit=1)[0]
        grouped.setdefault(symbol, []).append(entry)
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for symbol, symbol_entries in grouped.items():
        reference = symbol_entries[0]
        comparisons[symbol] = [
            {
                "reference": reference["path"],
                "candidate": candidate["path"],
                "same_rows": candidate["rows"] == reference["rows"],
                "same_logical_content": candidate["logical_sha256"] == reference["logical_sha256"],
                "same_first_tick": candidate["first_tick_utc"] == reference["first_tick_utc"],
                "same_last_tick": candidate["last_tick_utc"] == reference["last_tick_utc"],
            }
            for candidate in symbol_entries[1:]
        ]
    return {
        "schema_version": "tick-cache-timeline-audit-v1",
        "entries": entries,
        "comparisons_to_first_path_per_symbol": comparisons,
        "scope_note": (
            "Parquet cache inspection only. Interval bounds do not prove schedule-wide quote "
            "coverage, source provenance, or data-use/publication rights."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_caches(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
