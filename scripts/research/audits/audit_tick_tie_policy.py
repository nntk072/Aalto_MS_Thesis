#!/usr/bin/env python3
"""Verify the deterministic source-order tie rule on full replacement TickBooks.

The audit checks every duplicate timestamp group in each validated tick cache.
It verifies that left-sided lookup identifies the first stable source row and
records hashes of the selected first quotes. It cannot recover their true
sub-millisecond chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configuration.paths import repository_root

ROOT = repository_root(__file__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def audit_symbol(
    root: Path,
    symbol: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    cache_info = coverage["symbols"][symbol]["tick_cache"]
    cache_path = Path(str(cache_info["path"])).resolve()
    cache_path.relative_to(root)
    cache_hash = sha256_file(cache_path)
    if cache_hash != cache_info["sha256"]:
        raise ValueError(f"{symbol} tick-cache hash differs from raw/cache audit")

    frame = pd.read_parquet(cache_path, columns=["ts_ns", "bid", "ask"])
    timestamps = frame["ts_ns"].to_numpy(dtype=np.int64, copy=False)
    bids = frame["bid"].to_numpy(copy=False)
    asks = frame["ask"].to_numpy(copy=False)
    if timestamps.size != int(cache_info["rows"]):
        raise ValueError(f"{symbol} tick-cache row count differs from raw/cache audit")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError(f"{symbol} cache timestamps are not nondecreasing")

    group_starts = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1]
    group_ends = np.r_[group_starts[1:], len(timestamps)]
    duplicate_mask = group_ends - group_starts > 1
    duplicate_starts = group_starts[duplicate_mask]
    duplicate_ends = group_ends[duplicate_mask]
    first_rows = duplicate_starts
    last_rows = duplicate_ends - 1
    first_match_positions = np.searchsorted(timestamps, timestamps[first_rows], side="left")
    if not np.array_equal(first_match_positions, first_rows):
        raise ValueError(f"{symbol} left-sided lookup does not select the first tie row")

    first_quotes_differ_from_last = (bids[first_rows] != bids[last_rows]) | (
        asks[first_rows] != asks[last_rows]
    )
    return {
        "tick_cache": {
            "path": str(cache_path),
            "sha256": cache_hash,
            "rows": int(timestamps.size),
            "source_cache_exact_match_in_raw_coverage_report": bool(
                cache_info["source_cache_exact_match"]
            ),
        },
        "timestamp_groups": {
            "unique_timestamp_count": int(group_starts.size),
            "duplicate_timestamp_group_count": int(duplicate_starts.size),
            "duplicate_rows_beyond_first": int(np.sum(duplicate_ends - duplicate_starts - 1)),
            "duplicate_groups_with_different_first_and_last_quote": int(
                first_quotes_differ_from_last.sum()
            ),
            "stable_first_row_selection_verified_for_every_duplicate_group": True,
            "first_source_order_duplicate_quote_sha256": array_sha256(
                timestamps[first_rows], bids[first_rows], asks[first_rows]
            ),
            "last_source_order_duplicate_quote_sha256": array_sha256(
                timestamps[last_rows], bids[last_rows], asks[last_rows]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--prior-tie-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    coverage_path = args.coverage_report
    if not coverage_path.is_absolute():
        coverage_path = ROOT / coverage_path
    coverage_path = coverage_path.resolve()
    coverage_path.relative_to(ROOT)
    tie_report_path = args.prior_tie_report
    if not tie_report_path.is_absolute():
        tie_report_path = ROOT / tie_report_path
    tie_report_path = tie_report_path.resolve()
    tie_report_path.relative_to(ROOT)
    output = args.out if args.out.is_absolute() else ROOT / args.out
    output = output.resolve()
    output.relative_to(ROOT)
    if not coverage_path.is_file():
        raise FileNotFoundError(coverage_path)
    if not tie_report_path.is_file():
        raise FileNotFoundError(tie_report_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tie-policy audit: {output}")

    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    prior_ties = json.loads(tie_report_path.read_text(encoding="utf-8"))
    if coverage["disposition"] != "PRIMARY_COVERAGE_COMPLETE_INPUTS_MATCH":
        raise ValueError("tick coverage report does not establish complete primary input parity")
    if not all(
        details["tick_cache"]["source_cache_exact_match"]
        for details in coverage["symbols"].values()
    ):
        raise ValueError("tick coverage report does not establish tick source/cache parity")

    symbols = {symbol: audit_symbol(ROOT, symbol, coverage) for symbol in coverage["symbols"]}
    for symbol, details in symbols.items():
        observed = details["timestamp_groups"]
        previous = prior_ties["symbols"][symbol]
        if observed["duplicate_timestamp_group_count"] != previous["duplicate_timestamp_groups"]:
            raise ValueError(f"{symbol} duplicate-group count differs from prior tie audit")
        if (
            observed["duplicate_rows_beyond_first"]
            != previous["duplicate_rows_beyond_first_per_timestamp"]
        ):
            raise ValueError(f"{symbol} duplicate-row count differs from prior tie audit")
    source_path = ROOT / "market_data/loading/ticks.py"
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "host": platform.node(),
        "architecture": platform.machine(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": {
            "market_data/loading/ticks.py": sha256_file(source_path),
            "scripts/research/audits/audit_tick_tie_policy.py": sha256_file(
                Path(__file__).resolve()
            ),
            str(coverage_path.relative_to(ROOT)): sha256_file(coverage_path),
            str(tie_report_path.relative_to(ROOT)): sha256_file(tie_report_path),
        },
        "tie_policy": {
            "sort": "np.argsort(timestamp_ns, kind='stable') preserves source-row order within equal timestamps.",
            "lookup": "np.searchsorted(timestamp_ns, arrival_ns, side='left') selects the first source-ordered quote at/after arrival.",
            "scope": "Deterministic resolution of equal stored millisecond timestamps.",
            "does_not_claim": "True sub-millisecond event chronology or original exchange sequencing inside a timestamp tie.",
        },
        "coverage_report": {
            "path": str(coverage_path.relative_to(ROOT)),
            "sha256": sha256_file(coverage_path),
        },
        "prior_tie_report": {
            "path": str(tie_report_path.relative_to(ROOT)),
            "sha256": sha256_file(tie_report_path),
            "counts_match": True,
        },
        "symbols": symbols,
        "limitations": [
            "This verifies the deterministic tie-break against every duplicate timestamp group in the exact validated caches; it does not reconstruct true sub-millisecond ordering.",
            "This is tick-input semantics evidence, not a policy evaluation, PnL result, campaign result, or thesis-readiness finding.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {symbol: value["timestamp_groups"] for symbol, value in symbols.items()}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
