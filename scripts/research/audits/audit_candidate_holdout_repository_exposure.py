"""Inventory repository evidence for pre-intake exposure to the replacement tail."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis")
SOURCE_FILES = (
    ROOT / "data/US100.cash_202412300105_202610162349.csv",
    ROOT / "data/US500.cash_202412300105_202610162349.csv",
)
OLD_M1_CACHES = (
    ROOT / "cache/US100.cash_M1_v2-full-day.parquet",
    ROOT / "cache/US500.cash_M1_v2-full-day.parquet",
)
SEARCH_ROOTS = (ROOT / "outputs", ROOT / "cache", ROOT / "data/prebuilt_feature_cache")
REPORT = ROOT / "outputs/data_refresh_20261017/candidate_holdout_exposure.json"
EVIDENCE = ROOT / "docs/thesis/evidence/replacement_data_20261017/candidate_holdout_exposure.json"

_TAIL_DATE = re.compile(
    r"2026[-/]07[-/](?:0[4-9]|[12]\d|3[01])"
    r"|2026[-/]08[-/]\d{2}|2026[-/]09[-/]\d{2}"
    r"|2026[-/]10[-/](?:0[1-9]|1[0-5])"
    r"|\b(?:0[4-9]|[12]\d|3[01])[-/.](?:07|08|09|10)[-/.]2026\b"
)
_INTAKE_DAY = re.compile(r"2026[-/]10[-/]16")
_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}
_MAX_TEXT_FILE_BYTES = 128 * 1024 * 1024


def _pre_intake_matches(cutoff_ns: int) -> dict[str, Any]:
    files: list[Path] = []
    excluded_file_count = 0
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.stat().st_mtime_ns >= cutoff_ns:
                continue
            if (
                path.suffix.lower() in _TEXT_SUFFIXES
                and path.stat().st_size <= _MAX_TEXT_FILE_BYTES
            ):
                files.append(path)
            else:
                excluded_file_count += 1

    tail_matches: list[str] = []
    tail_match_examples: list[dict[str, Any]] = []
    intake_day_matches: list[str] = []
    for start in range(0, len(files), 100):
        for path in files[start : start + 100]:
            try:
                contents = path.read_bytes().decode("utf-8", errors="ignore")
                relative_path = str(path.relative_to(ROOT))
                tail_lines = [
                    (number, line[:240])
                    for number, line in enumerate(contents.splitlines(), start=1)
                    if _TAIL_DATE.search(line)
                ]
                if tail_lines:
                    tail_matches.append(relative_path)
                    line_number, sample = tail_lines[0]
                    tail_match_examples.append(
                        {"path": relative_path, "line": line_number, "sample": sample}
                    )
                if _INTAKE_DAY.search(contents):
                    intake_day_matches.append(relative_path)
            except OSError:
                continue
    return {
        "roots": [str(path.relative_to(ROOT)) for path in SEARCH_ROOTS],
        "pre_intake_searchable_text_file_count": len(files),
        "pre_intake_excluded_binary_or_large_file_count": excluded_file_count,
        "candidate_tail_match_files": tail_matches,
        "candidate_tail_match_file_count": len(tail_matches),
        "candidate_tail_match_examples": tail_match_examples,
        "candidate_tail_search_pattern": _TAIL_DATE.pattern,
        "intake_day_reference_file_count": len(intake_day_matches),
        "intake_day_reference_suffix_counts": dict(
            Counter(Path(path).suffix.lower() for path in intake_day_matches)
        ),
        "intake_day_reference_examples": intake_day_matches[:20],
        "intake_day_interpretation": (
            "2026-10-16 references can be artifact-generation timestamps or the candidate "
            "segment endpoint; they are not classified as market-data exposure by this scan."
        ),
        "match_interpretation": (
            "Date-string matches are retained for manual review. A match alone does not prove "
            "that a model or reviewer saw market data or policy results for that date."
        ),
        "maximum_searched_file_bytes": _MAX_TEXT_FILE_BYTES,
        "searched_suffixes": sorted(_TEXT_SUFFIXES),
        "archive_contents_searched": False,
    }


def main() -> None:
    source_info = [
        {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "mtime_local": pd.Timestamp(
                path.stat().st_mtime, unit="s", tz="Europe/Helsinki"
            ).isoformat(),
        }
        for path in SOURCE_FILES
    ]
    cutoff_ns = min(path.stat().st_mtime_ns for path in SOURCE_FILES)

    old_cache_ranges = []
    for path in OLD_M1_CACHES:
        bars = pd.read_parquet(path, columns=["close"])
        old_cache_ranges.append(
            {
                "path": str(path.relative_to(ROOT)),
                "mtime_local": pd.Timestamp(
                    path.stat().st_mtime, unit="s", tz="Europe/Helsinki"
                ).isoformat(),
                "rows": len(bars),
                "first_bar": str(bars.index.min()),
                "last_bar": str(bars.index.max()),
            }
        )

    report = {
        "report_date": pd.Timestamp.now(tz="Europe/Helsinki").isoformat(),
        "candidate_tail": {"start_inclusive": "2026-07-04", "end_inclusive": "2026-10-16"},
        "replacement_source_files": source_info,
        "repository_artifacts_before_earliest_source_mtime": _pre_intake_matches(cutoff_ns),
        "pre_intake_m1_cache_ranges": old_cache_ranges,
        "disposition": "candidate_fresh_segment_for_repository_review_only",
        "limitations": [
            "Filesystem modification times are not an authenticated acquisition ledger.",
            "This inventory cannot determine human or model exposure outside the scanned repository paths.",
            "Archive contents were not searched.",
            "No data-use rights, provider identity, broker calendar, or confirmation eligibility is established.",
            "A candidate segment must remain non-confirmatory until external exposure is reviewed and the model and protocol are frozen.",
        ],
    }
    for path in (REPORT, EVIDENCE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
