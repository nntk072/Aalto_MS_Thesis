"""Search pre-intake text inside repository archives for holdout-date references.

The report is a repository exposure screen. It does not establish what a person
or model saw outside these files, and it deliberately does not deserialize
binary checkpoints or treat a date-string match as proof of strategy exposure.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

ROOT = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis")
SEARCH_ROOTS = (ROOT / "outputs", ROOT / "cache", ROOT / "data/prebuilt_feature_cache")
SOURCE_FILES = (
    ROOT / "data/US100.cash_202412300105_202610162349.csv",
    ROOT / "data/US500.cash_202412300105_202610162349.csv",
)
TAIL_DATE = re.compile(
    rb"2026[-/](?:07[-/](?:0[4-9]|[12]\d|3[01])|08[-/]\d{2}|09[-/]\d{2}|"
    rb"10[-/](?:0[1-9]|1[0-5]))|"
    rb"2026(?:07(?:0[4-9]|[12]\d|3[01])|08\d{2}|09\d{2}|10(?:0[1-9]|1[0-5]))|"
    rb"\b(?:0[4-9]|[12]\d|3[01])[-/.](?:07|08|09|10)[-/.]2026\b"
)
ARCHIVE_SUFFIXES = {".zip", ".gz"}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".toml",
}
TEXT_BASENAMES = {"data", "config", "manifest"}
MAX_MEMBER_BYTES = 128 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024
MAX_MATCHES = 500


class _ReadableBinary(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def _scan_stream(stream: _ReadableBinary, *, label: str, matches: list[dict[str, str]]) -> int:
    overlap = b""
    scanned = 0
    while chunk := stream.read(CHUNK_BYTES):
        scanned += len(chunk)
        combined = overlap + chunk
        hit = TAIL_DATE.search(combined)
        if hit and len(matches) < MAX_MATCHES:
            start = max(0, hit.start() - 100)
            end = min(len(combined), hit.end() + 100)
            matches.append(
                {"member": label, "sample": combined[start:end].decode("utf-8", errors="replace")}
            )
        overlap = combined[-256:]
    return scanned


def _archives(cutoff_ns: int) -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in ARCHIVE_SUFFIXES
                and path.stat().st_mtime_ns < cutoff_ns
            ):
                files.append(path)
    return sorted(files)


def _scan_zip(path: Path, member_matches: list[dict[str, str]]) -> dict[str, Any]:
    scanned_members = 0
    skipped_members = 0
    scanned_bytes = 0
    member_name_hits: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if TAIL_DATE.search(info.filename.encode()):
                    member_name_hits.append(info.filename)
                member_path = Path(info.filename)
                is_text = (
                    member_path.suffix.lower() in TEXT_SUFFIXES
                    or member_path.name.lower() in TEXT_BASENAMES
                )
                if not is_text or info.file_size > MAX_MEMBER_BYTES:
                    skipped_members += 1
                    continue
                with archive.open(info) as stream:
                    scanned_bytes += _scan_stream(
                        stream,
                        label=f"{path.relative_to(ROOT)}::{info.filename}",
                        matches=member_matches,
                    )
                scanned_members += 1
        return {
            "path": str(path.relative_to(ROOT)),
            "archive_type": "zip",
            "size_bytes": path.stat().st_size,
            "scanned_text_members": scanned_members,
            "skipped_binary_or_large_members": skipped_members,
            "scanned_uncompressed_text_bytes": scanned_bytes,
            "member_name_tail_hits": member_name_hits,
            "status": "scanned",
        }
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        return {
            "path": str(path.relative_to(ROOT)),
            "archive_type": "zip",
            "size_bytes": path.stat().st_size,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _scan_gzip(path: Path, member_matches: list[dict[str, str]]) -> dict[str, Any]:
    try:
        with gzip.open(path, "rb") as stream:
            scanned = _scan_stream(
                stream,
                label=f"{path.relative_to(ROOT)}::gzip-payload",
                matches=member_matches,
            )
        return {
            "path": str(path.relative_to(ROOT)),
            "archive_type": "gzip",
            "size_bytes": path.stat().st_size,
            "scanned_uncompressed_bytes": scanned,
            "status": "scanned",
        }
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return {
            "path": str(path.relative_to(ROOT)),
            "archive_type": "gzip",
            "size_bytes": path.stat().st_size,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else ROOT / args.out
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")

    missing_sources = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"replacement source files missing: {missing_sources}")
    cutoff_ns = min(path.stat().st_mtime_ns for path in SOURCE_FILES)
    archives = _archives(cutoff_ns)
    member_matches: list[dict[str, str]] = []
    archive_results = [
        _scan_gzip(path, member_matches)
        if path.suffix.lower() == ".gz"
        else _scan_zip(path, member_matches)
        for path in archives
    ]
    errors = [item for item in archive_results if item["status"] != "scanned"]
    report = {
        "schema_version": "candidate-holdout-archive-exposure-v1",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_tail": {"start_inclusive": "2026-07-04", "end_inclusive": "2026-10-16"},
        "source_intake_cutoff_local": datetime.fromtimestamp(cutoff_ns / 1e9)
        .astimezone()
        .isoformat(),
        "archive_count_before_source_intake": len(archives),
        "archive_type_counts": {
            suffix: sum(path.suffix.lower() == suffix for path in archives)
            for suffix in sorted(ARCHIVE_SUFFIXES)
        },
        "archive_scan_errors": errors,
        "archives": archive_results,
        "candidate_tail_member_content_match_count_capped": len(member_matches),
        "candidate_tail_member_content_matches": member_matches,
        "scope": {
            "searched": "Pre-intake ZIP and GZIP archive member names and text-like members; SB3 metadata member named data is included.",
            "not_searched": "Binary checkpoint tensors, pickle payloads, members over 128 MiB, nested archive payloads, archive formats other than ZIP/GZIP, files outside configured repository roots, and external systems or human memory.",
            "interpretation": "A date reference can be source code, metadata, or actual data/results. Matches require contextual review and do not prove exposure; no match cannot establish no human or external model exposure.",
        },
        "disposition": "repository_archive_text_screen_only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "archive_count_before_source_intake",
                    "archive_type_counts",
                    "candidate_tail_member_content_match_count_capped",
                    "scope",
                    "disposition",
                )
            },
            indent=2,
        )
    )
    print(f"report: {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
