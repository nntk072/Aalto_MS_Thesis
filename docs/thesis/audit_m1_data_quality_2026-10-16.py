"""Reproduce the bounded M1 bar-file quality inventory for the seed-50 screen."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
M1_FILES = (
    "data/US100.cash_M1_202412300105_202607031959.csv",
    "data/US500.cash_M1_202412300105_202607031959.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path, campaign_inputs: dict[str, str]) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t")
    timestamps = pd.to_datetime(
        frame["<DATE>"].astype(str) + " " + frame["<TIME>"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    numeric_columns = (
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
        "<TICKVOL>",
        "<VOL>",
        "<SPREAD>",
    )
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_rows_valid = numeric.notna().all(axis=1)
    open_price = numeric["<OPEN>"]
    high = numeric["<HIGH>"]
    low = numeric["<LOW>"]
    close = numeric["<CLOSE>"]
    ohlc_valid = (
        (high >= low)
        & (high >= open_price)
        & (high >= close)
        & (low <= open_price)
        & (low <= close)
    )
    gaps_seconds = timestamps.diff().dt.total_seconds().dropna()
    relative_path = path.relative_to(ROOT).as_posix()
    digest = _sha256(path)
    first, last = timestamps.iloc[0], timestamps.iloc[-1]

    return {
        "symbol_from_filename": path.name.split("_M1_")[0],
        "sha256_current": digest,
        "sha256_campaign_manifest": campaign_inputs.get(relative_path),
        "sha256_matches_campaign_manifest": digest == campaign_inputs.get(relative_path),
        "size_bytes": path.stat().st_size,
        "row_count": int(len(frame)),
        "unparseable_timestamps": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "out_of_order_adjacent_timestamps": int((gaps_seconds < 0).sum()),
        "first_timestamp_broker_clock_naive": first.strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp_broker_clock_naive": last.strftime("%Y-%m-%d %H:%M:%S"),
        "rows_train_through_2025_12_31": int(
            (timestamps.dt.date <= pd.Timestamp("2025-12-31").date()).sum()
        ),
        "rows_exposed_2026_01_01_through_available_end": int(
            (timestamps.dt.date >= pd.Timestamp("2026-01-01").date()).sum()
        ),
        "rows_with_non_numeric_or_missing_numeric_values": int((~numeric_rows_valid).sum()),
        "invalid_ohlc_order_rows": int((numeric_rows_valid & ~ohlc_valid).sum()),
        "negative_tick_volume_rows": int((numeric["<TICKVOL>"] < 0).sum()),
        "negative_spread_rows": int((numeric["<SPREAD>"] < 0).sum()),
        "zero_real_volume_rows": int((numeric["<VOL>"] == 0).sum()),
        "adjacent_timestamp_gap_counts_seconds": {
            "greater_than_60": int((gaps_seconds > 60).sum()),
            "greater_than_300": int((gaps_seconds > 300).sum()),
            "greater_than_3600": int((gaps_seconds > 3600).sum()),
            "largest": sorted(
                (int(value) for value in gaps_seconds if pd.notna(value)), reverse=True
            )[:10],
        },
    }


def build_report() -> dict[str, Any]:
    campaign_dir = ROOT / "outputs" / "gpu_campaigns" / CAMPAIGN_ID
    campaign_path = campaign_dir / "campaign.json"
    manifest_bytes = campaign_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    inputs = manifest["data_inputs"]
    bar_files = {relative: _inventory(ROOT / relative, inputs) for relative in M1_FILES}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "schema": "aalto.thesis_data_quality_audit.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_head_at_audit": head,
        "working_tree_status_at_audit": dirty_paths,
        "audit_script": "docs/thesis/audit_m1_data_quality_2026-10-16.py",
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__},
        "campaign_id": CAMPAIGN_ID,
        "campaign_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_recorded_git_sha": manifest["git_sha"],
        "data_role": "training through 2025-12-31; 2026-01-01 through 2026-07-03 is previously inspected/exposed development data",
        "configuration": {
            "symbols": ["US100.cash", "US500.cash"],
            "bar_source": "M1 CSV files; higher bar timeframes are resampled by the pipeline",
            "recorded_timezone_assumption": "Etc/GMT-3 fixed UTC+3; timestamps in files are timezone-naive broker-clock strings",
            "train_end_inclusive": "2025-12-31",
            "exposed_period_start": "2026-01-01",
            "available_data_end": "2026-07-03",
        },
        "m1_bar_files": bar_files,
        "tick_inputs_recorded_by_campaign": {
            key: value for key, value in inputs.items() if "_M1_" not in key
        },
        "check_method": {
            "timestamp": "Parse tab-separated <DATE>/<TIME> with %Y.%m.%d %H:%M:%S; timestamps remain naive broker-clock values.",
            "numeric_rows": "Require finite numeric conversion for OHLC, TICKVOL, VOL, and SPREAD columns; report rows with any missing/non-numeric value.",
            "ohlc_order": "Require HIGH >= LOW, OPEN, CLOSE and LOW <= OPEN, CLOSE.",
            "timestamp_gaps": "Count adjacent source-order deltas above 60, 300, and 3600 seconds; do not classify them as missing bars without an authoritative market schedule.",
            "row_roles": "Train rows have date <= 2025-12-31; exposed rows have date >= 2026-01-01.",
        },
        "interpretation": {
            "timestamp_gap_counts": "Elapsed discontinuities include expected market closures and holidays; they are not classified as missing bars without an authoritative trading-calendar schedule.",
            "zero_volume": "MT5 CFD bar exports record VOL=0; tick volume is the activity field. This is not real traded volume.",
        },
        "g1_status": "PARTIAL; this scan covers basic M1 integrity and identity fingerprints only.",
        "open_g1_items": [
            "Vendor/feed identity, export provenance, and applicable data-use/license terms are not established by the CSVs or repository metadata inspected here.",
            "Tick input hashes are recorded in the active campaign manifest but tick-file content/quality was not rescanned in this audit.",
            "Timestamp discontinuities are not compared against an authoritative exchange/broker calendar; they are not asserted to be missing bars.",
            "The fixed Etc/GMT-3 convention is an assumption; source broker timezone and broker DST schedule are not independently verified. Session/DST impact must be treated as a validity limitation.",
            "This scan does not revalidate feature causality/prefix invariance, fold-local fitting, purged walk-forward masks, MTF close/confirmation alignment, or strict cache hash behavior; retain their separate evidence and review them against current sources.",
            "Full-split episode coverage requires terminal deterministic evaluation; the active screen is incomplete.",
        ],
        "evidence_role": "bounded data-quality inventory; not proof of data rights, vendor authenticity, strategy validity, or confirmatory OOS evidence",
    }


def main() -> None:
    output = ROOT / "docs" / "thesis" / "data_quality_audit_2026-10-16.json"
    output.write_text(
        json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
