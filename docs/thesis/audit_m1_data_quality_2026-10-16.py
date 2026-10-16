"""Reproduce the bounded M1 bar-file quality inventory for the seed-50 screen."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

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


def _inventory(path: Path, campaign_inputs: dict[str, str]) -> tuple[dict[str, Any], pd.Series]:
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

    inventory = {
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
    return inventory, timestamps


def _observed_ny_sessions(
    timestamps: pd.Series,
    *,
    data_tz: str,
    session_tz: str,
    session_start: time,
    session_end: time,
) -> dict[str, Any]:
    """Report training-period NY-session row counts under the configured clock.

    This describes observed rows only. Without an authoritative broker
    calendar it does not label short sessions or absent rows as data errors.
    """
    train_end = pd.Timestamp("2025-12-31").date()
    session_start_dt = datetime.combine(train_end, session_start)
    session_end_dt = datetime.combine(train_end, session_end)
    if session_end_dt <= session_start_dt:
        session_end_dt += timedelta(days=1)
    nominal_rows = int((session_end_dt - session_start_dt).total_seconds() // 60)
    train = timestamps.notna() & (timestamps.dt.date <= train_end)
    data_index = pd.DatetimeIndex(timestamps.loc[train]).tz_localize(data_tz)
    local = data_index.tz_convert(session_tz)
    in_ny = (local.time >= session_start) & (local.time < session_end)
    ny_index = data_index[in_ny]
    ny_dates = np.asarray(local.date)[in_ny]
    if not len(ny_index):
        return {
            "session_timezone": session_tz,
            "data_timezone_assumption": data_tz,
            "training_end_inclusive": "2025-12-31",
            "nominal_m1_rows_for_configured_ny_window": nominal_rows,
            "session_window": f"[{session_start.isoformat()}, {session_end.isoformat()})",
            "observed_session_count": 0,
            "sessions": [],
        }

    row_counts: dict[date, int] = {}
    last_timestamps_ns: dict[date, int] = {}
    timestamps_ns = ny_index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    for session_date_value, timestamp_ns in zip(ny_dates, timestamps_ns, strict=True):
        session_date = cast(date, session_date_value)
        row_counts[session_date] = row_counts.get(session_date, 0) + 1
        last_timestamps_ns[session_date] = int(timestamp_ns)
    sessions: list[dict[str, Any]] = []
    for session_date, observed_count in row_counts.items():
        close_local = pd.Timestamp.combine(session_date, session_end).tz_localize(session_tz)
        expected_last_data_tz = (close_local - pd.Timedelta(minutes=1)).tz_convert(data_tz)
        observed_last_data_tz = pd.Timestamp(
            last_timestamps_ns[session_date], unit="ns", tz=data_tz
        )
        sessions.append(
            {
                "session_date_ct": session_date.isoformat(),
                "observed_m1_rows": int(observed_count),
                "matches_nominal_row_count": int(observed_count) == nominal_rows,
                "observed_last_bar": observed_last_data_tz.isoformat(),
                "nominal_final_bar_present": observed_last_data_tz == expected_last_data_tz,
            }
        )

    counts = [session["observed_m1_rows"] for session in sessions]
    return {
        "session_timezone": session_tz,
        "data_timezone_assumption": data_tz,
        "training_end_inclusive": "2025-12-31",
        "session_window": f"[{session_start.isoformat()}, {session_end.isoformat()})",
        "nominal_m1_rows_for_configured_ny_window": nominal_rows,
        "observed_session_count": len(sessions),
        "sessions_matching_nominal_row_count": sum(
            session["matches_nominal_row_count"] for session in sessions
        ),
        "sessions_with_nominal_final_bar": sum(
            session["nominal_final_bar_present"] for session in sessions
        ),
        "observed_rows_min_median_max": {
            "min": int(min(counts)),
            "median": float(pd.Series(counts).median()),
            "max": int(max(counts)),
        },
        "sessions": sessions,
        "interpretation": (
            f"Observed rows under the configured {data_tz} to {session_tz} conversion. "
            f"Counts are compared with a nominal {nominal_rows}-minute window, but short sessions "
            "are not classified as missing bars or market closures without an "
            "authoritative calendar."
        ),
    }


def _campaign_ny_session_configs(
    campaign_dir: Path,
    jobs: list[dict[str, Any]],
    expected_window: dict[str, str],
) -> list[dict[str, str]]:
    """Verify every saved worker config uses the expected feature NY window."""
    if len(jobs) != 18:
        raise RuntimeError(f"expected 18 campaign jobs, found {len(jobs)}")
    job_keys = [str(job.get("key", "")) for job in jobs]
    if any(not key for key in job_keys) or len(set(job_keys)) != len(job_keys):
        raise RuntimeError("campaign job keys must be present and unique")
    attempts_root = (campaign_dir / "runs" / "attempts").resolve()
    records: list[dict[str, str]] = []
    for job in jobs:
        remote_pid_file = Path(str(job["remote_pid_file"])).resolve()
        attempt_dir = remote_pid_file.parent
        try:
            attempt_dir.relative_to(attempts_root)
        except ValueError as exc:
            raise RuntimeError(
                f"campaign job attempt is outside the attempts tree: {job.get('key')}"
            ) from exc
        artifact_hashes = job.get("artifact_hashes")
        if not isinstance(artifact_hashes, dict):
            raise RuntimeError(f"job has no artifact hash map: {job.get('key')}")
        config_artifacts = [
            (str(relative), str(digest))
            for relative, digest in artifact_hashes.items()
            if str(relative).endswith("/config.yaml")
        ]
        if len(config_artifacts) != 1:
            raise RuntimeError(
                f"expected one saved config artifact for campaign job {job.get('key')}"
            )
        relative, expected_sha = config_artifacts[0]
        config_path = (attempt_dir / relative).resolve()
        try:
            config_path.relative_to(attempt_dir)
        except ValueError as exc:
            raise RuntimeError(f"saved config escaped its attempt directory: {relative}") from exc
        if not config_path.is_file():
            raise FileNotFoundError(f"saved worker config is missing: {config_path}")
        actual_sha = _sha256(config_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"saved worker config hash changed: {config_path}")
        cfg = OmegaConf.load(config_path)
        window = {
            "timezone": str(cfg.features.sessions.tz),
            "start": time.fromisoformat(str(cfg.features.sessions.ny.start)).isoformat(),
            "end": time.fromisoformat(str(cfg.features.sessions.ny.end)).isoformat(),
        }
        if window != expected_window:
            raise RuntimeError(
                f"campaign NY feature window differs from the audited default for "
                f"{job.get('key')}: {window!r} != {expected_window!r}"
            )
        records.append(
            {
                "job_key": str(job["key"]),
                "config_path": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": actual_sha,
                **window,
            }
        )
    if not jobs or len(records) != len(jobs):
        raise RuntimeError("not every campaign worker config was validated")
    return records


def build_report() -> dict[str, Any]:
    campaign_dir = ROOT / "outputs" / "gpu_campaigns" / CAMPAIGN_ID
    campaign_path = campaign_dir / "campaign.json"
    manifest_bytes = campaign_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    inputs = manifest["data_inputs"]
    feature_config_path = ROOT / "quant_rl/config/default.yaml"
    feature_config = OmegaConf.load(feature_config_path)
    session_config = feature_config.features.sessions
    session_tz = str(session_config.tz)
    session_start = time.fromisoformat(str(session_config.ny.start))
    session_end = time.fromisoformat(str(session_config.ny.end))
    expected_window = {
        "timezone": session_tz,
        "start": session_start.isoformat(),
        "end": session_end.isoformat(),
    }
    worker_session_configs = _campaign_ny_session_configs(
        campaign_dir,
        manifest.get("jobs", []),
        expected_window,
    )
    bar_files = {}
    for relative in M1_FILES:
        path = ROOT / relative
        inventory, timestamps = _inventory(path, inputs)
        bar_files[relative] = {
            **inventory,
            "training_ny_session_observation": _observed_ny_sessions(
                timestamps,
                data_tz=str(feature_config.data.tz),
                session_tz=session_tz,
                session_start=session_start,
                session_end=session_end,
            ),
        }
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
        "schema": "aalto.thesis_data_quality_audit.v2",
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
            "recorded_timezone_assumption": (
                f"{feature_config.data.tz} from quant_rl/config/default.yaml; "
                "timestamps in files are timezone-naive broker-clock strings"
            ),
            "feature_session_config": "quant_rl/config/default.yaml",
            "feature_session_config_sha256": _sha256(feature_config_path),
            "ny_session_window": {
                "timezone": session_tz,
                "start": session_start.isoformat(),
                "end_exclusive": session_end.isoformat(),
            },
            "campaign_worker_session_configs": worker_session_configs,
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
            "training_ny_session_observation": (
                f"Under the configured {feature_config.data.tz} to {session_tz} conversion, "
                f"count observed M1 rows in each [{session_start.isoformat()}, "
                f"{session_end.isoformat()}) session and check the nominal final bar. "
                "Do not label deviations as missing bars or closures without a calendar."
            ),
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
            (
                "The terminal recovery campaign lacks current-attempt full-split episode "
                "telemetry for ladder_a0_flat; terminal campaign state alone does not "
                "establish full-split exposure."
            ),
        ],
        "evidence_role": "bounded data-quality inventory; not proof of data rights, vendor authenticity, strategy validity, or confirmatory OOS evidence",
    }


def main() -> None:
    output = ROOT / "docs" / "thesis" / "data_quality_audit_2026-10-16_v4.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing data-quality audit: {output}")
    output.write_text(
        json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
