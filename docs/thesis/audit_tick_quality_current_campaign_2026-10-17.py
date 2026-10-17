#!/usr/bin/env python3
"""Stream and provenance-check the raw tick files from a campaign manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / (
    "outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_"
    "recovery_20261016/campaign.json"
)
DEFAULT_CONFIG = ROOT / "quant_rl/config/default.yaml"
DEFAULT_OUTPUT = ROOT / "docs/thesis/g1_tick_quality_current_recovery_2026-10-17.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_tick_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    columns = ["<DATE>", "<TIME>", "<BID>", "<ASK>"]
    rows = invalid_time = nonfinite = nonpositive = crossed = locked = 0
    order_breaks = duplicate_rows = 0
    min_ns: int | None = None
    max_ns: int | None = None
    previous_ns: int | None = None
    bid_min = ask_min = float("inf")
    bid_max = ask_max = float("-inf")
    observed_m1_minutes: set[int] = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=columns,
        dtype={column: "string" for column in columns},
        keep_default_na=True,
        chunksize=1_000_000,
        engine="c",
    ):
        rows += len(chunk)
        timestamps = pd.to_datetime(
            chunk["<DATE>"].astype(str).to_numpy() + " " + chunk["<TIME>"].astype(str).to_numpy(),
            format="%Y.%m.%d %H:%M:%S.%f",
            errors="coerce",
        )
        invalid_time += int(timestamps.isna().sum())
        bid = pd.to_numeric(chunk["<BID>"], errors="coerce").to_numpy(dtype=np.float64)
        ask = pd.to_numeric(chunk["<ASK>"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(bid) & np.isfinite(ask)
        nonfinite += int((~finite).sum())
        nonpositive += int(((bid <= 0.0) | (ask <= 0.0)).sum())
        crossed += int(((ask < bid) & finite).sum())
        locked += int(((ask == bid) & finite).sum())

        valid_timestamp = ~timestamps.isna()
        usable_quote = finite & (bid > 0.0) & (ask >= bid)
        valid_quote_rows = np.asarray(valid_timestamp, dtype=bool) & usable_quote
        if valid_quote_rows.any():
            current = timestamps[valid_quote_rows].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            observed_m1_minutes.update((current // 60_000_000_000).tolist())
        valid = valid_timestamp
        if valid.any():
            current = timestamps[valid].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            if previous_ns is not None:
                order_breaks += int(current[0] < previous_ns)
                duplicate_rows += int(current[0] == previous_ns)
            if current.size > 1:
                delta = np.diff(current)
                order_breaks += int((delta < 0).sum())
                duplicate_rows += int((delta == 0).sum())
            previous_ns = int(current[-1])
            lower, upper = int(current.min()), int(current.max())
            min_ns = lower if min_ns is None else min(min_ns, lower)
            max_ns = upper if max_ns is None else max(max_ns, upper)

        if finite.any():
            bid_min = min(bid_min, float(bid[finite].min()))
            bid_max = max(bid_max, float(bid[finite].max()))
            ask_min = min(ask_min, float(ask[finite].min()))
            ask_max = max(ask_max, float(ask[finite].max()))

    actual_sha256 = sha256_file(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "rows": rows,
        "invalid_timestamp_rows": invalid_time,
        "nonfinite_bid_ask_rows": nonfinite,
        "nonpositive_bid_ask_rows": nonpositive,
        "crossed_quote_rows_ask_below_bid": crossed,
        "locked_quote_rows_ask_equals_bid": locked,
        "raw_order_breaks": order_breaks,
        "adjacent_duplicate_timestamp_rows": duplicate_rows,
        "min_timestamp_naive": None if min_ns is None else str(pd.Timestamp(min_ns)),
        "max_timestamp_naive": None if max_ns is None else str(pd.Timestamp(max_ns)),
        "bid_min": None if bid_min == float("inf") else bid_min,
        "bid_max": None if bid_max == float("-inf") else bid_max,
        "ask_min": None if ask_min == float("inf") else ask_min,
        "ask_max": None if ask_max == float("-inf") else ask_max,
        "sha256": actual_sha256,
        "campaign_expected_sha256": expected_sha256,
        "campaign_hash_match": actual_sha256 == expected_sha256,
        "_observed_m1_minutes": observed_m1_minutes,
    }


def load_m1_minutes(path: Path) -> set[int]:
    """Read M1 bar-start minutes as epoch-minute integers, with bounded memory."""
    result: set[int] = set()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["<DATE>", "<TIME>"],
        dtype={"<DATE>": "string", "<TIME>": "string"},
        chunksize=500_000,
        engine="c",
    ):
        timestamps = pd.to_datetime(
            chunk["<DATE>"].astype(str).to_numpy() + " " + chunk["<TIME>"].astype(str).to_numpy(),
            format="%Y.%m.%d %H:%M:%S",
            errors="coerce",
        )
        valid = ~timestamps.isna()
        if valid.any():
            current = timestamps[valid].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            result.update((current // 60_000_000_000).tolist())
    return result


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    config_path = args.config.resolve()
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite tick audit: {output_path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"configuration must be a mapping: {config_path}")
    raw_dir = (ROOT / str(cfg.data.raw_dir)).resolve()
    data_inputs = campaign.get("data_inputs", {})
    tick_files = dict(cfg.data.tick_files)
    m1_files = dict(cfg.data.m1_files)
    if not tick_files:
        raise RuntimeError("configured tick_files is empty")

    symbols: dict[str, dict[str, Any]] = {}
    for symbol, configured_name in tick_files.items():
        relative_path = str((Path(str(cfg.data.raw_dir)) / str(configured_name)).as_posix())
        expected_sha256 = data_inputs.get(relative_path)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(f"campaign lacks a SHA-256 input identity for {relative_path}")
        input_path = (raw_dir / str(configured_name)).resolve()
        input_path.relative_to(ROOT)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        m1_name = m1_files.get(symbol)
        if not m1_name:
            raise RuntimeError(f"no configured M1 input for {symbol}")
        m1_relative_path = str((Path(str(cfg.data.raw_dir)) / str(m1_name)).as_posix())
        m1_expected_sha256 = data_inputs.get(m1_relative_path)
        if not isinstance(m1_expected_sha256, str) or len(m1_expected_sha256) != 64:
            raise RuntimeError(f"campaign lacks a SHA-256 input identity for {m1_relative_path}")
        m1_path = (raw_dir / str(m1_name)).resolve()
        m1_path.relative_to(ROOT)
        if not m1_path.is_file():
            raise FileNotFoundError(m1_path)
        m1_minutes = load_m1_minutes(m1_path)
        if not m1_minutes:
            raise RuntimeError(f"M1 input has no parseable bars: {m1_path}")
        tick_result = scan_tick_file(input_path, expected_sha256)
        observed_m1_minutes = tick_result.pop("_observed_m1_minutes")
        covered_minutes = m1_minutes.intersection(observed_m1_minutes)
        train_end = pd.Timestamp(str(cfg.data.split.train_end)).normalize()
        train_end_minute = int(
            (train_end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)).value // 60_000_000_000
        )
        train_m1_minutes = {minute for minute in m1_minutes if minute <= train_end_minute}
        covered_train_minutes = train_m1_minutes.intersection(observed_m1_minutes)
        test_start = pd.Timestamp(str(cfg.data.split.test_start))
        test_start_minute = int(test_start.value // 60_000_000_000)
        test_m1_minutes = {minute for minute in m1_minutes if minute >= test_start_minute}
        covered_test_minutes = test_m1_minutes.intersection(observed_m1_minutes)
        tick_result["m1_path"] = str(m1_path)
        tick_result["m1_bytes"] = m1_path.stat().st_size
        tick_result["m1_sha256"] = sha256_file(m1_path)
        tick_result["m1_campaign_expected_sha256"] = m1_expected_sha256
        tick_result["m1_campaign_hash_match"] = tick_result["m1_sha256"] == m1_expected_sha256
        tick_result["m1_total_unique_minutes"] = len(m1_minutes)
        tick_result["m1_minutes_with_at_least_one_valid_raw_quote"] = len(covered_minutes)
        tick_result["m1_minutes_without_valid_raw_quotes"] = len(m1_minutes) - len(covered_minutes)
        tick_result["m1_minute_valid_quote_coverage_fraction"] = len(covered_minutes) / len(
            m1_minutes
        )
        tick_result["training_split_end_inclusive"] = str(cfg.data.split.train_end)
        tick_result["training_m1_unique_minutes"] = len(train_m1_minutes)
        tick_result["training_m1_minutes_with_valid_raw_quote"] = len(covered_train_minutes)
        tick_result["training_m1_minutes_without_valid_raw_quote"] = len(train_m1_minutes) - len(
            covered_train_minutes
        )
        tick_result["training_m1_valid_quote_coverage_fraction"] = (
            len(covered_train_minutes) / len(train_m1_minutes) if train_m1_minutes else None
        )
        tick_result["testing_split_start_inclusive"] = str(cfg.data.split.test_start)
        tick_result["testing_m1_unique_minutes"] = len(test_m1_minutes)
        tick_result["testing_m1_minutes_with_valid_raw_quote"] = len(covered_test_minutes)
        tick_result["testing_m1_minutes_without_valid_raw_quote"] = len(test_m1_minutes) - len(
            covered_test_minutes
        )
        tick_result["testing_m1_valid_quote_coverage_fraction"] = (
            len(covered_test_minutes) / len(test_m1_minutes) if test_m1_minutes else None
        )
        tick_result["m1_min_timestamp_naive"] = str(
            pd.Timestamp(min(m1_minutes) * 60_000_000_000, unit="ns")
        )
        tick_result["m1_max_timestamp_naive"] = str(
            pd.Timestamp(max(m1_minutes) * 60_000_000_000, unit="ns")
        )
        symbols[str(symbol)] = tick_result

    source_files = (
        Path(__file__).resolve(),
        ROOT / "quant_rl/data/ticks.py",
        ROOT / "quant_rl/data/pipeline.py",
        config_path,
        campaign_path,
    )
    input_hashes_match = all(
        result["campaign_hash_match"] and result["m1_campaign_hash_match"]
        for result in symbols.values()
    )
    report = {
        "schema_version": 1,
        "audit_date": datetime.now().astimezone().date().isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "tracked_worktree_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "campaign_path": str(campaign_path),
        "campaign_fingerprint": campaign.get("fingerprint"),
        "campaign_manifest_sha256": sha256_file(campaign_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "timezone_contract": str(cfg.data.tz),
        "feed_and_data_use_terms": "Not established by available repository evidence.",
        "input_integrity_disposition": "verified" if input_hashes_match else "mismatch",
        "quote_quality_disposition": "descriptive_only_not_acceptance_gated",
        "quote_quality_acceptance_thresholds": None,
        "quote_quality_acceptance_note": (
            "This report records invalid timestamps, malformed/nonpositive/crossed quotes, "
            "ordering, duplicates, and M1-minute coverage. No feed-specific acceptance "
            "thresholds or authoritative source terms are available, so these observations "
            "are not classified as a quality pass or failure."
        ),
        "scope": (
            "Streamed raw bid/ask and timestamp scan, plus exact unique-M1-minute "
            "coverage for finite, positive, non-crossed raw quotes, tied to current "
            "recovery campaign input hashes. This is raw-row coverage, not proof of "
            "effective fill parity. Does not establish feed identity, broker "
            "clock/DST, "
            "data-use rights, stale-quote execution impact, or tick-cache parity."
        ),
        "coverage_definition": (
            "A configured M1 bar-start minute is covered when at least one raw tick "
            "row in that minute has finite positive bid/ask with ask >= bid."
        ),
        "symbols": symbols,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "campaign_fingerprint": report["campaign_fingerprint"],
                "input_integrity_disposition": report["input_integrity_disposition"],
                "quote_quality_disposition": report["quote_quality_disposition"],
                "symbols": {
                    symbol: {
                        key: result[key]
                        for key in (
                            "rows",
                            "nonfinite_bid_ask_rows",
                            "nonpositive_bid_ask_rows",
                            "crossed_quote_rows_ask_below_bid",
                            "raw_order_breaks",
                            "adjacent_duplicate_timestamp_rows",
                            "campaign_hash_match",
                            "m1_campaign_hash_match",
                            "m1_total_unique_minutes",
                            "m1_minutes_with_at_least_one_valid_raw_quote",
                            "m1_minutes_without_valid_raw_quotes",
                            "m1_minute_valid_quote_coverage_fraction",
                            "training_m1_unique_minutes",
                            "training_m1_minutes_with_valid_raw_quote",
                            "training_m1_minutes_without_valid_raw_quote",
                            "training_m1_valid_quote_coverage_fraction",
                        )
                    }
                    for symbol, result in symbols.items()
                },
            },
            indent=2,
        )
    )
    return 0 if input_hashes_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
