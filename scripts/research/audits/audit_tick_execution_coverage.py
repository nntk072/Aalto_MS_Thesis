#!/usr/bin/env python3
"""Audit randomized tick coverage for the trainer's actual NY step schedule.

This is an input-coverage audit, not a training or strategy-quality result. It
loads the exact M1 cache used by ``run_pipeline``, checks it against a fresh M1
rebuild, and compares each configured v3 TickBook cache with a fresh rebuild
from the configured raw tick export. It then applies the same schedule helper
used by the trainer for the explicit ``[0, latency_ms)`` execution scenario.
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
from market_data.loading.pipeline import BAR_CACHE_VERSION

ROOT = repository_root(__file__)

from configuration import load_config  # noqa: E402
from market_data.loading.bars import load_bars  # noqa: E402
from market_data.loading.ticks import TICK_CACHE_VERSION, TickBook, build_tick_book  # noqa: E402
from market_data.splits.definitions import get_split_config, make_train_mask  # noqa: E402
from market_data.transforms.cleaning import clean  # noqa: E402
from market_data.transforms.sessions import add_session_id, add_session_labels  # noqa: E402
from training.configuration.variant_resolution import (  # noqa: E402
    THESIS_RANDOMIZED_FILL_LATENCY_MS,  # noqa: E402
)
from training.setup.data import _tick_execution_split_coverage  # noqa: E402


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


def tickbook_from_cache(path: Path) -> TickBook:
    frame = pd.read_parquet(path)
    required = {"ts_ns", "bid", "ask"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"tick cache missing required columns: {sorted(required - set(frame.columns))}"
        )
    return TickBook(
        frame["ts_ns"].to_numpy(),
        frame["bid"].to_numpy(),
        frame["ask"].to_numpy(),
    )


def m1_from_source(cfg: Any, symbol: str) -> pd.DataFrame:
    training_data = cfg.data.datasets.training
    raw_dir = (ROOT / str(training_data.raw_dir)).resolve()
    source = (raw_dir / str(training_data.m1_files[symbol])).resolve()
    source.relative_to(ROOT)
    bars = clean(load_bars(source), tz=str(cfg.data.tz))
    bars = add_session_labels(bars, tz=str(cfg.data.tz), ny_end=str(cfg.session.end))
    return add_session_id(bars)


def _same_tickbook(left: TickBook, right: TickBook) -> bool:
    return all(
        np.array_equal(a, b, equal_nan=True)
        for a, b in (
            (left._ts, right._ts),
            (left._bid, right._bid),
            (left._ask, right._ask),
        )
    )


def _same_m1(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.index.equals(right.index) and left.equals(right)


def audit_symbol(
    cfg: Any,
    symbol: str,
    *,
    latency_ms: float,
    max_quote_wait_ms: int,
) -> dict[str, Any]:
    training_data = cfg.data.datasets.training
    raw_dir = (ROOT / str(training_data.raw_dir)).resolve()
    cache_dir = (ROOT / str(cfg.data.cache_dir) / "training").resolve()
    m1_source = (raw_dir / str(training_data.m1_files[symbol])).resolve()
    tick_source = (raw_dir / str(training_data.tick_files[symbol])).resolve()
    m1_source.relative_to(ROOT)
    tick_source.relative_to(ROOT)

    m1_cache_path = cache_dir / f"{symbol}_M1_{BAR_CACHE_VERSION}.parquet"
    tick_cache_path = cache_dir / f"{symbol}_ticks_{TICK_CACHE_VERSION}.parquet"
    if not m1_cache_path.is_file():
        raise FileNotFoundError(f"training M1 cache not found: {m1_cache_path}")
    if not tick_cache_path.is_file():
        raise FileNotFoundError(f"training tick cache not found: {tick_cache_path}")

    m1_cache = pd.read_parquet(m1_cache_path)
    m1_source_frame = m1_from_source(cfg, symbol)
    m1_cache_matches_source = _same_m1(m1_cache, m1_source_frame)

    cached_book = tickbook_from_cache(tick_cache_path)
    rebuilt_book = build_tick_book(
        tick_source,
        tz=str(cfg.data.tz),
        session_start=None,
        session_end=None,
        cache_path=None,
    )
    tick_cache_matches_source = _same_tickbook(cached_book, rebuilt_book)

    train_end, test_start = get_split_config(cfg)
    m1_index = pd.DatetimeIndex(m1_cache.index)
    train_mask = make_train_mask(m1_index, train_end)
    coverage_cache = _tick_execution_split_coverage(
        cached_book,
        m1_index,
        train_mask,
        str(test_start),
        session_labels=m1_cache["session"],
        max_latency_ns_exclusive=int(latency_ms * 1_000_000),
        max_quote_wait_ms=max_quote_wait_ms,
    )
    coverage_rebuilt = _tick_execution_split_coverage(
        rebuilt_book,
        m1_index,
        train_mask,
        str(test_start),
        session_labels=m1_cache["session"],
        max_latency_ns_exclusive=int(latency_ms * 1_000_000),
        max_quote_wait_ms=max_quote_wait_ms,
    )

    return {
        "is_configured_primary": symbol == str(cfg.data.primary),
        "m1_source": {
            "path": str(m1_source),
            "bytes": m1_source.stat().st_size,
            "sha256": sha256_file(m1_source),
        },
        "m1_cache": {
            "path": str(m1_cache_path),
            "bytes": m1_cache_path.stat().st_size,
            "sha256": sha256_file(m1_cache_path),
            "rows": len(m1_cache),
            "source_cache_exact_match": m1_cache_matches_source,
            "columns": [str(column) for column in m1_cache.columns],
            "first_bar": str(m1_cache.index[0]),
            "last_bar": str(m1_cache.index[-1]),
        },
        "tick_source": {
            "path": str(tick_source),
            "bytes": tick_source.stat().st_size,
            "sha256": sha256_file(tick_source),
        },
        "tick_cache": {
            "path": str(tick_cache_path),
            "bytes": tick_cache_path.stat().st_size,
            "sha256": sha256_file(tick_cache_path),
            "rows": len(cached_book),
            "array_sha256": array_sha256(cached_book._ts, cached_book._bid, cached_book._ask),
            "source_cache_exact_match": tick_cache_matches_source,
        },
        "fresh_tick_rebuild": {
            "rows": len(rebuilt_book),
            "array_sha256": array_sha256(rebuilt_book._ts, rebuilt_book._bid, rebuilt_book._ask),
            "coverage": coverage_rebuilt,
        },
        "training_tick_cache_coverage": coverage_cache,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latency-ms", type=float, default=30.0)
    parser.add_argument("--max-quote-wait-ms", type=int, default=60_000)
    args = parser.parse_args()
    if not 0.0 < args.latency_ms <= 30.0:
        parser.error("--latency-ms must be greater than 0 and no more than 30")
    if args.max_quote_wait_ms <= 0:
        parser.error("--max-quote-wait-ms must be positive")

    output = args.out if args.out.is_absolute() else ROOT / args.out
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")

    cfg = load_config()
    symbols = [str(symbol) for symbol in cfg.data.symbols]
    symbol_reports = {
        symbol: audit_symbol(
            cfg,
            symbol,
            latency_ms=args.latency_ms,
            max_quote_wait_ms=args.max_quote_wait_ms,
        )
        for symbol in symbols
    }
    primary = str(cfg.data.primary)
    primary_report = symbol_reports[primary]
    primary_train = primary_report["training_tick_cache_coverage"]["train"]
    primary_test = primary_report["training_tick_cache_coverage"]["test"]
    primary_integrity = (
        primary_report["m1_cache"]["source_cache_exact_match"]
        and primary_report["tick_cache"]["source_cache_exact_match"]
    )
    primary_coverage_complete = (
        primary_train["unavailable_count"] == 0 and primary_test["unavailable_count"] == 0
    )

    tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "config/base/default.yaml",
        ROOT / "market_data/loading/pipeline.py",
        ROOT / "market_data/transforms/cleaning.py",
        ROOT / "market_data/loading/bars.py",
        ROOT / "market_data/loading/pipeline.py",
        ROOT / "market_data/transforms/sessions.py",
        ROOT / "market_data/splits/definitions.py",
        ROOT / "market_data/loading/ticks.py",
        ROOT / "training/configuration/variant_resolution.py",
        ROOT / "training/runner/training.py",
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).splitlines(),
        "tracked_worktree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "host": platform.node(),
        "architecture": platform.machine(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config_path": "config/base/default.yaml",
        "config_sha256": sha256_file(ROOT / "config/base/default.yaml"),
        "configured_primary": primary,
        "configured_secondary": str(cfg.data.secondary),
        "split": {
            "train_end_inclusive": str(cfg.data.split.train_end),
            "test_start_inclusive": str(cfg.data.split.test_start),
        },
        "execution_contract": {
            "base_config_randomized_latency_ms": float(cfg.execution.randomized_fill_latency_ms),
            "thesis_variant_resolver_randomized_latency_ms": THESIS_RANDOMIZED_FILL_LATENCY_MS,
            "audited_randomized_latency_interval_ms": f"[0, {args.latency_ms})",
            "coverage_audit_matches_thesis_variant_resolver": (
                float(args.latency_ms) == THESIS_RANDOMIZED_FILL_LATENCY_MS
            ),
            "max_tick_quote_wait_ms": args.max_quote_wait_ms,
            "schedule": "NY-labeled M1 environment steps, each mapped to the next market-bar timestamp as in train_rl._tick_execution_split_coverage",
            "coverage_semantics": "Conservative preflight: every allowed latency draw must have a valid cached quote at/after arrival and within the quote-wait bound from each candidate environment step timestamp.",
        },
        "feed_identity_and_data_use_terms": "Unknown; no provider or publication-use terms are documented in the available repository evidence.",
        "disposition": (
            "PRIMARY_COVERAGE_COMPLETE_INPUTS_MATCH"
            if primary_integrity and primary_coverage_complete
            else "BLOCKED_PRIMARY_TICK_COVERAGE_OR_CACHE_INPUT_MISMATCH"
        ),
        "symbols": symbol_reports,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "scope_limitations": [
            "This is an input coverage and exact cache-parity audit, not a backtest, model evaluation, or strategy result.",
            "US500 coverage is descriptive only; it does not authorize changing the configured US100 primary instrument.",
            "Coverage and cache parity do not establish feed identity, data-use rights, or publication permission.",
            "No campaign readiness or thesis readiness is implied.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(output),
                "disposition": report["disposition"],
                "primary": primary,
                "train_coverage": primary_train,
                "test_coverage": primary_test,
                "symbols": {
                    symbol: {
                        "m1_cache_exact": details["m1_cache"]["source_cache_exact_match"],
                        "tick_cache_exact": details["tick_cache"]["source_cache_exact_match"],
                        "train_coverage": details["training_tick_cache_coverage"]["train"],
                        "test_coverage": details["training_tick_cache_coverage"]["test"],
                    }
                    for symbol, details in symbol_reports.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
