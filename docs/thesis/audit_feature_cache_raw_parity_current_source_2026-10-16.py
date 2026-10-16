#!/usr/bin/env python3
"""Compare current-source raw feature builds with the campaign's existing caches.

This audit is read-only with respect to caches: raw features are built without a
cache path, while existing parquet files and hash sidecars are only read.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from quant_rl.config import apply_overrides, load_config
from quant_rl.data.clean import clean
from quant_rl.data.loader import load_bars
from quant_rl.data.session import add_session_id, add_session_labels
from quant_rl.data.split import get_split_config, make_train_mask
from quant_rl.features.build import (
    FEATURE_CACHE_VERSION,
    build_features,
    feature_cache_content_hash,
    feature_cache_path,
)
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = (
    ROOT
    / "outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
    / "campaign.json"
)
VARIANTS = (
    "ladder_a0_flat",
    "ladder_a1_sl",
    "ladder_a0_flat_po3",
    "ladder_a0_flat_dist",
)
OUT = ROOT / "docs/thesis/g1_feature_cache_raw_parity_current_source_2026-10-16.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolved_config(name: str) -> tuple[DictConfig, dict[str, Any]]:
    resolved = resolve_variant(name)
    cfg = load_config(config_path=ROOT / str(resolved["base_config"]))
    strategy_config = {
        "po3_ifvg": ROOT / "config" / "2_po3_ifvg.yaml",
        "distribution": ROOT / "config" / "3_distribution.yaml",
    }.get(str(resolved["strategy"]))
    if strategy_config is not None:
        merged = OmegaConf.merge(cfg, OmegaConf.load(strategy_config))
        if not isinstance(merged, DictConfig):
            raise TypeError(f"resolved strategy config for {name} is not a mapping")
        cfg = merged
    cfg = apply_overrides(cfg, list(resolved["overrides"]))
    return cfg, resolved


def compare(fresh: pd.DataFrame, cached: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fresh_shape": list(fresh.shape),
        "cached_shape": list(cached.shape),
        "index_equal": bool(fresh.index.equals(cached.index)),
        "columns_equal": list(fresh.columns) == list(cached.columns),
        "compared_columns": 0,
        "changed_columns": [],
        "max_abs_delta": 0.0,
        "value_mismatch_count": 0,
        "nan_mask_mismatch_count": 0,
    }
    if not result["index_equal"] or not result["columns_equal"]:
        return result

    for column in fresh.columns:
        left = pd.to_numeric(fresh[column], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(cached[column], errors="coerce").to_numpy(dtype=np.float64)
        nan_mismatch = np.isnan(left) != np.isnan(right)
        finite = np.isfinite(left) & np.isfinite(right)
        equal = np.ones(len(left), dtype=bool)
        equal[finite] = np.isclose(left[finite], right[finite], rtol=1e-10, atol=1e-12)
        nonfinite = ~finite
        equal[nonfinite] = left[nonfinite] == right[nonfinite]
        both_nan = np.isnan(left) & np.isnan(right)
        equal[both_nan] = True
        mismatch = ~equal | nan_mismatch
        delta = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        result["max_abs_delta"] = max(float(result["max_abs_delta"]), delta)
        result["value_mismatch_count"] += int(mismatch.sum())
        result["nan_mask_mismatch_count"] += int(nan_mismatch.sum())
        result["compared_columns"] += 1
        if mismatch.any():
            result["changed_columns"].append(
                {
                    "column": str(column),
                    "mismatches": int(mismatch.sum()),
                    "max_abs_delta": delta,
                }
            )
    return result


def source_paths() -> list[Path]:
    paths = [
        path
        for folder in (ROOT / "quant_rl" / "features", ROOT / "quant_rl" / "data")
        for path in folder.glob("*.py")
        if path.is_file()
    ]
    paths.extend(
        ROOT / item
        for item in (
            "quant_rl/config/default.yaml",
            "quant_rl/config/__init__.py",
            "quant_rl/train/variant_resolver.py",
            "config/experiments.yaml",
            "config/2_po3_ifvg.yaml",
            "config/3_distribution.yaml",
            "config/features/full_po3_mtf.yaml",
            "scripts/variant_list.txt",
        )
    )
    return sorted(path for path in set(paths) if path.is_file())


def main() -> int:
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    git_head_start = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    base = load_config(config_path=ROOT / "quant_rl/config/default.yaml")
    primary_symbol = str(base.data.primary)
    secondary_symbol = str(base.data.secondary)
    raw_dir = ROOT / str(base.data.raw_dir)
    primary_path = (raw_dir / str(base.data.m1_files[primary_symbol])).resolve()
    secondary_path = (raw_dir / str(base.data.m1_files[secondary_symbol])).resolve()

    primary = add_session_id(
        add_session_labels(
            clean(load_bars(primary_path), tz=str(base.data.tz)),
            tz=str(base.data.tz),
            ny_end=str(base.session.end),
        )
    )
    secondary = add_session_id(
        add_session_labels(
            clean(load_bars(secondary_path), tz=str(base.data.tz)),
            tz=str(base.data.tz),
            ny_end=str(base.session.end),
        )
    )
    train_end, test_start = get_split_config(base)
    train_mask = make_train_mask(pd.DatetimeIndex(primary.index), train_end)
    inputs = {report_path(path): sha256_file(path) for path in (primary_path, secondary_path)}
    results: list[dict[str, Any]] = []
    started_at = datetime.now().astimezone().isoformat()

    for variant in VARIANTS:
        cfg, identity = resolved_config(variant)
        cache = feature_cache_path(
            Path(str(cfg.data.cache_dir)),
            primary_symbol,
            cfg,
            primary,
            train_mask=train_mask,
            secondary=secondary,
        )
        if not cache.is_absolute():
            cache = ROOT / cache
        cache = cache.resolve()
        prebuilt = OmegaConf.select(cfg, "data.prebuilt_feature_cache")
        if prebuilt:
            cache = Path(str(prebuilt)).expanduser()
            if not cache.is_absolute():
                cache = ROOT / cache
            cache = cache.resolve()

        sidecar = Path(str(cache) + ".hash")
        expected_hash = feature_cache_content_hash(
            cfg,
            primary,
            train_mask=train_mask,
            secondary=secondary,
        )
        sidecar_value = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else None
        print(
            f"building variant={variant} cache={cache} expected_hash={expected_hash}",
            flush=True,
        )

        # No cache_path is supplied: force recomputation cannot overwrite a cache.
        fresh = build_features(
            primary,
            secondary=secondary,
            cfg=cfg,
            train_mask=train_mask,
            force=True,
        )
        cached = pd.read_parquet(cache) if cache.is_file() else pd.DataFrame()
        comparison = compare(fresh, cached)
        result = {
            "variant": variant,
            "strategy": identity.get("strategy"),
            "cache_path": report_path(cache),
            "cache_exists": cache.is_file(),
            "cache_bytes": cache.stat().st_size if cache.is_file() else None,
            "cache_sha256": sha256_file(cache) if cache.is_file() else None,
            "sidecar_path": report_path(sidecar),
            "sidecar_value": sidecar_value,
            "expected_feature_hash": expected_hash,
            "sidecar_matches_expected": sidecar_value == expected_hash,
            "htf_sweeps_sidecar_exists": Path(str(cache) + ".htf_sweeps.parquet").is_file(),
            "raw_feature_to_cache": comparison,
        }
        results.append(result)
        report = {
            "audit": "current_source_raw_feature_to_cache_parity",
            "repository_head_start": git_head_start,
            "campaign_id": campaign.get("campaign_id"),
            "campaign_fingerprint": campaign.get("fingerprint"),
            "campaign_source_git_sha": campaign.get("git_sha"),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "python": platform.python_version(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "omegaconf")
            },
            "symbols": [primary_symbol, secondary_symbol],
            "rows": len(primary),
            "start": primary.index[0].isoformat(),
            "end": primary.index[-1].isoformat(),
            "train_end": train_end,
            "test_start": test_start,
            "input_files": inputs,
            "source_hashes": {report_path(path): sha256_file(path) for path in source_paths()},
            "results": results,
            "summary": {
                "profiles": len(results),
                "passed": sum(
                    item["cache_exists"]
                    and item["sidecar_matches_expected"]
                    and item["raw_feature_to_cache"]["index_equal"]
                    and item["raw_feature_to_cache"]["columns_equal"]
                    and item["raw_feature_to_cache"]["value_mismatch_count"] == 0
                    for item in results
                ),
                "failed": sum(
                    not (
                        item["cache_exists"]
                        and item["sidecar_matches_expected"]
                        and item["raw_feature_to_cache"]["index_equal"]
                        and item["raw_feature_to_cache"]["columns_equal"]
                        and item["raw_feature_to_cache"]["value_mismatch_count"] == 0
                    )
                    for item in results
                ),
            },
            "scope_note": (
                "Four representative feature configurations are checked against their existing caches. "
                "Raw feature builds are uncached and the audit never writes cache files. This is not "
                "strategy-performance or complete evaluation-validity evidence."
            ),
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "variant": variant,
                    "cache": result["cache_exists"],
                    "sidecar_matches": result["sidecar_matches_expected"],
                    "index_equal": comparison["index_equal"],
                    "columns_equal": comparison["columns_equal"],
                    "value_mismatch_count": comparison["value_mismatch_count"],
                    "max_abs_delta": comparison["max_abs_delta"],
                }
            ),
            flush=True,
        )
        del fresh, cached

    final_summary = {
        "profiles": len(results),
        "passed": sum(
            item["cache_exists"]
            and item["sidecar_matches_expected"]
            and item["raw_feature_to_cache"]["index_equal"]
            and item["raw_feature_to_cache"]["columns_equal"]
            and item["raw_feature_to_cache"]["value_mismatch_count"] == 0
            for item in results
        ),
    }
    final_summary["failed"] = final_summary["profiles"] - final_summary["passed"]
    print(f"report={OUT} summary={final_summary}", flush=True)
    return 0 if final_summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
