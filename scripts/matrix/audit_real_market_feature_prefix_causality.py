#!/usr/bin/env python3
"""Check real-market feature prefix invariance at frozen train/OOS cutoffs.

Rebuild the full feature matrix from configured raw M1 files, then rebuild
prefixes ending at selected timestamps. If future bars affect any earlier
feature, the prefix and full matrices diverge before that cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, cast

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
)
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUTOFFS = (
    "2025-12-15 12:37:00",
    "2026-03-09 12:37:00",
    "2026-06-15 12:37:00",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


AUDIT_SCRIPT_SHA256 = sha256_file(Path(__file__).resolve())


def load_market_bars(cfg: Any, symbol: str) -> tuple[pd.DataFrame, Path]:
    source = (ROOT / str(cfg.data.raw_dir) / str(cfg.data.m1_files[symbol])).resolve()
    raw = load_bars(source)
    bars = clean(raw, tz=str(cfg.data.tz))
    bars = add_session_labels(bars, tz=str(cfg.data.tz), ny_end=str(cfg.session.end))
    bars = add_session_id(bars)
    return bars, source


def compare_prefix(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    expected = reference.loc[:cutoff]
    if not expected.index.equals(candidate.index):
        return {
            "cutoff": cutoff.isoformat(),
            "passed": False,
            "reason": "prefix indexes differ",
            "reference_rows": len(expected),
            "prefix_rows": len(candidate),
        }
    if list(expected.columns) != list(candidate.columns):
        return {
            "cutoff": cutoff.isoformat(),
            "passed": False,
            "reason": "feature columns differ",
            "reference_columns": len(expected.columns),
            "prefix_columns": len(candidate.columns),
        }

    changed_features: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    compared_values = 0
    for column in expected.columns:
        left = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(candidate[column], errors="coerce").to_numpy(dtype=np.float64)
        left_nan = np.isnan(left)
        right_nan = np.isnan(right)
        nan_mismatch = left_nan != right_nan
        finite = np.isfinite(left) & np.isfinite(right)
        close = np.ones(len(left), dtype=bool)
        close[finite] = np.isclose(left[finite], right[finite], rtol=1e-10, atol=1e-12)
        nonfinite = ~finite
        close[nonfinite] = left[nonfinite] == right[nonfinite]
        close[left_nan & right_nan] = True
        compared_values += int(len(left))
        if finite.any():
            delta = np.abs(left[finite] - right[finite])
            column_max = float(delta.max(initial=0.0))
            max_abs_delta = max(max_abs_delta, column_max)
        else:
            column_max = 0.0
        mismatch = np.flatnonzero(~close)
        if len(mismatch):
            first = int(mismatch[0])
            changed_features.append(
                {
                    "feature": str(column),
                    "mismatch_rows": int(len(mismatch)),
                    "first_timestamp": expected.index[first].isoformat(),
                    "max_abs_delta": column_max,
                    "nan_mask_mismatches": int(nan_mismatch.sum()),
                }
            )
    return {
        "cutoff": cutoff.isoformat(),
        "passed": not changed_features,
        "rows": len(expected),
        "features": len(expected.columns),
        "compared_values": compared_values,
        "max_abs_delta": max_abs_delta,
        "changed_feature_count": len(changed_features),
        "changed_features": changed_features[:100],
    }


def config_for_variant(name: str) -> tuple[Any, dict[str, Any]]:
    resolved = resolve_variant(name)
    cfg: DictConfig = load_config(config_path=ROOT / str(resolved["base_config"]))
    strategy_config = {
        "po3_ifvg": ROOT / "config" / "2_po3_ifvg.yaml",
        "distribution": ROOT / "config" / "3_distribution.yaml",
    }.get(str(resolved["strategy"]))
    if strategy_config is not None:
        merged_cfg = OmegaConf.merge(cfg, OmegaConf.load(strategy_config))
        if not isinstance(merged_cfg, DictConfig):
            raise TypeError("resolved strategy config must be a mapping")
        cfg = merged_cfg
    cfg = apply_overrides(cfg, list(resolved["overrides"]))
    return cfg, resolved


def feature_config_key(cfg: Any) -> str:
    payload = {
        "features": OmegaConf.to_container(cfg.features, resolve=True),
        "timezones": {
            "data.tz": str(cfg.data.tz),
            "session.tz": str(cfg.session.tz),
            "strategy.session.timezone": str(cfg.strategy.session.timezone),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-file", type=Path, default=ROOT / "scripts" / "variant_list.txt")
    parser.add_argument("--cutoffs", nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON report path, relative to the repository when not absolute",
    )
    args = parser.parse_args()

    variant_names = [
        line.strip()
        for line in args.variant_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not variant_names:
        raise ValueError("variant list is empty")
    variant_configs = [(name, *config_for_variant(name)) for name in variant_names]
    feature_groups: dict[str, list[tuple[str, Any, dict[str, Any]]]] = {}
    for name, variant_cfg, resolved in variant_configs:
        feature_groups.setdefault(feature_config_key(variant_cfg), []).append(
            (name, variant_cfg, resolved)
        )
    cfg = variant_configs[0][1]
    primary_symbol = str(cfg.data.primary)
    secondary_symbol = str(cfg.data.secondary)
    primary, primary_path = load_market_bars(cfg, primary_symbol)
    secondary, secondary_path = load_market_bars(cfg, secondary_symbol)
    train_end, test_start = get_split_config(cfg)
    primary_index = cast(pd.DatetimeIndex, primary.index)
    train_mask = make_train_mask(primary_index, train_end)

    group_results: list[dict[str, Any]] = []
    for group_key, group in feature_groups.items():
        representative, group_cfg, resolved = group[0]
        feature_hash = feature_cache_content_hash(
            group_cfg, primary, train_mask=train_mask, secondary=secondary
        )
        print(
            f"building profile={representative} variants={[item[0] for item in group]} "
            f"rows={len(primary)} features={FEATURE_CACHE_VERSION} hash={feature_hash}",
            flush=True,
        )
        reference = build_features(
            primary,
            secondary=secondary,
            cfg=group_cfg,
            train_mask=train_mask,
            force=True,
        )
        if not reference.index.equals(primary.index):
            raise RuntimeError(f"full feature reference for {representative} is misaligned")

        cutoff_results: list[dict[str, Any]] = []
        for cutoff_text in args.cutoffs:
            cutoff = pd.Timestamp(cutoff_text)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize(primary_index.tz)
            else:
                cutoff = cutoff.tz_convert(primary_index.tz)
            prefix_primary = primary.loc[:cutoff]
            prefix_secondary = secondary.loc[:cutoff]
            if prefix_primary.empty:
                raise ValueError(f"cutoff precedes source data: {cutoff}")
            prefix_index = cast(pd.DatetimeIndex, prefix_primary.index)
            prefix_mask = make_train_mask(prefix_index, train_end)
            print(
                f"rebuilding profile={representative} prefix cutoff={cutoff.isoformat()} "
                f"rows={len(prefix_primary)}",
                flush=True,
            )
            prefix = build_features(
                prefix_primary,
                secondary=prefix_secondary,
                cfg=group_cfg,
                train_mask=prefix_mask,
                force=True,
            )
            cutoff_results.append(compare_prefix(reference, prefix, cutoff))
            del prefix, prefix_primary, prefix_secondary, prefix_mask
        group_results.append(
            {
                "profile_id": group_key,
                "representative_variant": representative,
                "variants": [item[0] for item in group],
                "base_config": resolved["base_config"],
                "feature_profile": resolved["feature_profile"],
                "feature_content_hash": feature_hash,
                "feature_columns": [str(column) for column in reference.columns],
                "cutoffs": cutoff_results,
                "summary": {
                    "passed": sum(bool(item["passed"]) for item in cutoff_results),
                    "failed": sum(not bool(item["passed"]) for item in cutoff_results),
                    "cutoffs": len(cutoff_results),
                },
            }
        )
        del reference

    feature_sources = sorted(
        path
        for group in (ROOT / "quant_rl" / "features", ROOT / "quant_rl" / "data")
        for path in group.glob("*.py")
        if path.is_file()
    )
    feature_sources.extend(
        path
        for path in (
            ROOT / "quant_rl" / "config" / "default.yaml",
            ROOT / "quant_rl" / "config" / "__init__.py",
            ROOT / "quant_rl" / "train" / "variant_resolver.py",
            ROOT / "config" / "experiments.yaml",
            ROOT / "config" / "2_po3_ifvg.yaml",
            ROOT / "config" / "3_distribution.yaml",
            ROOT / "config" / "features" / "full_po3_mtf.yaml",
            args.variant_file.resolve(),
        )
        if path.is_file()
    )
    feature_sources = sorted(set(feature_sources))
    all_cutoffs = [item for group in group_results for item in group["cutoffs"]]
    config_paths = sorted(
        {str((ROOT / str(item[2]["base_config"])).relative_to(ROOT)) for item in variant_configs}
    )
    report = {
        "audit": "real_market_feature_prefix_invariance",
        "base_configs": config_paths,
        "resolved_variants": [name for name, _, _ in variant_configs],
        "feature_version": FEATURE_CACHE_VERSION,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "omegaconf")
            },
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
        "profile_groups": group_results,
        "rows": len(primary),
        "start": primary.index[0].isoformat(),
        "end": primary.index[-1].isoformat(),
        "train_end": train_end,
        "test_start": test_start,
        "input_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (primary_path, secondary_path)
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in feature_sources
        },
        "audit_script_sha256": AUDIT_SCRIPT_SHA256,
        "cutoffs": all_cutoffs,
        "summary": {
            "passed": sum(bool(item["passed"]) for item in all_cutoffs),
            "failed": sum(not bool(item["passed"]) for item in all_cutoffs),
            "cutoffs": len(all_cutoffs),
            "feature_profiles": len(group_results),
        },
        "scope_note": (
            "Prefix invariance at the listed real-market cutoffs; it tests whether later bars alter "
            "feature values up to each cutoff. It does not by itself prove every execution, label, "
            "or model-selection path is leakage-free."
        ),
    }
    output = args.out if args.out.is_absolute() else ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={output} summary={report['summary']}", flush=True)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
