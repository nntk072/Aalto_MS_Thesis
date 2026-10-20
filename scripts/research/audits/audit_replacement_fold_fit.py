#!/usr/bin/env python3
"""Poison validation bars to test train-fold fitting on replacement data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

ROOT = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis")
sys.path.insert(0, str(ROOT))
from configuration import apply_overrides, load_config  # noqa: E402
from features import (  # noqa: E402
    FEATURE_CACHE_VERSION,
    build_walkforward_fold_features,
    select_obs_columns,
)
from market_data.loading.bars import load_bars  # noqa: E402
from market_data.splits.definitions import get_split_config, make_train_mask  # noqa: E402
from market_data.splits.walkforward import purged_walk_forward  # noqa: E402
from market_data.transforms.cleaning import clean  # noqa: E402
from market_data.transforms.sessions import add_session_id, add_session_labels  # noqa: E402
from trading.strategies import (  # noqa: E402
    BaselineStrategy,
    DistributionStrategy,
    PO3IFVGStrategy,
)
from training.configuration.variant_resolution import resolve_variant  # noqa: E402

VARIANT_FILE = ROOT / "config/campaigns/variant_list.txt"
OUT = ROOT / "outputs/data_refresh_20261017/replacement_fold_fit.json"
STRATEGIES = {
    "baseline": BaselineStrategy,
    "po3_ifvg": PO3IFVGStrategy,
    "distribution": DistributionStrategy,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_report_path(report_path: Path, root: Path = ROOT) -> tuple[Path, Path]:
    """Resolve a report path and require it to stay inside the repository."""
    root = root.resolve()
    resolved = (report_path if report_path.is_absolute() else root / report_path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"report must be under repository root: {resolved}") from exc
    return resolved, relative


def config_for_variant(name: str) -> tuple[DictConfig, dict[str, Any]]:
    resolved = resolve_variant(name)
    cfg: DictConfig = load_config(config_path=ROOT / str(resolved["base_config"]))
    strategy_path = {
        "po3_ifvg": ROOT / "config/2_po3_ifvg.yaml",
        "distribution": ROOT / "config/3_distribution.yaml",
    }.get(str(resolved["strategy"]))
    if strategy_path is not None:
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(strategy_path)))
    return apply_overrides(cfg, list(resolved["overrides"])), resolved


def profile_key(cfg: DictConfig) -> str:
    value = {
        "features": OmegaConf.to_container(cfg.features, resolve=True),
        "timezones": {
            key: str(OmegaConf.select(cfg, key))
            for key in (
                "data.tz",
                "data.timestamp_clock",
                "session.tz",
                "strategy.session.timezone",
            )
        },
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def digest_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def main(report_path: Path = OUT) -> None:
    report_path, report_relative_path = resolve_report_path(report_path)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    variants = [
        line.strip()
        for line in VARIANT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    configs = [(name, *config_for_variant(name)) for name in variants]
    groups: dict[str, list[tuple[str, DictConfig, dict[str, Any]]]] = {}
    for item in configs:
        groups.setdefault(profile_key(item[1]), []).append(item)
    base = configs[0][1]
    primary_symbol, secondary_symbol = str(base.data.primary), str(base.data.secondary)
    raw_dir = ROOT / str(base.data.raw_dir)
    primary_path = raw_dir / str(base.data.m1_files[primary_symbol])
    secondary_path = raw_dir / str(base.data.m1_files[secondary_symbol])

    def load_symbol(path: Path) -> pd.DataFrame:
        bars = clean(load_bars(path), tz=str(base.data.tz))
        bars = add_session_labels(bars, tz=str(base.data.tz), ny_end=str(base.session.end))
        return add_session_id(bars)

    primary, secondary = load_symbol(primary_path), load_symbol(secondary_path)
    train_end, test_start = get_split_config(base)
    train = primary.loc[make_train_mask(pd.DatetimeIndex(primary.index), train_end)]
    folds = list(
        purged_walk_forward(len(train), n_splits=5, test_size=0.2, purge_bars=60, embargo_bars=20)
    )
    if len(folds) != 5:
        raise RuntimeError(f"expected five purged folds, got {len(folds)}")

    source_paths = [
        ROOT / "features/pipeline/builder.py",
        ROOT / "features/pipeline/normalization.py",
        ROOT / "features/context/session_levels.py",
        ROOT / "market_data/splits/walkforward.py",
        ROOT / "market_data/transforms/cleaning.py",
        ROOT / "market_data/transforms/sessions.py",
        ROOT / "market_data/splits/definitions.py",
        ROOT / "training/configuration/variant_resolution.py",
        VARIANT_FILE,
        ROOT / "config/base/default.yaml",
        ROOT / "config/2_po3_ifvg.yaml",
        ROOT / "config/3_distribution.yaml",
    ]
    source_paths.extend(
        ROOT / str(resolved["base_config"])
        for _, _, resolved in configs
        if (ROOT / str(resolved["base_config"])).is_file()
    )
    results: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="replacement-fold-fit-", dir="/tmp"))
    try:
        for group in groups.values():
            name, cfg, resolved = group[0]
            strategy_name = str(OmegaConf.select(cfg, "strategy.name", default="baseline"))
            strategy = STRATEGIES[strategy_name]()
            for fold in folds:
                fold_train, fold_validation = train.iloc[fold.train_idx], train.iloc[fold.test_idx]
                if fold_train.index.max() >= fold_validation.index.min():
                    raise RuntimeError(f"fold {fold.fold} train/validation overlap")
                poisoned_primary, poisoned_secondary = primary.copy(), secondary.copy()
                poisoned_rows: dict[str, int] = {}
                for symbol, bars in (
                    (primary_symbol, poisoned_primary),
                    (secondary_symbol, poisoned_secondary),
                ):
                    overlap = bars.index.intersection(fold_validation.index)
                    if overlap.empty:
                        raise RuntimeError(f"fold {fold.fold} has no validation bars for {symbol}")
                    for column in ("open", "high", "low", "close"):
                        bars.loc[overlap, column] *= 1.2
                    poisoned_rows[symbol] = len(overlap)
                cache_root = temp_root / name / f"fold-{fold.fold}"
                original_train, original_valid, _ = build_walkforward_fold_features(
                    primary,
                    secondary,
                    cfg,
                    fold_train,
                    fold_validation,
                    cache_root / "original",
                    primary_symbol,
                )
                poison_train, poison_valid, _ = build_walkforward_fold_features(
                    poisoned_primary,
                    poisoned_secondary,
                    cfg,
                    fold_train,
                    fold_validation,
                    cache_root / "poisoned",
                    primary_symbol,
                )
                changed_columns = [
                    str(column)
                    for column in original_valid.columns
                    if not original_valid[column].equals(poison_valid[column])
                ]
                visible_columns = set(
                    select_obs_columns(original_train, strategy.raw_columns).columns
                )
                train_equal = original_train.equals(poison_train)
                validation_changed = not original_valid.equals(poison_valid)
                visible_change = any(column in visible_columns for column in changed_columns)
                passed = train_equal and validation_changed and visible_change
                results.append(
                    {
                        "profile": name,
                        "group_variants": [row[0] for row in group],
                        "strategy": strategy_name,
                        "fold": int(fold.fold),
                        "train_rows": len(fold_train),
                        "validation_rows": len(fold_validation),
                        "purge_bars": 60,
                        "embargo_bars": 20,
                        "train_last": str(fold_train.index.max()),
                        "validation_first": str(fold_validation.index.min()),
                        "validation_last": str(fold_validation.index.max()),
                        "poisoned_validation_rows": poisoned_rows,
                        "train_features_identical_after_validation_poison": train_equal,
                        "validation_features_changed": validation_changed,
                        "changed_validation_features": changed_columns,
                        "changed_validation_features_model_visible": [
                            c for c in changed_columns if c in visible_columns
                        ],
                        "passed": passed,
                        "train_sha256_original": digest_frame(original_train),
                        "train_sha256_poisoned": digest_frame(poison_train),
                        "validation_sha256_original": digest_frame(original_valid),
                        "validation_sha256_poisoned": digest_frame(poison_valid),
                    }
                )
                print(f"{'PASS' if passed else 'FAIL'} {name} fold={fold.fold}", flush=True)
                del (
                    poisoned_primary,
                    poisoned_secondary,
                    original_train,
                    original_valid,
                    poison_train,
                    poison_valid,
                )
    finally:
        shutil.rmtree(temp_root)

    result = {
        "schema": "aalto.replacement_fold_fit_poison_audit.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "train_end_inclusive": str(train_end),
            "test_start_inclusive": str(test_start),
            "primary_rows": len(primary),
            "train_rows": len(train),
            "raw_input_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in (primary_path, secondary_path)
            },
        },
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "report_path": str(report_relative_path),
        "variants": variants,
        "profile_count": len(groups),
        "profiles": [
            {
                "representative_variant": group[0][0],
                "variants": [row[0] for row in group],
                "base_config": str(group[0][2]["base_config"]),
                "strategy": str(group[0][2]["strategy"]),
                "effective_config_sha256": hashlib.sha256(
                    json.dumps(
                        OmegaConf.to_container(group[0][1], resolve=True),
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest(),
            }
            for group in groups.values()
        ],
        "fold_count": len(folds),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "all_profiles_pass": len(results) == len(groups) * len(folds)
        and all(row["passed"] for row in results),
        "results": results,
        "scope_limitations": [
            "Tests fold-local fit invariance under 20 percent OHLC poisoning for each default expanding purged training fold and representative configured feature profile.",
            "Does not prove every feature or execution path causal, validate calendar/DST, establish data rights, or provide strategy-performance evidence.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"AUDIT COMPLETE: {sum(bool(row['passed']) for row in results)}/{len(results)}; report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=OUT,
        help="new JSON report path (defaults to the historical report path; never overwritten)",
    )
    main(parser.parse_args().report)
