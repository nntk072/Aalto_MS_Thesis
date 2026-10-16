#!/usr/bin/env python3
"""Check fold-scoped feature fitting against real campaign market data.

For one early purged walk-forward fold per feature profile, poison only the
validation-period primary bars and verify that fitted training features remain
unchanged. Temporary feature caches are written under node-local /tmp and
removed automatically; campaign caches are read-only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from quant_rl.data.clean import clean
from quant_rl.data.loader import load_bars
from quant_rl.data.session import add_session_id, add_session_labels
from quant_rl.data.split import get_split_config, make_train_mask
from quant_rl.envs.strategies import BaselineStrategy, DistributionStrategy, PO3IFVGStrategy
from quant_rl.evaluation.walkforward import purged_walk_forward
from quant_rl.features.build import build_walkforward_fold_features, select_obs_columns

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
CAMPAIGN = ROOT / "outputs/gpu_campaigns" / CAMPAIGN_ID
RESOLUTION_REPORT = ROOT / "docs/thesis/g1_campaign_feature_cache_resolution_2026-10-16.json"
OUTPUT = ROOT / "docs/thesis/g1_real_market_fold_fit_scope_2026-10-16.json"
VARIANTS = ("ladder_a0_flat", "ladder_a1_sl", "ladder_a0_flat_po3", "ladder_a0_flat_dist")
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


def frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def max_numeric_delta(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if not left.index.equals(right.index) or list(left.columns) != list(right.columns):
        raise RuntimeError("compared fold feature frames changed index or schema")
    a = left.to_numpy(dtype=np.float64)
    b = right.to_numpy(dtype=np.float64)
    both_finite = np.isfinite(a) & np.isfinite(b)
    if not both_finite.any():
        return 0.0
    return float(np.max(np.abs(a[both_finite] - b[both_finite])))


def train_difference_detail(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    a = left.to_numpy(dtype=np.float64)
    b = right.to_numpy(dtype=np.float64)
    equal = (a == b) | (np.isnan(a) & np.isnan(b))
    changed = ~equal
    columns = list(left.columns)
    changed_columns = [columns[i] for i in np.flatnonzero(changed.any(axis=0))]
    rows: list[dict[str, Any]] = []
    for row_index in np.flatnonzero(changed.any(axis=1))[:8]:
        rows.append(
            {
                "timestamp": str(left.index[row_index]),
                "columns": [columns[i] for i in np.flatnonzero(changed[row_index])[:12]],
            }
        )
    return {
        "changed_column_count": len(changed_columns),
        "changed_columns": changed_columns,
        "first_changed_rows": rows,
    }


def main() -> int:
    resolution = json.loads(RESOLUTION_REPORT.read_text(encoding="utf-8"))
    if (
        resolution["campaign_fingerprint"]
        != json.loads((CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))["fingerprint"]
    ):
        raise RuntimeError("cache-resolution report and active campaign fingerprints differ")
    resolution_by_profile: dict[str, dict[str, Any]] = {}
    for row in resolution["results"]:
        resolution_by_profile.setdefault(row["matched_parity_profile"], row)
    config_by_variant = {
        variant: ROOT / resolution_by_profile[variant]["config_path"] for variant in VARIANTS
    }

    configs: dict[str, DictConfig] = {}
    for variant in VARIANTS:
        cfg = OmegaConf.load(config_by_variant[variant])
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"saved run config is not a mapping: {variant}")
        configs[variant] = cfg

    base = configs[VARIANTS[0]]
    primary_symbol = str(base.data.primary)
    secondary_symbol = str(base.data.secondary)
    raw_dir = ROOT / str(base.data.raw_dir)
    primary_path = raw_dir / str(base.data.m1_files[primary_symbol])
    secondary_path = raw_dir / str(base.data.m1_files[secondary_symbol])
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
    train_end, _ = get_split_config(base)
    full_train = primary.loc[make_train_mask(pd.DatetimeIndex(primary.index), train_end)]
    # Use an early, bounded real-data fold while retaining real M1/HTF context.
    fold_input = full_train.iloc[:60_000]
    splits = list(
        purged_walk_forward(
            len(fold_input),
            n_splits=3,
            test_size=0.3,
            purge_bars=60,
            embargo_bars=20,
        )
    )
    if not splits:
        raise RuntimeError("no purged walk-forward fold was produced")
    split = splits[0]
    fold_train = fold_input.iloc[split.train_idx]
    fold_validation = fold_input.iloc[split.test_idx]
    if fold_train.index.max() >= fold_validation.index.min():
        raise RuntimeError("purge/embargo failed to separate train and validation")

    poisoned_primary = primary.copy()
    for column in ("open", "high", "low", "close"):
        poisoned_primary.loc[fold_validation.index, column] *= 1.2

    results: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="thesis-fold-fit-"))
    try:
        for variant, cfg in configs.items():
            cache_root = temp_root / variant
            original_train, original_validation, original_cache = build_walkforward_fold_features(
                primary,
                secondary,
                cfg,
                fold_train,
                fold_validation,
                cache_root / "original",
                primary_symbol,
            )
            poison_train, poison_validation, poison_cache = build_walkforward_fold_features(
                poisoned_primary,
                secondary,
                cfg,
                fold_train,
                fold_validation,
                cache_root / "poisoned",
                primary_symbol,
            )

            train_delta = max_numeric_delta(original_train, poison_train)
            validation_delta = max_numeric_delta(original_validation, poison_validation)
            train_equal = original_train.equals(poison_train)
            validation_changed = validation_delta > 0.0
            detail = train_difference_detail(original_train, poison_train)
            strategy_name = str(OmegaConf.select(cfg, "strategy.name", default="baseline"))
            strategy = STRATEGIES[strategy_name]()
            observation_columns = select_obs_columns(original_train, strategy.raw_columns).columns

            results.append(
                {
                    "variant": variant,
                    "feature_hash": resolution_by_profile[variant]["feature_hash"],
                    "config_path": str(config_by_variant[variant].relative_to(ROOT)),
                    "config_sha256": sha256_file(config_by_variant[variant]),
                    "fold": int(split.fold),
                    "train_rows": len(fold_train),
                    "validation_rows": len(fold_validation),
                    "purge_bars": 60,
                    "embargo_bars": 20,
                    "train_last": str(fold_train.index.max()),
                    "validation_first": str(fold_validation.index.min()),
                    "train_features_identical_after_validation_poison": train_equal,
                    "train_changed_feature_detail": detail,
                    "train_max_abs_delta": train_delta,
                    "validation_max_abs_delta": validation_delta,
                    "validation_features_changed": validation_changed,
                    "changed_features_model_visible": {
                        column: column in observation_columns
                        for column in detail["changed_columns"]
                    },
                    "passed": train_equal and validation_changed,
                    "train_feature_sha256_original": frame_digest(original_train),
                    "train_feature_sha256_poisoned": frame_digest(poison_train),
                    "validation_feature_sha256_original": frame_digest(original_validation),
                    "validation_feature_sha256_poisoned": frame_digest(poison_validation),
                    "temporary_cache_keys": [original_cache.name, poison_cache.name],
                }
            )
            print(
                f"{'PASS' if train_equal and validation_changed else 'FAIL'} {variant}: "
                f"train_delta={train_delta:.3g}, "
                f"validation_delta={validation_delta:.3g}",
                flush=True,
            )
    finally:
        import shutil

        shutil.rmtree(temp_root)

    campaign = json.loads((CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))
    failed_profiles = {item["variant"] for item in results if not item["passed"]}
    affected_campaign_runs = [
        row for row in resolution["results"] if row["matched_parity_profile"] in failed_profiles
    ]
    changed_columns = sorted(
        {
            column
            for item in results
            for column in item["train_changed_feature_detail"]["changed_columns"]
        }
    )
    source_paths = (
        ROOT / "quant_rl/features/build.py",
        ROOT / "quant_rl/features/normalize.py",
        ROOT / "quant_rl/features/session_ohlc.py",
        ROOT / "quant_rl/evaluation/walkforward.py",
        ROOT / "quant_rl/data/clean.py",
        ROOT / "quant_rl/data/session.py",
        ROOT / "quant_rl/data/split.py",
    )
    if failed_profiles:
        scope_note = (
            f"FAILED: only {sum(bool(item['passed']) for item in results)}/{len(results)} "
            "profile perturbation checks passed. Validation OHLC changes altered training "
            f"features {changed_columns} for profiles {sorted(failed_profiles)}; "
            f"the mapped saved campaign configs include {len(affected_campaign_runs)}/18 runs. "
            "This is a real-market causality failure for the tested slice, not a strategy "
            "result. The active campaign source was not changed."
        )
    else:
        scope_note = (
            "Bounded empirical check for one early purged fold and four feature profiles. "
            "Validation OHLC was perturbed by 20 percent; training-fold feature outputs "
            "remained exactly identical while validation features changed. This supports "
            "fold-local normalization isolation for the tested source/config/data slice. "
            "It is not a full five-fold training/evaluation run, exhaustive leakage proof, "
            "or strategy result."
        )
    report = {
        "audit": "real_market_purged_fold_fit_scope_poison_test",
        "scope_note": scope_note,
        "source_diagnosis": (
            "The only changed training columns in failing profiles are prev_ny_high and "
            "prev_ny_low. Source inspection finds prev_ny_high_low computes each session's "
            "final extreme and shifts by one row within that same session group, exposing "
            "the future final NY-session levels after the first bar of that session."
        ),
        "affected_campaign_runs": [row["variant"] for row in affected_campaign_runs],
        "affected_campaign_run_count": len(affected_campaign_runs),
        "failed_profiles": sorted(failed_profiles),
        "campaign_id": CAMPAIGN_ID,
        "campaign_fingerprint": campaign["fingerprint"],
        "campaign_git_sha": campaign["git_sha"],
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "slurm_job_id": campaign.get("job_id"),
        "raw_inputs_sha256": {
            str(primary_path.relative_to(ROOT)): sha256_file(primary_path),
            str(secondary_path.relative_to(ROOT)): sha256_file(secondary_path),
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        "cache_resolution_report_sha256": sha256_file(RESOLUTION_REPORT),
        "fold_rows_available": len(fold_input),
        "results": results,
        "all_profiles_pass": all(item["passed"] for item in results),
        "temporary_cache_root_removed": not temp_root.exists(),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    report["audit_script_sha256"] = sha256_file(Path(__file__).resolve())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed_count = sum(bool(item["passed"]) for item in results)
    print(f"AUDIT COMPLETE: {passed_count}/{len(configs)} profiles pass; report={OUTPUT}")
    return 0 if report["all_profiles_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
