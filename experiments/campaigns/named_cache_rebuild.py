#!/usr/bin/env python3
"""Stage, verify, and transactionally publish the thesis variant caches.

Run only after the training source boundary. The script uses the same variant
resolution and config merge path as ``train_rl``. It builds into a temporary
directory, verifies content hashes and parquet round-trip contents, preserves the
existing cache files in a required backup directory, and rolls back published
files if any replacement fails. The immutable report contains resolved configs,
raw-data hashes, loaded project-source hashes, runtime/dependency identity, and
tracked/untracked worktree state needed to interpret a local rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from configuration.paths import repository_root
from features import (
    FEATURE_CACHE_VERSION,
    build_features,
    feature_cache_content_hash,
)
from market_data.loading.pipeline import run_pipeline
from market_data.splits.definitions import get_split_config, make_train_mask
from training.configuration.variant_resolution import resolve_variant

ROOT = repository_root(__file__)
PROJECT_SOURCE_ROOTS = frozenset(
    {
        "configuration",
        "artifacts",
        "market_data",
        "features",
        "trading",
        "simulation",
        "models",
        "training",
        "evaluation",
        "experiments",
        "live",
        "integrations",
        "compute",
        "orchestra",
    }
)
VARIANTS = (
    "ladder_a0_flat",
    "ladder_a1_sl",
    "ladder_a0_flat_po3",
    "ladder_a0_flat_dist",
)
CAMPAIGN_VARIANTS_PATH = ROOT / "config" / "campaigns" / "variant_list.txt"
REQUIRED_EXISTING_CACHE_FILES = {
    name
    for cache_name in (
        "baseline.parquet",
        "idea1_po3_ifvg.parquet",
        "idea2_distribution.parquet",
    )
    for name in (cache_name, f"{cache_name}.hash")
}
STRATEGY_CONFIGS = {
    "po3_ifvg": "config/2_po3_ifvg.yaml",
    "distribution": "config/3_distribution.yaml",
}


def _historical_named_cache_path(name: str, resolved: dict[str, Any]) -> str:
    """Map a historical thesis variant to the legacy named feature cache."""
    strategy = str(resolved["strategy"])
    if strategy == "po3_ifvg":
        filename = "idea1_po3_ifvg.parquet"
    elif strategy == "distribution":
        filename = "idea2_distribution.parquet"
    elif bool(resolved.get("strategy_actions", False)):
        filename = "baseline_strategy_actions.parquet"
    else:
        filename = "baseline.parquet"
    return f"data/prebuilt_feature_cache/{filename}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_config_record(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "value": resolved,
    }


def _loaded_project_source_hashes() -> dict[str, str]:
    """Hash project Python modules loaded by this cache-build process."""
    paths: set[Path] = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if not module_path:
            continue
        path = Path(module_path).resolve()
        try:
            relative = path.relative_to(ROOT.resolve())
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in PROJECT_SOURCE_ROOTS and path.suffix == ".py":
            paths.add(path)
    return {
        path.relative_to(ROOT.resolve()).as_posix(): sha256_file(path) for path in sorted(paths)
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "pyarrow", "omegaconf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_fresh_market(
    cfg: DictConfig,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild bars from raw inputs without reading or replacing shared caches."""
    raw_dir = Path(str(cfg.data.raw_dir))
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    raw_dir = raw_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="named-cache-raw-bars-") as cache_dir:
        isolated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if not isinstance(isolated_cfg, DictConfig):
            raise TypeError("isolated data-pipeline configuration must be a mapping")
        isolated_cfg.data.raw_dir = str(raw_dir)
        isolated_cfg.data.cache_dir = cache_dir
        return run_pipeline(isolated_cfg, force=True)


def load_variant(name: str) -> tuple[dict[str, Any], DictConfig]:
    from training.configuration.cli import _load_merged_config

    resolved = resolve_variant(name, steps=100_000, seeds=[42])
    resolved = {
        **resolved,
        "overrides": [
            *resolved["overrides"],
            f"data.prebuilt_feature_cache={_historical_named_cache_path(name, resolved)}",
        ],
    }
    cfg = _load_merged_config(
        argparse.Namespace(
            config=resolved["base_config"],
            strategy=resolved["strategy"],
            overrides=resolved["overrides"],
            variant=None,
            quality_variant=None,
        )
    )
    explicit_cache = OmegaConf.select(cfg, "data.prebuilt_feature_cache")
    if not explicit_cache:
        raise RuntimeError(f"{name} has no explicit prebuilt feature-cache path")
    cache_path = Path(str(explicit_cache))
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    expected_name = {
        "ladder_a0_flat": "baseline.parquet",
        "ladder_a1_sl": "baseline_strategy_actions.parquet",
        "ladder_a0_flat_po3": "idea1_po3_ifvg.parquet",
        "ladder_a0_flat_dist": "idea2_distribution.parquet",
    }[name]
    if cache_path.resolve() != (ROOT / "data" / "prebuilt_feature_cache" / expected_name).resolve():
        raise RuntimeError(f"unexpected named cache for {name}: {cache_path}")
    return resolved, cfg


def _raw_input_paths(cfg: DictConfig) -> dict[str, Path]:
    raw_dir = Path(str(cfg.data.raw_dir))
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    files = cfg.data.m1_files
    symbols = (str(cfg.data.primary), str(cfg.data.secondary))
    return {symbol: (raw_dir / str(files[symbol])).resolve() for symbol in symbols}


def _cache_config_signature(cfg: DictConfig) -> str:
    payload = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("resolved data configuration must be a mapping")
    payload.pop("prebuilt_feature_cache", None)
    return json.dumps(payload, sort_keys=True, default=str)


def _requires_htf_sweep_audit(cfg: DictConfig) -> bool:
    return str(OmegaConf.select(cfg, "features.po3_reference_mode") or "m1_legacy") == (
        "htf_confirmed"
    )


def _validate_staged_sweep_audit(cache_path: Path, cfg: DictConfig) -> Path | None:
    audit_path = Path(f"{cache_path}.htf_sweeps.parquet")
    if _requires_htf_sweep_audit(cfg) and not audit_path.is_file():
        raise RuntimeError(
            f"HTF-confirmed feature profile omitted its required sweep-audit parquet: {audit_path}"
        )
    return audit_path if audit_path.is_file() else None


def plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in VARIANTS:
        resolved, cfg = load_variant(name)
        cache = Path(str(cfg.data.prebuilt_feature_cache))
        if not cache.is_absolute():
            cache = ROOT / cache
        config_paths = {
            "config/base/default.yaml",
            str(resolved["base_config"]),
            "config/experiments.yaml",
        }
        strategy_config = STRATEGY_CONFIGS.get(str(resolved["strategy"]))
        if strategy_config is not None:
            config_paths.add(strategy_config)
        if (ROOT / "config/features/pd_context.yaml").is_file():
            config_paths.add("config/features/pd_context.yaml")
        rows.append(
            {
                "variant": name,
                "strategy": resolved["strategy"],
                "feature_profile": resolved["feature_profile"],
                "base_config": resolved["base_config"],
                "resolved_overrides": resolved["overrides"],
                "strategy_config": strategy_config,
                "config_sha256": {path: sha256_file(ROOT / path) for path in sorted(config_paths)},
                "effective_config": _resolved_config_record(cfg),
                "cache_path": str(cache.resolve()),
                "cache_path_exists": cache.is_file(),
                "sidecar_exists": Path(f"{cache}.hash").is_file(),
                "requires_htf_sweep_audit": _requires_htf_sweep_audit(cfg),
                "htf_sweep_audit_exists": Path(f"{cache}.htf_sweeps.parquet").is_file(),
                "cache_config_signature": _cache_config_signature(cfg),
            }
        )
    cache_paths = {row["cache_path"] for row in rows}
    if len(cache_paths) != len(VARIANTS):
        raise RuntimeError("feature-profile representatives must resolve to unique cache files")
    relative_cache_paths = {
        Path(path).resolve().relative_to(ROOT.resolve()).as_posix() for path in cache_paths
    }
    names = [
        line.strip() for line in CAMPAIGN_VARIANTS_PATH.read_text().splitlines() if line.strip()
    ]
    if len(names) != 18 or len(names) != len(set(names)):
        raise RuntimeError("the thesis campaign variant list must contain 18 unique variants")
    bindings: dict[str, str] = {}
    for name in names:
        resolved = resolve_variant(name, steps=100_000, seeds=[42])
        relative_cache_path = _historical_named_cache_path(name, resolved)
        if relative_cache_path not in relative_cache_paths:
            raise RuntimeError(f"{name} maps outside the four verified feature profiles")
        bindings[name] = relative_cache_path
    if set(bindings.values()) != relative_cache_paths:
        raise RuntimeError("not every prebuilt feature profile is referenced by the campaign")
    rows[0]["campaign_variant_cache_bindings"] = bindings
    return rows


def _publish_staged_files(
    staged_by_target: dict[Path, Path],
    backup_dir: Path,
    report_tmp: Path,
    report_path: Path,
    validate_published: Callable[[], None],
) -> None:
    """Replace the staged artifacts and restore old targets on any failure."""
    published: list[Path] = []
    try:
        for target, staged in staged_by_target.items():
            os.replace(staged, target)
            published.append(target)
        validate_published()
        os.replace(report_tmp, report_path)
    except Exception:
        for target in reversed(published):
            saved = backup_dir / target.name
            if saved.is_file():
                shutil.copy2(saved, target)
            else:
                target.unlink(missing_ok=True)
        raise


def rebuild(backup_dir: Path, report_path: Path) -> dict[str, Any]:
    rows = plan()
    plan_by_variant = {str(row["variant"]): row for row in rows}
    report_path = report_path.resolve()
    try:
        report_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("cache rebuild report path must stay inside the repository") from exc
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    cache_paths = [Path(row["cache_path"]) for row in rows]
    if len({path.parent for path in cache_paths}) != 1:
        raise RuntimeError("transactional publication requires all caches in one directory")
    cache_dir = cache_paths[0].parent
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError(f"prebuilt cache directory must be a real directory: {cache_dir}")
    if not backup_dir.is_absolute():
        backup_dir = ROOT / backup_dir
    backup_dir = backup_dir.resolve()
    try:
        backup_dir.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("cache backup directory must stay inside the repository") from exc
    if backup_dir.exists():
        raise FileExistsError(f"refusing to reuse existing backup directory: {backup_dir}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_cfgs = [load_variant(name)[1] for name in VARIANTS]
    signatures = {_cache_config_signature(cfg) for cfg in all_cfgs}
    if len(signatures) != 1:
        raise RuntimeError("named cache variants disagree on raw-data or split configuration")
    base_cfg = all_cfgs[0]
    input_paths = _raw_input_paths(base_cfg)
    for symbol, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"raw M1 input missing for {symbol}: {path}")
    raw_hashes_before = {symbol: sha256_file(path) for symbol, path in input_paths.items()}

    cache_files = [
        artifact
        for cache in cache_paths
        for artifact in (
            cache,
            Path(f"{cache}.hash"),
            Path(f"{cache}.htf_sweeps.parquet"),
        )
    ]
    existing_files = [path for path in cache_files if path.is_file()]
    required_files = [
        artifact for cache in cache_paths for artifact in (cache, Path(f"{cache}.hash"))
    ]
    missing_required = [
        path
        for path in required_files
        if path.name in REQUIRED_EXISTING_CACHE_FILES and not path.is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            f"existing caches must be complete before rebuild: {missing_required}"
        )

    market = _load_fresh_market(base_cfg)
    primary = market[str(base_cfg.data.primary)]["M1"]
    secondary = market.get(str(base_cfg.data.secondary), {}).get("M1")
    primary_timeframe_bars = market[str(base_cfg.data.primary)]
    secondary_timeframe_bars = market.get(str(base_cfg.data.secondary), {})
    train_end, _ = get_split_config(base_cfg)
    train_mask = make_train_mask(cast(pd.DatetimeIndex, primary.index), train_end)

    staged_dir = Path(tempfile.mkdtemp(prefix="named-feature-cache-stage-", dir=cache_dir))
    staged_by_target: dict[Path, Path] = {}
    cache_results: list[dict[str, Any]] = []
    report_tmp: Path | None = None
    try:
        for name, cfg in zip(VARIANTS, all_cfgs, strict=True):
            target = Path(str(cfg.data.prebuilt_feature_cache))
            if not target.is_absolute():
                target = ROOT / target
            target = target.resolve()
            staged_cache = staged_dir / target.name
            expected_hash = feature_cache_content_hash(
                cfg,
                primary,
                train_mask=train_mask,
                secondary=secondary,
                timeframe_bars=primary_timeframe_bars,
                secondary_timeframe_bars=secondary_timeframe_bars,
            )
            features = build_features(
                primary,
                secondary=secondary,
                cfg=cfg,
                train_mask=train_mask,
                cache_path=staged_cache,
                force=True,
                timeframe_bars=primary_timeframe_bars,
                secondary_timeframe_bars=secondary_timeframe_bars,
            )
            staged_hash = Path(f"{staged_cache}.hash")
            if not staged_cache.is_file() or not staged_hash.is_file():
                raise RuntimeError(f"feature builder omitted cache or hash for {name}")
            staged_aux = _validate_staged_sweep_audit(staged_cache, cfg)
            actual_hash = staged_hash.read_text(encoding="utf-8").strip()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{name}: expected feature hash {expected_hash}, got {actual_hash}"
                )
            cached = pd.read_parquet(staged_cache)
            try:
                pd.testing.assert_frame_equal(cached, features, check_exact=True, check_freq=False)
            except AssertionError as exc:
                raise RuntimeError(
                    f"{name}: staged parquet changed feature values or schema"
                ) from exc
            staged_by_target[target] = staged_cache
            staged_by_target[Path(f"{target}.hash")] = staged_hash
            if staged_aux is not None:
                target_aux = Path(f"{target}.htf_sweeps.parquet")
                staged_by_target[target_aux] = staged_aux
            cache_results.append(
                {
                    "variant": name,
                    "target": target.relative_to(ROOT).as_posix(),
                    "feature_cache_hash": expected_hash,
                    "previous_cache_sha256": sha256_file(target) if target.is_file() else None,
                    "previous_sidecar_sha256": (
                        sha256_file(Path(f"{target}.hash"))
                        if Path(f"{target}.hash").is_file()
                        else None
                    ),
                    "cache_sha256": sha256_file(staged_cache),
                    "sidecar_sha256": sha256_file(staged_hash),
                    "requires_htf_sweep_audit": _requires_htf_sweep_audit(cfg),
                    "htf_sweep_audit_path": (
                        Path(f"{target}.htf_sweeps.parquet").relative_to(ROOT).as_posix()
                        if staged_aux is not None
                        else None
                    ),
                    "previous_htf_sweep_audit_sha256": (
                        sha256_file(Path(f"{target}.htf_sweeps.parquet"))
                        if Path(f"{target}.htf_sweeps.parquet").is_file()
                        else None
                    ),
                    "htf_sweep_audit_sha256": (
                        sha256_file(staged_aux) if staged_aux is not None else None
                    ),
                    "rows": int(len(cached)),
                    "columns": int(cached.shape[1]),
                    "resolved_variant": plan_by_variant[name],
                }
            )
            del cached, features

        raw_hashes_after = {symbol: sha256_file(path) for symbol, path in input_paths.items()}
        if raw_hashes_after != raw_hashes_before:
            raise RuntimeError("raw market inputs changed during cache rebuild")

        source_hashes = _loaded_project_source_hashes()
        repository_status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        )
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--binary", "HEAD"],
        )
        report = {
            "schema_version": 2,
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "packages": _dependency_versions(),
                "dependency_files_sha256": {
                    path: sha256_file(ROOT / path)
                    for path in ("pyproject.toml", "uv.lock")
                    if (ROOT / path).is_file()
                },
            },
            "repository_head": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "repository_dirty": bool(repository_status),
            "repository_status_porcelain": repository_status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "bar_ingestion": {
                "source": "raw_m1_csv",
                "force_rebuild": True,
                "intermediate_cache": "isolated_temporary_directory",
                "shared_bar_cache_modified": False,
            },
            "backup_dir": backup_dir.relative_to(ROOT).as_posix(),
            "raw_m1_sha256": raw_hashes_after,
            "loaded_project_source_sha256": source_hashes,
            "loaded_project_source_count": len(source_hashes),
            "caches": cache_results,
            "campaign_variant_cache_bindings": plan_by_variant[VARIANTS[0]][
                "campaign_variant_cache_bindings"
            ],
        }
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent
        )
        os.close(tmp_fd)
        report_tmp = Path(tmp_name)
        report_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        backup_dir.mkdir(parents=True)
        try:
            for source in existing_files:
                shutil.copy2(source, backup_dir / source.name)
        except Exception:
            shutil.rmtree(backup_dir, ignore_errors=True)
            raise

        def validate_published() -> None:
            for item in cache_results:
                target = ROOT / item["target"]
                actual = Path(f"{target}.hash").read_text(encoding="utf-8").strip()
                if actual != item["feature_cache_hash"]:
                    raise RuntimeError(f"published hash verification failed: {target}")
                audit_path = item["htf_sweep_audit_path"]
                if item["requires_htf_sweep_audit"] and audit_path is None:
                    raise RuntimeError(f"required HTF sweep audit was not published: {target}")
                if audit_path is not None:
                    actual_audit_hash = sha256_file(ROOT / audit_path)
                    if actual_audit_hash != item["htf_sweep_audit_sha256"]:
                        raise RuntimeError(f"published HTF sweep audit hash mismatch: {audit_path}")

        if report_tmp is None:
            raise RuntimeError("cache rebuild evidence report was not staged")
        _publish_staged_files(
            staged_by_target,
            backup_dir,
            report_tmp,
            report_path,
            validate_published,
        )
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
        if report_tmp is not None:
            report_tmp.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="resolve paths without writing")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "caches": plan()}, indent=2))
        return 0
    if args.backup_dir is None or args.report is None:
        parser.error("--backup-dir and --report are required unless --plan-only is used")
    report_path = (args.report if args.report.is_absolute() else ROOT / args.report).resolve()
    backup_dir = (
        args.backup_dir if args.backup_dir.is_absolute() else ROOT / args.backup_dir
    ).resolve()
    result = rebuild(backup_dir, report_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
