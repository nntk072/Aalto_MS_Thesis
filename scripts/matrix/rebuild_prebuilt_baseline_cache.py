#!/usr/bin/env python3
"""Back up and rebuild the named baseline cache for the resolved A0 config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, make_train_mask
from quant_rl.features.build import build_features, feature_cache_content_hash
from quant_rl.train.train_rl import _load_merged_config
from quant_rl.train.variant_resolver import resolve_variant

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebuild(backup_dir: Path, report_path: Path) -> dict[str, Any]:
    resolved = resolve_variant("ladder_a0_flat", steps=100_000, seeds=[42])
    cfg = _load_merged_config(
        argparse.Namespace(
            config=resolved["base_config"],
            strategy=resolved["strategy"],
            overrides=resolved["overrides"],
            variant=None,
        )
    )
    cache_value = resolved.get("overrides", [])
    cache_override = next(
        (
            value.split("=", 1)[1]
            for value in cache_value
            if value.startswith("data.prebuilt_feature_cache=")
        ),
        None,
    )
    if cache_override is None:
        raise RuntimeError("resolved A0 does not name its prebuilt baseline cache")
    cache_path = Path(cache_override)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache_path = cache_path.resolve()
    sidecar = Path(f"{cache_path}.hash")
    for source in (cache_path, sidecar):
        if not source.is_file():
            raise FileNotFoundError(source)
    old_hash = sidecar.read_text(encoding="utf-8").strip()
    if backup_dir.exists():
        if not all((backup_dir / source.name).is_file() for source in (cache_path, sidecar)):
            raise FileExistsError(f"incomplete existing cache backup: {backup_dir}")
    else:
        backup_dir.mkdir(parents=True)
        for source in (cache_path, sidecar):
            shutil.copy2(source, backup_dir / source.name)

    started = time.monotonic()
    market = run_pipeline(cfg)
    primary = market[str(cfg.data.primary)]["M1"]
    secondary = market.get(str(cfg.data.secondary), {}).get("M1")
    train_end, _ = get_split_config(cfg)
    train_mask = make_train_mask(cast(pd.DatetimeIndex, primary.index), train_end)
    expected_hash = feature_cache_content_hash(
        cfg,
        primary,
        train_mask=train_mask,
        secondary=secondary,
    )
    actual_hash = old_hash
    if actual_hash != expected_hash:
        with tempfile.TemporaryDirectory(
            prefix="baseline-cache-rebuild-", dir=cache_path.parent
        ) as tmp:
            temp_cache = Path(tmp) / cache_path.name
            features = build_features(
                primary,
                secondary=secondary,
                cfg=cfg,
                train_mask=train_mask,
                cache_path=temp_cache,
                force=True,
            )
            temp_sidecar = Path(f"{temp_cache}.hash")
            actual_hash = temp_sidecar.read_text(encoding="utf-8").strip()
            if actual_hash != expected_hash or not temp_cache.is_file():
                raise RuntimeError(
                    f"rebuilt cache hash mismatch: expected {expected_hash}, found {actual_hash}"
                )
            os.replace(temp_cache, cache_path)
            os.replace(temp_sidecar, sidecar)
    else:
        features = pd.read_parquet(cache_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"rebuilt cache hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "variant": "ladder_a0_flat",
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "effective_semantics": {
            "strategy": resolved["strategy"],
            "strategy_actions": False,
            "feature_profile": resolved["feature_profile"],
            "action_schema": "legacy Discrete(20)",
            "reward_mode": resolved["reward_mode"],
        },
        "market": {
            "primary_symbol": str(cfg.data.primary),
            "secondary_symbol": str(cfg.data.secondary),
            "rows": int(len(primary)),
            "train_end": train_end,
            "train_rows": int(train_mask.sum()),
        },
        "cache": {
            "path": cache_path.relative_to(ROOT).as_posix(),
            "previous_hash": old_hash,
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
            "matches_effective_config": actual_hash == expected_hash,
            "previous_cache_backup": backup_dir.relative_to(ROOT).as_posix(),
            "previous_cache_sha256": _sha256(backup_dir / cache_path.name),
            "previous_sidecar_sha256": _sha256(backup_dir / sidecar.name),
            "rebuilt_cache_sha256": _sha256(cache_path),
            "rebuilt_sidecar_sha256": _sha256(sidecar),
            "feature_rows": int(len(features)),
            "feature_columns": int(features.shape[1]),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = rebuild(args.backup_dir.resolve(), args.report.resolve())
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
