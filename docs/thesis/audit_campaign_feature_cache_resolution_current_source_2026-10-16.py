#!/usr/bin/env python3
"""Verify the active 18-run campaign resolves to audited feature caches.

This is a read-only audit. It reads the saved run configs, M1 inputs, existing
cache files, and cache hash sidecars, then writes only its JSON report.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from quant_rl.data.clean import clean
from quant_rl.data.loader import load_bars
from quant_rl.data.session import add_session_id, add_session_labels
from quant_rl.data.split import get_split_config, make_train_mask
from quant_rl.features.build import (
    FEATURE_CACHE_VERSION,
    feature_cache_content_hash,
    feature_cache_path,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
CAMPAIGN = ROOT / "outputs/gpu_campaigns" / CAMPAIGN_ID
PARITY_REPORT = ROOT / "docs/thesis/g1_feature_cache_raw_parity_current_source_2026-10-16.json"
OUTPUT = ROOT / "docs/thesis/g1_campaign_feature_cache_resolution_2026-10-16.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_input_signature(cfg: DictConfig) -> tuple[Any, ...]:
    return (
        str(cfg.data.raw_dir),
        str(cfg.data.primary),
        str(cfg.data.secondary),
        str(cfg.data.tz),
        str(cfg.session.end),
        json.dumps(OmegaConf.to_container(cfg.data.m1_files, resolve=True), sort_keys=True),
        tuple(get_split_config(cfg)),
    )


def main() -> int:
    campaign = json.loads((CAMPAIGN / "campaign.json").read_text(encoding="utf-8"))
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    config_paths = sorted((CAMPAIGN / "runs/attempts").rglob("config.yaml"))
    if len(config_paths) != 18:
        raise RuntimeError(f"expected 18 saved worker configs, found {len(config_paths)}")

    configs: list[tuple[Path, DictConfig]] = []
    for path in config_paths:
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"saved worker config is not a mapping: {path}")
        configs.append((path, cfg))
    signatures = {resolved_input_signature(cfg) for _, cfg in configs}
    if len(signatures) != 1:
        raise RuntimeError("worker configs do not share one raw-input and split identity")

    first = configs[0][1]
    primary_symbol = str(first.data.primary)
    secondary_symbol = str(first.data.secondary)
    raw_dir = ROOT / str(first.data.raw_dir)
    primary_path = raw_dir / str(first.data.m1_files[primary_symbol])
    secondary_path = raw_dir / str(first.data.m1_files[secondary_symbol])
    primary = add_session_id(
        add_session_labels(
            clean(load_bars(primary_path), tz=str(first.data.tz)),
            tz=str(first.data.tz),
            ny_end=str(first.session.end),
        )
    )
    secondary = add_session_id(
        add_session_labels(
            clean(load_bars(secondary_path), tz=str(first.data.tz)),
            tz=str(first.data.tz),
            ny_end=str(first.session.end),
        )
    )
    train_end, _ = get_split_config(first)
    train_mask = make_train_mask(pd.DatetimeIndex(primary.index), train_end)

    parity_by_hash = {item["expected_feature_hash"]: item for item in parity["results"]}
    if len(parity_by_hash) != len(parity["results"]):
        raise RuntimeError("raw/cache parity report contains duplicate profile hashes")

    rows: list[dict[str, Any]] = []
    for config_path, cfg in configs:
        identity = config_path.relative_to(CAMPAIGN / "runs/attempts").parts[0]
        feature_hash = feature_cache_content_hash(
            cfg, primary, train_mask=train_mask, secondary=secondary
        )
        profile = parity_by_hash.get(feature_hash)
        if profile is None:
            raise RuntimeError(f"run {identity} has unaudited feature profile {feature_hash}")

        explicit_cache = OmegaConf.select(cfg, "data.prebuilt_feature_cache")
        cache_path = (
            Path(str(explicit_cache))
            if explicit_cache
            else feature_cache_path(
                Path(str(cfg.data.cache_dir)),
                primary_symbol,
                cfg,
                primary,
                train_mask=train_mask,
                secondary=secondary,
            )
        )
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
        cache_path = cache_path.resolve()
        sidecar = Path(f"{cache_path}.hash")
        if not cache_path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"run {identity} resolved to missing cache or sidecar: {cache_path}")

        cache_sha = sha256_file(cache_path)
        if sidecar.read_text(encoding="utf-8").strip() != feature_hash:
            raise RuntimeError(f"run {identity} cache sidecar does not match {feature_hash}")
        if cache_sha != profile["cache_sha256"]:
            raise RuntimeError(
                f"run {identity} cache bytes differ from raw/cache audited profile "
                f"{profile['variant']}"
            )

        rows.append(
            {
                "variant": identity,
                "config_path": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256_file(config_path),
                "feature_hash": feature_hash,
                "matched_parity_profile": profile["variant"],
                "cache_path": str(cache_path.relative_to(ROOT)),
                "cache_sha256": cache_sha,
                "sidecar_value": sidecar.read_text(encoding="utf-8").strip(),
                "explicit_prebuilt_cache": bool(explicit_cache),
            }
        )

    rows.sort(key=lambda row: row["variant"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "audit": "active campaign worker configs resolve to raw/cache parity-audited profiles",
        "scope_note": (
            "All 18 saved attempt configs were hashed against the campaign M1 inputs and split. "
            "Each resolved cache's bytes and sidecar were matched to one of four representative "
            "raw-to-cache parity profiles. This establishes cache resolution/parity coverage, "
            "not strategy validity, run completion, or full G1 closure."
        ),
        "campaign_id": CAMPAIGN_ID,
        "campaign_fingerprint": campaign["fingerprint"],
        "campaign_git_sha": campaign["git_sha"],
        "slurm_job_id": campaign.get("job_id"),
        "repository_head": head,
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "raw_cache_parity_report": str(PARITY_REPORT.relative_to(ROOT)),
        "raw_cache_parity_report_sha256": sha256_file(PARITY_REPORT),
        "raw_input_sha256": {
            str(primary_path.relative_to(ROOT)): sha256_file(primary_path),
            str(secondary_path.relative_to(ROOT)): sha256_file(secondary_path),
        },
        "config_count": len(rows),
        "distinct_feature_hashes": len({row["feature_hash"] for row in rows}),
        "matched_profile_counts": {
            key: sum(row["matched_parity_profile"] == key for row in rows)
            for key in sorted({row["matched_parity_profile"] for row in rows})
        },
        "results": rows,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS: {len(rows)}/18 configs matched {len(parity_by_hash)} audited profiles; "
        f"hash groups={report['matched_profile_counts']}"
    )
    print(f"report={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
