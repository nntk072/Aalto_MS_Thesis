#!/usr/bin/env python3
"""Validate and CPU-load every resumable checkpoint selected for a campaign."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any

from experiments.campaigns.execution import (
    ROOT,
    atomic_json,
    build_initial_checkpoint_map,
    sha256_file,
    utc,
    validate_initial_checkpoints,
    verify_initial_checkpoint_models,
)


def _read_variants(path: pathlib.Path) -> list[str]:
    variants = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variant file must contain unique, non-empty names")
    return variants


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_explicit_checkpoint_map(
    map_path: pathlib.Path,
    jobs: list[dict[str, Any]],
    target_steps: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate a frozen checkpoint map without selecting a newer campaign attempt."""
    source_path = map_path.resolve()
    try:
        source_path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("checkpoint map must be inside this repository") from None
    if not source_path.is_file():
        raise FileNotFoundError(f"checkpoint map does not exist: {source_path}")
    source_sha256 = sha256_file(source_path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("checkpoints"), list):
        raise ValueError("checkpoint map must contain a checkpoints list")
    input_schema_version = raw.get("schema_version")
    added_schema_version = input_schema_version is None
    if input_schema_version is not None and input_schema_version != 1:
        raise ValueError(f"unsupported checkpoint map schema_version: {input_schema_version!r}")
    normalized = dict(raw)
    normalized["schema_version"] = 1
    fd, temp_name = tempfile.mkstemp(
        prefix=".explicit-checkpoint-map-", suffix=".json", dir=source_path.parent
    )
    normalized_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
        records = validate_initial_checkpoints(normalized_path, jobs, target_steps)
        normalized_sha256 = sha256_file(normalized_path)
    finally:
        normalized_path.unlink(missing_ok=True)
    provenance = {
        "input_path": str(source_path),
        "input_sha256": source_sha256,
        "input_schema_version": input_schema_version,
        "normalized_schema_version": 1,
        "schema_version_added": added_schema_version,
        "normalized_map_sha256": normalized_sha256,
    }
    return records, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-campaign",
        action="append",
        type=pathlib.Path,
        help="source campaign directory; repeat for parent manifests",
    )
    source_group.add_argument(
        "--checkpoint-map",
        type=pathlib.Path,
        help="frozen controller checkpoint map; bypass automatic latest-attempt selection",
    )
    parser.add_argument(
        "--variant-file", type=pathlib.Path, default=ROOT / "config/campaigns/variant_list.txt"
    )
    parser.add_argument("--seeds", required=True, help="space- or comma-separated seeds")
    parser.add_argument("--timesteps", required=True, type=int)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    variants = _read_variants(args.variant_file)
    seeds = [int(seed) for seed in args.seeds.replace(",", " ").split()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed list must be non-empty and unique")
    jobs = [
        {"key": f"{variant}__seed{seed}", "variant": variant, "seed": seed}
        for variant in variants
        for seed in seeds
    ]
    required_keys = {str(job["key"]) for job in jobs}
    output = args.out.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("report path must be inside this repository") from None
    if output.exists():
        raise FileExistsError(f"refusing to replace checkpoint-load report: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_map is not None:
        records, checkpoint_map_provenance = _load_explicit_checkpoint_map(
            args.checkpoint_map, jobs, args.timesteps
        )
    else:
        checkpoint_map = build_initial_checkpoint_map(
            args.source_campaign or [], required_keys, args.timesteps
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint-load-map-", suffix=".json", dir=output.parent
        )
        map_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(checkpoint_map, handle, indent=2, sort_keys=True)
                handle.write("\n")
            records = validate_initial_checkpoints(map_path, jobs, args.timesteps)
            checkpoint_map_provenance = {
                "input_path": None,
                "input_sha256": None,
                "input_schema_version": 1,
                "normalized_schema_version": 1,
                "schema_version_added": False,
                "normalized_map_sha256": sha256_file(map_path),
            }
        finally:
            map_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    for key, record in sorted(records.items()):
        checkpoint = pathlib.Path(str(record["checkpoint_path"]))
        expected_sha = str(record["checkpoint_sha256"])
        item: dict[str, Any] = {
            "key": key,
            "source_campaign_id": record["source_campaign_id"],
            "source_campaign_git_sha": record["source_campaign_git_sha"],
            "source_campaign_fingerprint": record["source_campaign_fingerprint"],
            "source_attempt": record["source_attempt"],
            "source_attempt_provenance_sha256": record["source_attempt_provenance_sha256"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": expected_sha,
            "expected_num_timesteps": int(record["source_num_timesteps"]),
            "device": "cpu",
            "status": "failed",
        }
        try:
            if sha256_file(checkpoint) != expected_sha:
                raise ValueError("checkpoint changed after lineage validation")
            loaded_steps = verify_initial_checkpoint_models({key: checkpoint}, {key: record})
            if sha256_file(checkpoint) != expected_sha:
                raise ValueError("checkpoint changed while loading")
            item["loaded_num_timesteps"] = loaded_steps[key]
            item["status"] = "passed"
        except Exception as exc:  # preserve a per-checkpoint diagnostic in the report
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
        print(f"{item['status'].upper()} {key} steps={item['expected_num_timesteps']}")

    packages = {name: _package_version(name) for name in ("numpy", "stable-baselines3", "torch")}
    report = {
        "schema_version": 1,
        "created_at": utc(),
        "repository": {
            "root": str(ROOT),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_dirty_paths": [
                line[3:]
                for line in subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=ROOT,
                    text=True,
                ).splitlines()
                if len(line) >= 4
            ],
            "audit_source_sha256": {
                "scripts/research/campaigns/audit_campaign_checkpoint_loads.py": sha256_file(
                    pathlib.Path(__file__).resolve()
                ),
                "experiments/campaigns/execution.py": sha256_file(
                    ROOT / "experiments/campaigns/execution.py"
                ),
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        "target_timesteps": args.timesteps,
        "checkpoint_map": checkpoint_map_provenance,
        "device": "cpu",
        "checkpoint_count": len(results),
        "passed": sum(item["status"] == "passed" for item in results),
        "failed": sum(item["status"] != "passed" for item in results),
        "results": results,
        "scope_note": (
            "This confirms local CPU deserialization and saved timestep identity. "
            "It does not establish deserialization on the Triton GH200 runtime or "
            "training/evaluation success."
        ),
    }
    if output.exists():
        raise FileExistsError(f"refusing to replace checkpoint-load report: {output}")
    atomic_json(output, report)
    print(
        json.dumps({"report": str(output), "passed": report["passed"], "failed": report["failed"]})
    )
    return 0 if report["failed"] == 0 and report["passed"] == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
