"""Focused campaign execution source snapshot services."""

from __future__ import annotations

import hashlib
import json
import numbers
import pathlib
import subprocess
import zipfile
from typing import Any

from experiments.campaigns.constants import ROOT

_CACHE_FILE_HASHES: dict[tuple[str, int, int, int], str] = {}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_data_inputs(
    data_config: dict[str, Any], *, root: pathlib.Path = ROOT
) -> dict[str, str]:
    """Fingerprint the exact split bar/tick inputs and date-level calendar."""
    data_inputs: dict[str, str] = {}
    configured_datasets = data_config.get("datasets")
    if configured_datasets:
        dataset_sources = [
            (str(name), root / str(spec["raw_dir"]), spec)
            for name, spec in configured_datasets.items()
        ]
    else:
        dataset_sources = [("all", root / str(data_config["raw_dir"]), data_config)]

    for dataset_name, raw_dir, dataset in dataset_sources:
        for group in ("m1_files", "tick_files"):
            for filename in dataset.get(group, {}).values():
                path = raw_dir / str(filename)
                relative = str(path.relative_to(root))
                data_inputs[f"{dataset_name}:{relative}"] = (
                    sha256_file(path) if path.is_file() else "MISSING"
                )
        for timeframe, symbol_files in dataset.get("timeframe_files", {}).items():
            for filename in symbol_files.values():
                path = raw_dir / str(filename)
                relative = str(path.relative_to(root))
                data_inputs[f"{dataset_name}:{relative}"] = (
                    sha256_file(path) if path.is_file() else "MISSING"
                )

    schedule_value = data_config.get("session_schedule_file")
    if schedule_value:
        schedule_path = root / str(schedule_value)
        relative = str(schedule_path.relative_to(root))
        data_inputs[f"calendar:{relative}"] = (
            sha256_file(schedule_path) if schedule_path.is_file() else "MISSING"
        )
    return data_inputs


def cached_sha256_file(path: pathlib.Path) -> str:
    """Hash a shared campaign cache once per unchanged file identity."""
    stat = path.stat()
    identity = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    digest = _CACHE_FILE_HASHES.get(identity)
    if digest is None:
        digest = sha256_file(path)
        _CACHE_FILE_HASHES[identity] = digest
    return digest


def action_space_matches_width(action_space: Any, width: int) -> bool:
    """Check vector and legacy discrete action spaces without conflating them."""
    discrete_width = getattr(action_space, "n", None)
    if isinstance(discrete_width, numbers.Integral):
        return int(discrete_width) == width and tuple(getattr(action_space, "shape", ())) == ()
    return getattr(action_space, "shape", None) == (width,)


def campaign_source_files(
    root: pathlib.Path, campaign_spec: pathlib.Path | None = None
) -> list[pathlib.Path]:
    """Return tracked and untracked source inputs hashed into campaign identity."""
    pathspecs = [
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
        "config",
        "scripts/compute/slurm",
        "scripts/research",
        "pyproject.toml",
        "uv.lock",
    ]
    tracked = subprocess.check_output(["git", "ls-files", "-z", *pathspecs], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", *pathspecs], cwd=root
    )
    files = {
        (root / item.decode()).resolve()
        for listing in (tracked, untracked)
        for item in listing.split(b"\0")
        if item
    }
    if campaign_spec is not None:
        files.add(campaign_spec.resolve())
    return sorted(path for path in files if path.is_file())


def write_campaign_source_snapshot(
    root: pathlib.Path,
    source_files: list[pathlib.Path],
    destination: pathlib.Path,
    *,
    git_sha: str,
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Archive the exact hashed campaign inputs before any worker is launched."""
    root = root.resolve()
    records: dict[str, str] = {}
    resolved_files: dict[str, pathlib.Path] = {}
    for path in source_files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"campaign source must be inside repository: {resolved}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"campaign source disappeared before snapshot: {resolved}")
        if relative in resolved_files:
            continue
        resolved_files[relative] = resolved
        records[relative] = sha256_file(resolved)

    if not records:
        raise ValueError("campaign source snapshot cannot be empty")
    if records != expected_source_hashes:
        raise RuntimeError("campaign source files changed while creating the source snapshot")
    snapshot_manifest = {
        "schema_version": 1,
        "git_sha": git_sha,
        "files": dict(sorted(records.items())),
    }
    manifest_bytes = (json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        strict_timestamps=True,
    ) as archive:
        for relative, path in sorted(resolved_files.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, mode="w") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
        manifest_info = zipfile.ZipInfo("source_manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(manifest_info, manifest_bytes, compresslevel=6)

    with zipfile.ZipFile(temporary, mode="r") as archive:
        for relative, expected_hash in records.items():
            digest = hashlib.sha256()
            with archive.open(relative, mode="r") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"campaign source snapshot verification failed: {relative}")
        if archive.read("source_manifest.json") != manifest_bytes:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("campaign source snapshot manifest verification failed")
    temporary.replace(destination)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(records),
    }


def verify_campaign_source_snapshot(archive_path: pathlib.Path, record: object) -> tuple[bool, str]:
    """Check an archived campaign source bundle against its campaign manifest."""
    if not isinstance(record, dict):
        return False, "source snapshot record is invalid"
    expected_hash = record.get("sha256")
    if not archive_path.is_file():
        return False, "source snapshot archive is missing"
    if not isinstance(expected_hash, str) or sha256_file(archive_path) != expected_hash:
        return False, "source snapshot archive hash mismatch"
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest_bytes = archive.read("source_manifest.json")
            source_manifest = json.loads(manifest_bytes)
            files = source_manifest["files"]
            if not isinstance(files, dict) or len(files) != record.get("file_count"):
                return False, "source snapshot file manifest is invalid"
            if hashlib.sha256(manifest_bytes).hexdigest() != record.get("manifest_sha256"):
                return False, "source snapshot manifest hash mismatch"
            if set(archive.namelist()) != set(files) | {"source_manifest.json"}:
                return False, "source snapshot contents differ from its file manifest"
            for name, expected_file_hash in files.items():
                digest = hashlib.sha256()
                with archive.open(name, mode="r") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_file_hash:
                    return False, f"source snapshot file hash mismatch: {name}"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, f"source snapshot is unreadable: {exc}"
    return True, "verified"
