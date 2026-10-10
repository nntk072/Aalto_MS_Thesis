"""Atomic, read-only numpy bundles shared by training environment workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

BUNDLE_VERSION = 2
_ARRAY_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_RELEVANT_CONFIG_PATHS = (
    "env.obs_window",
    "env.strategy_actions",
    "strategy.raw_columns",
    "features",
    "feature_config_hash",
    "data.tz",
    "data.primary",
    "data.secondary",
    "data.symbols",
    "data.m1_files",
    "data.tick_files",
    "data.split.train_end",
    "data.split.test_start",
)


@dataclass(frozen=True)
class BundleArrays:
    """Read-only memory maps and their JSON manifest."""

    path: Path
    arrays: Mapping[str, NDArray[Any]]
    manifest: Mapping[str, Any]

    def __getitem__(self, name: str) -> NDArray[Any]:
        """Return one named read-only array."""
        return self.arrays[name]


def _lookup_path(config: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    """Read a nested mapping value by dot path, including DictConfig values."""
    current: Any = config
    for part in dotted_path.split("."):
        if not hasattr(current, "get"):
            return False, None
        sentinel = object()
        current = current.get(part, sentinel)
        if current is sentinel:
            return False, None
    return True, current


def _relevant_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only settings that affect the bundled environment data."""
    relevant: dict[str, Any] = {}
    for dotted_path in _RELEVANT_CONFIG_PATHS:
        exists, value = _lookup_path(config, dotted_path)
        if exists:
            target = relevant
            parts = dotted_path.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
    return relevant


def _json_compatible(value: Any) -> Any:
    """Convert config containers into stable JSON primitives."""
    if hasattr(value, "items"):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def bundle_key(
    config: Mapping[str, Any],
    source_fingerprints: Mapping[str, Any],
    bundle_version: int = BUNDLE_VERSION,
) -> str:
    """Hash relevant config, source fingerprints, and bundle format version."""
    payload = {
        "bundle_version": bundle_version,
        "config": _relevant_config(config),
        "sources": source_fingerprints,
    }
    canonical = json.dumps(
        _json_compatible(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_array(path: Path, array: NDArray[Any]) -> None:
    """Write and fsync one numpy array file without pickle support."""
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write and fsync the bundle manifest."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory entries before and after the atomic rename."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_arrays(arrays: Mapping[str, NDArray[Any]]) -> None:
    """Reject unsafe names and object arrays before creating files."""
    if not arrays:
        raise ValueError("a bundle must contain at least one array")
    for name, array in arrays.items():
        if not _ARRAY_NAME.fullmatch(name):
            raise ValueError(f"invalid bundle array name: {name!r}")
        if not isinstance(array, np.ndarray):
            raise TypeError(f"bundle value {name!r} must be a numpy array")
        if array.dtype.hasobject:
            raise TypeError(f"bundle array {name!r} cannot use object dtype")


def _write_bundle_directory(
    temp_path: Path,
    arrays: Mapping[str, NDArray[Any]],
    *,
    key: str,
    config: Mapping[str, Any],
    source_fingerprints: Mapping[str, Any],
    column_names: Mapping[str, Any] | None,
    labels: Mapping[str, Any] | None,
    bundle_version: int,
) -> None:
    """Write array files followed by the manifest inside a temporary dir."""
    array_manifest: dict[str, dict[str, Any]] = {}
    for name in sorted(arrays):
        array = arrays[name]
        filename = f"{name}.npy"
        _write_array(temp_path / filename, array)
        array_manifest[name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    manifest = {
        "bundle_version": bundle_version,
        "key": key,
        "config": _json_compatible(_relevant_config(config)),
        "source_fingerprints": _json_compatible(source_fingerprints),
        "arrays": array_manifest,
        "column_names": _json_compatible(dict(column_names or {})),
        "labels": _json_compatible(dict(labels or {})),
    }
    _write_manifest(temp_path / "manifest.json", manifest)
    _fsync_directory(temp_path)


def build_bundle(
    cache_dir: Path | str,
    *,
    arrays: Mapping[str, NDArray[Any]],
    config: Mapping[str, Any],
    source_fingerprints: Mapping[str, Any],
    column_names: Mapping[str, Any] | None = None,
    labels: Mapping[str, Any] | None = None,
    bundle_version: int = BUNDLE_VERSION,
) -> Path:
    """Build or reuse a content-keyed bundle under ``cache_dir/shared_env``.

    Builders for the same key serialize with ``flock``. The manifest is the
    final file written in a private temporary directory, which is renamed into
    place atomically only after every file has reached disk.
    """
    _validate_arrays(arrays)
    key = bundle_key(config, source_fingerprints, bundle_version)
    root = Path(cache_dir) / "shared_env"
    root.mkdir(parents=True, exist_ok=True)
    bundle_path = root / key
    lock_path = root / f"{key}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if bundle_path.exists():
            load_bundle(bundle_path)
            return bundle_path
        temp_path = Path(tempfile.mkdtemp(prefix=f"{key}.tmp-", dir=root))
        try:
            _write_bundle_directory(
                temp_path,
                arrays,
                key=key,
                config=config,
                source_fingerprints=source_fingerprints,
                column_names=column_names,
                labels=labels,
                bundle_version=bundle_version,
            )
            os.replace(temp_path, bundle_path)
            _fsync_directory(root)
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)
    return bundle_path


def load_bundle(path: Path | str) -> BundleArrays:
    """Load each bundle array as a read-only memory map and validate metadata."""
    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, NDArray[Any]] = {}
    entries = manifest.get("arrays")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"bundle has no array entries: {bundle_path}")
    for name, metadata in entries.items():
        filename = metadata.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid array filename for {name!r}")
        array = np.load(bundle_path / filename, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata.get("shape") or array.dtype.str != metadata.get("dtype"):
            raise ValueError(f"array metadata mismatch for {name!r}")
        if array.flags.writeable:
            raise ValueError(f"bundle array {name!r} is not read-only")
        arrays[name] = array
    return BundleArrays(
        path=bundle_path,
        arrays=MappingProxyType(arrays),
        manifest=MappingProxyType(manifest),
    )
