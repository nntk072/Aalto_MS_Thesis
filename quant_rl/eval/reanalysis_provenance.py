"""Lightweight, fail-closed provenance checks for historical run reanalysis.

Keep these guards importable without the evaluation/training runtime so they
can be tested before an expensive checkpoint or data pipeline is loaded.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from quant_rl.data.split import make_train_mask
from quant_rl.eval.run_manifest import array_sha256, dataset_record, file_sha256
from quant_rl.features.build import feature_cache_content_hash, feature_cache_path

log = logging.getLogger(__name__)


def _load_run_config(run_dir: Path) -> DictConfig:
    """Require the original run snapshot for checkpoint evaluation parity."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Run config snapshot missing: {cfg_path}; refusing to evaluate "
            "a checkpoint against unrelated default settings"
        )
    log.info("Loading config snapshot: %s", cfg_path)
    cfg = OmegaConf.load(cfg_path)
    if not isinstance(cfg, DictConfig) or not cfg:
        raise ValueError(
            f"Run config snapshot must be a nonempty configuration mapping: {cfg_path}"
        )
    # get_split_config() deliberately falls back to hard-coded defaults for
    # legacy training callers. Historical checkpoint evaluation must instead
    # use the date boundaries saved with the checkpoint.
    for field in ("train_end", "test_start"):
        value = OmegaConf.select(cfg, f"data.split.{field}", default=None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Run config snapshot is missing explicit data.split.{field}: {cfg_path}"
            )
    return cfg


def _validate_reanalysis_request(
    source: Path, output: Path, *, allow_historical: bool = False
) -> None:
    """Require holdout acknowledgement and an entirely new output location."""
    if not allow_historical:
        raise ValueError("historical holdout reanalysis requires --allow-historical")
    source_path, output_path = source.resolve(), output.resolve()
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("reanalysis requires a separate output outside the source run")
    if output_path.exists():
        raise FileExistsError(f"reanalysis output already exists: {output_path}")


def _feature_build_scope(
    bars: Any,
    cfg: DictConfig,
    cache_dir: Path,
    symbol: str,
    train_end: str,
    secondary: Any | None = None,
) -> tuple[Any, Path]:
    """Reproduce the mask and content-hashed cache identity of training."""
    mask = make_train_mask(cast(Any, bars.index), train_end)
    prebuilt_value = OmegaConf.select(cfg, "data.prebuilt_feature_cache", default=None)
    if prebuilt_value:
        cache_path = Path(str(prebuilt_value)).expanduser()
        if not cache_path.is_absolute():
            cache_path = Path(__file__).resolve().parents[2] / cache_path
        cache_path = cache_path.resolve()
        hash_path = Path(f"{cache_path}.hash")
        expected_hash = feature_cache_content_hash(
            cfg,
            bars,
            train_mask=mask,
            secondary=secondary,
        )
        if not cache_path.is_file() or not hash_path.is_file():
            raise FileNotFoundError(
                f"configured prebuilt feature cache or its .hash sidecar is missing: {cache_path}"
            )
        actual_hash = hash_path.read_text(encoding="utf-8").strip()
        if actual_hash != expected_hash:
            raise ValueError(
                "configured prebuilt feature cache does not match the saved config/data hash: "
                f"{cache_path} (expected {expected_hash}, found {actual_hash})"
            )
        reference_mode = str(
            OmegaConf.select(cfg, "features.po3_reference_mode", default="m1_legacy")
        )
        if reference_mode == "htf_confirmed":
            sweep_audit = Path(f"{cache_path}.htf_sweeps.parquet")
            if not sweep_audit.is_file():
                raise FileNotFoundError(
                    f"configured prebuilt feature cache is missing HTF sweep audit: {sweep_audit}"
                )
        return mask, cache_path
    return mask, feature_cache_path(
        cache_dir, symbol, cfg, bars, train_mask=mask, secondary=secondary
    )


def _validate_reanalysis_feature_cache(
    source_feature_cache: Any,
    evaluated_cache_path: Path,
    *,
    allow_legacy: bool,
) -> tuple[str, str, str]:
    """Require feature-cache identity parity, or explicitly label legacy runs.

    A current feature cache is not evidence that a legacy checkpoint was
    trained with those feature values. Versioned source manifests must match
    both the content-derived cache identity and the cache's recorded digest.
    """
    hash_path = Path(f"{evaluated_cache_path}.hash")
    if not hash_path.is_file():
        raise FileNotFoundError(f"feature-cache hash sidecar is missing: {hash_path}")
    evaluated_hash = hash_path.read_text(encoding="utf-8").strip()
    if not evaluated_hash:
        raise ValueError(f"feature-cache hash sidecar is empty: {hash_path}")

    if source_feature_cache is None:
        if not allow_legacy:
            raise ValueError(
                "source run manifest has no feature-cache identity; pass "
                "--allow-legacy-data-provenance only for an explicitly unverified legacy replay"
            )
        disposition = "legacy_unverified"
    else:
        if not isinstance(source_feature_cache, dict):
            raise ValueError("source run manifest feature_cache must be an object")
        expected_identity = source_feature_cache.get("identity")
        expected_hash = source_feature_cache.get("sha256")
        expected_file_sha256 = source_feature_cache.get("file_sha256")
        if not isinstance(expected_identity, str) or not expected_identity:
            raise ValueError("source run manifest feature_cache.identity is missing")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError("source run manifest feature_cache.sha256 is missing")
        if evaluated_cache_path.name != expected_identity:
            raise ValueError(
                "reanalysis feature-cache identity differs from training: "
                f"expected {expected_identity}, got {evaluated_cache_path.name}"
            )
        if evaluated_hash != expected_hash:
            raise ValueError(
                "reanalysis feature-cache content identity differs from training: "
                f"expected {expected_hash}, got {evaluated_hash}"
            )
        evaluated_file_sha256 = file_sha256(evaluated_cache_path)
        if not isinstance(expected_file_sha256, str) or not expected_file_sha256:
            if not allow_legacy:
                raise ValueError(
                    "source run manifest feature_cache.file_sha256 is missing; "
                    "pass --allow-legacy-data-provenance only for an explicitly "
                    "unverified legacy replay"
                )
            disposition = "legacy_unverified"
        elif evaluated_file_sha256 != expected_file_sha256:
            raise ValueError(
                "reanalysis feature-cache file SHA-256 differs from training: "
                f"expected {expected_file_sha256}, got {evaluated_file_sha256}"
            )
        else:
            disposition = "verified"
    if source_feature_cache is None:
        evaluated_file_sha256 = file_sha256(evaluated_cache_path)
    return disposition, evaluated_hash, evaluated_file_sha256


def _load_exact_source_feature_cache(
    source_feature_cache: Any,
    cache_path: Path,
) -> tuple[pd.DataFrame, Path, str, str]:
    """Load a preserved training cache only when manifest identity and bytes match.

    This is for historical replay after the current feature builder/cache has
    changed. It never rebuilds or writes the supplied cache; the caller must
    still verify the train/test feature fingerprints against the run manifest.
    """
    if not isinstance(source_feature_cache, dict):
        raise ValueError("exact source-cache replay requires a feature_cache manifest object")
    if not isinstance(source_feature_cache.get("file_sha256"), str):
        raise ValueError(
            "exact source-cache replay requires feature_cache.file_sha256 in the run manifest"
        )
    resolved_path = cache_path.expanduser().resolve()
    disposition, cache_hash, file_hash = _validate_reanalysis_feature_cache(
        source_feature_cache,
        resolved_path,
        allow_legacy=False,
    )
    if disposition != "verified":
        raise ValueError("exact source-cache replay could not verify the training cache")
    features = pd.read_parquet(resolved_path)
    if not isinstance(features, pd.DataFrame):
        raise TypeError("source feature cache did not load as a DataFrame")
    return features, resolved_path, cache_hash, file_hash


def _validate_reanalysis_datasets(
    source_datasets: Any,
    evaluated_datasets: dict[str, dict[str, Any]],
    *,
    allow_legacy: bool,
) -> str:
    """Require the evaluation bars/features/ticks to match the training record."""
    if source_datasets is None:
        if not allow_legacy:
            raise ValueError(
                "source run manifest has no dataset identities; pass "
                "--allow-legacy-data-provenance only for an explicitly unverified legacy replay"
            )
        return "legacy_unverified"
    if not isinstance(source_datasets, dict):
        raise ValueError("source run manifest datasets must be an object")

    required = {"train_bars", "train_features", "test_bars", "test_features"}
    source_keys = set(source_datasets)
    if not required.issubset(source_keys):
        if allow_legacy and not source_keys.intersection(required):
            return "legacy_unverified"
        raise ValueError(
            "source run manifest has incomplete train/test dataset identities: "
            f"missing {sorted(required.difference(source_keys))}"
        )
    if source_keys != set(evaluated_datasets):
        raise ValueError(
            "reanalysis dataset set differs from training: "
            f"expected {sorted(source_keys)}, got {sorted(evaluated_datasets)}"
        )
    for name in sorted(source_keys):
        source_record = source_datasets[name]
        evaluated_record = evaluated_datasets[name]
        if not isinstance(source_record, dict) or not isinstance(evaluated_record, dict):
            raise ValueError(f"dataset identity record must be an object: {name}")
        source_digest = source_record.get("sha256")
        evaluated_digest = evaluated_record.get("sha256")
        if not isinstance(source_digest, str) or not source_digest:
            raise ValueError(f"source run manifest dataset SHA-256 is missing: {name}")
        if source_digest != evaluated_digest:
            raise ValueError(
                f"reanalysis dataset differs from training for {name}: "
                f"expected {source_digest}, got {evaluated_digest}"
            )
    return "verified"


def _evaluation_dataset_records(
    *,
    train_bars: Any,
    train_features: Any,
    test_bars: Any,
    test_features: Any,
    train_secondary: Any | None = None,
    test_secondary: Any | None = None,
    primary_ticks: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Fingerprint the exact frames and ticks consumed by checkpoint evaluation."""
    records = {
        "train_bars": dataset_record(train_bars),
        "train_features": dataset_record(train_features),
        "test_bars": dataset_record(test_bars),
        "test_features": dataset_record(test_features),
    }
    if train_secondary is not None:
        records["train_secondary"] = dataset_record(train_secondary)
    if test_secondary is not None:
        records["test_secondary"] = dataset_record(test_secondary)
    if primary_ticks is not None:
        tick_digest = hashlib.sha256(
            "".join(
                array_sha256(array)
                for array in (primary_ticks._ts, primary_ticks._bid, primary_ticks._ask)
            ).encode()
        ).hexdigest()
        records["primary_ticks"] = {
            "rows": int(len(primary_ticks._ts)),
            "columns": ["timestamp_ns", "bid", "ask"],
            "start": (
                str(pd.to_datetime(primary_ticks._ts[0], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "end": (
                str(pd.to_datetime(primary_ticks._ts[-1], utc=True))
                if len(primary_ticks._ts)
                else None
            ),
            "sha256": tick_digest,
        }
    return records


def _reanalysis_backend_disposition(
    source_protocol: Any, evaluation_backend: str
) -> tuple[str, str]:
    """Label whether report-only reanalysis uses the training execution backend."""
    if not isinstance(source_protocol, dict):
        return "unknown", "legacy_unverified"
    source_backend = source_protocol.get("data_source")
    if not isinstance(source_backend, str) or not source_backend:
        return "unknown", "legacy_unverified"
    # The training configuration/API uses ``frames`` while evaluation uses
    # the singular backend name ``frame``. Treat those spellings as one
    # execution mode so an ordinary frame replay is not falsely flagged.
    source_mode = "frame" if source_backend == "frames" else source_backend
    evaluation_mode = "frame" if evaluation_backend == "frames" else evaluation_backend
    if source_mode == evaluation_mode:
        return source_backend, "same_backend"
    if source_mode == "bundle" and evaluation_mode == "frame":
        return source_backend, "cross_backend_requires_checkpoint_parity"
    return source_backend, "backend_mismatch"
