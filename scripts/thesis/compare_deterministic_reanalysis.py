#!/usr/bin/env python3
"""Compare two immutable deterministic checkpoint-reanalysis packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

IDENTITY_FIELDS = (
    "artifact_role",
    "historical_holdout_inspected",
    "evaluation_provenance",
    "source_run",
    "source_config_sha256",
    "reanalysis_config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_manifest_hash_disposition",
    "action_schema_version",
    "action_layout_key",
    "action_schema",
    "observation_schema",
    "sequence_feature_schema",
    "sequence_feature_schema_disposition",
    "legacy_sequence_columns_used",
    "legacy_action_schema_opt_in",
    "legacy_observation_schema_opt_in",
    "legacy_sequence_schema_opt_in",
    "legacy_stop_schema_opt_in",
    "legacy_data_provenance_opt_in",
    "legacy_action_layout_selected",
    "stop_schema_disposition",
    "stop_candidate_schema",
    "feature_cache_identity",
    "feature_cache_hash",
    "feature_cache_file_sha256",
    "source_feature_cache_replay_path",
    "source_feature_cache",
    "feature_cache_disposition",
    "source_datasets",
    "evaluated_datasets",
    "dataset_parity_disposition",
    "training_data_source",
    "evaluation_backend",
    "backend_parity_disposition",
    "train_end",
    "test_start",
    "seed",
    "metrics_protocol",
    "policy_action_mode",
)
SPLITS = ("training", "testing")
NUMERICAL_ARTIFACTS = (
    "equity.csv",
    "trades.csv",
    "fills.csv",
    "positions.csv",
    "metrics.json",
    "evaluation_diagnostics.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "reanalysis_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"reanalysis manifest must be an object: {path}")
    if value.get("artifact_role") != "historical_reanalysis":
        raise ValueError(f"unsupported artifact role in {path}")
    if value.get("policy_action_mode") != "deterministic":
        raise ValueError(f"reanalysis is not deterministic: {path}")
    provenance = value.get("evaluation_provenance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("source"), dict)
        or not isinstance(provenance.get("runtime"), dict)
        or not isinstance(provenance.get("python_executable"), str)
    ):
        raise ValueError(f"evaluation implementation/runtime provenance is incomplete: {path}")
    source = provenance["source"]
    runtime = provenance["runtime"]
    if (
        source.get("available") is not True
        or not isinstance(source.get("head"), str)
        or not source["head"]
        or not isinstance(runtime.get("python"), str)
        or not isinstance(runtime.get("platform"), str)
    ):
        raise ValueError(f"evaluation source/runtime identity is incomplete: {path}")
    for field in (
        "legacy_sequence_columns_used",
        "legacy_action_schema_opt_in",
        "legacy_observation_schema_opt_in",
        "legacy_sequence_schema_opt_in",
        "legacy_stop_schema_opt_in",
        "legacy_data_provenance_opt_in",
    ):
        if type(value.get(field)) is not bool:
            raise ValueError(f"legacy mode field {field!r} is missing or invalid: {path}")
    if value.get("legacy_action_layout_selected") is not None and not isinstance(
        value.get("legacy_action_layout_selected"), str
    ):
        raise ValueError(f"legacy action layout is invalid: {path}")
    return value


def _source_manifest_errors(
    label: str,
    source_run: Path,
    reanalysis_manifest: dict[str, Any],
) -> list[str]:
    """Bind copied reanalysis provenance to the original training manifest."""
    errors: list[str] = []
    path = source_run / "run_manifest.json"
    if not path.is_file():
        return [f"{label} original run manifest is missing"]
    source_manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict):
        return [f"{label} original run manifest is not an object"]

    config = source_manifest.get("config")
    if not isinstance(config, dict) or config.get("sha256") != reanalysis_manifest.get(
        "source_config_sha256"
    ):
        errors.append(f"{label} source config hash differs from original run manifest")

    checkpoint = source_manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != reanalysis_manifest.get(
        "checkpoint_sha256"
    ):
        errors.append(f"{label} checkpoint hash differs from original run manifest")

    source_cache = source_manifest.get("feature_cache")
    replay_cache = reanalysis_manifest.get("source_feature_cache")
    if not isinstance(source_cache, dict) or not isinstance(replay_cache, dict):
        errors.append(f"{label} feature-cache record is missing from an original manifest")
    else:
        for field in ("identity", "sha256", "file_sha256"):
            if source_cache.get(field) != replay_cache.get(field):
                errors.append(f"{label} source feature-cache {field} differs from original")
        evaluated_cache_fields = {
            "identity": "feature_cache_identity",
            "sha256": "feature_cache_hash",
            "file_sha256": "feature_cache_file_sha256",
        }
        for source_field, evaluated_field in evaluated_cache_fields.items():
            if source_cache.get(source_field) != reanalysis_manifest.get(evaluated_field):
                errors.append(f"{label} evaluated feature cache {source_field} differs from source")

    source_datasets = source_manifest.get("datasets")
    replay_datasets = reanalysis_manifest.get("source_datasets")
    if not isinstance(source_datasets, dict) or not isinstance(replay_datasets, dict):
        errors.append(f"{label} dataset records are missing from an original manifest")
    elif set(source_datasets) != set(replay_datasets):
        errors.append(f"{label} dataset set differs from original run manifest")
    else:
        for name, source_record in source_datasets.items():
            replay_record = replay_datasets[name]
            if (
                not isinstance(source_record, dict)
                or not isinstance(replay_record, dict)
                or source_record.get("sha256") != replay_record.get("sha256")
            ):
                errors.append(f"{label} dataset hash differs from original: {name}")

    # The numerical-report verifier applies these same compatibility gates.
    # Repeatability is meaningless if both replays silently used a different
    # action, observation, stop, or sequence-feature contract than training.
    for schema_key in ("action_schema", "observation_schema", "stop_candidate_schema"):
        original_schema = source_manifest.get(schema_key)
        replay_schema = reanalysis_manifest.get(schema_key)
        if not isinstance(original_schema, dict):
            errors.append(f"{label} original run manifest {schema_key} is missing")
            continue
        if not isinstance(replay_schema, dict):
            errors.append(f"{label} reanalysis {schema_key} is missing")
            continue
        replay_contract = (
            {key: value for key, value in replay_schema.items() if key != "checkpoint_metadata"}
            if schema_key == "action_schema"
            else replay_schema
        )
        if replay_contract != original_schema:
            errors.append(f"{label} reanalysis {schema_key} differs from original run manifest")

    original_sequence_schema = source_manifest.get("sequence_feature_schema")
    replay_sequence_schema = reanalysis_manifest.get("sequence_feature_schema")
    sequence_disposition = reanalysis_manifest.get("sequence_feature_schema_disposition")
    if isinstance(original_sequence_schema, dict):
        if replay_sequence_schema != original_sequence_schema:
            errors.append(f"{label} sequence-feature schema differs from original run manifest")
        if sequence_disposition != "verified":
            errors.append(f"{label} sequence-feature schema is not verified")
        if reanalysis_manifest.get("legacy_sequence_schema_opt_in") is not False:
            errors.append(f"{label} sequence-feature schema used legacy opt-in")
        if reanalysis_manifest.get("legacy_sequence_columns_used") is not False:
            errors.append(f"{label} replay used legacy sequence columns")
    else:
        if sequence_disposition != "legacy_unverified":
            errors.append(f"{label} missing source sequence schema is not marked legacy")
        if reanalysis_manifest.get("legacy_sequence_schema_opt_in") is not True:
            errors.append(f"{label} missing source sequence schema lacks explicit opt-in")
        if reanalysis_manifest.get("legacy_sequence_columns_used") is not True:
            errors.append(f"{label} missing source sequence schema did not use legacy columns")

    if reanalysis_manifest.get("checkpoint_manifest_hash_disposition") != "verified":
        errors.append(f"{label} checkpoint manifest hash is not verified")
    if reanalysis_manifest.get("stop_schema_disposition") != "versioned":
        errors.append(f"{label} stop-candidate schema is not versioned")
    for opt_in in (
        "legacy_action_schema_opt_in",
        "legacy_observation_schema_opt_in",
        "legacy_stop_schema_opt_in",
    ):
        if reanalysis_manifest.get(opt_in) is not False:
            errors.append(f"{label} used unsupported legacy schema opt-in: {opt_in}")
    original_protocol = source_manifest.get("execution_protocol")
    if not isinstance(original_protocol, dict) or original_protocol.get(
        "data_source"
    ) != reanalysis_manifest.get("training_data_source"):
        errors.append(f"{label} training backend differs from original run manifest")
    return errors


def compare_reanalyses(first_dir: Path, second_dir: Path) -> dict[str, Any]:
    """Require shared provenance and byte-identical numerical outputs."""
    first = first_dir.resolve()
    second = second_dir.resolve()
    first_manifest = _read_manifest(first)
    second_manifest = _read_manifest(second)
    errors: list[str] = []

    identity: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        left = first_manifest.get(field)
        right = second_manifest.get(field)
        identity[field] = left
        if left != right:
            errors.append(f"reanalysis identity differs: {field}")
    if first_manifest.get("dataset_parity_disposition") != "verified":
        errors.append("first reanalysis does not have verified dataset parity")
    if second_manifest.get("dataset_parity_disposition") != "verified":
        errors.append("second reanalysis does not have verified dataset parity")
    if first_manifest.get("checkpoint_manifest_hash_disposition") != "verified":
        errors.append("first reanalysis checkpoint is not hash-verified")
    if second_manifest.get("checkpoint_manifest_hash_disposition") != "verified":
        errors.append("second reanalysis checkpoint is not hash-verified")

    first_config = first / "config.yaml"
    second_config = second / "config.yaml"
    if not first_config.is_file() or not second_config.is_file():
        errors.append("one or both reanalysis config snapshots are missing")
        config_hashes: dict[str, str | None] = {
            "first": _sha256(first_config) if first_config.is_file() else None,
            "second": _sha256(second_config) if second_config.is_file() else None,
        }
    else:
        config_hashes = {"first": _sha256(first_config), "second": _sha256(second_config)}
        if config_hashes["first"] != config_hashes["second"]:
            errors.append("reanalysis config snapshots differ")
        if first_manifest.get("reanalysis_config_sha256") != config_hashes["first"]:
            errors.append("first reanalysis config hash differs from its manifest")
        if second_manifest.get("reanalysis_config_sha256") != config_hashes["second"]:
            errors.append("second reanalysis config hash differs from its manifest")

    for label, manifest in (("first", first_manifest), ("second", second_manifest)):
        source_run = manifest.get("source_run")
        source_config_hash = manifest.get("source_config_sha256")
        checkpoint_path = manifest.get("checkpoint")
        checkpoint_hash = manifest.get("checkpoint_sha256")
        if not isinstance(source_run, str) or not isinstance(source_config_hash, str):
            errors.append(f"{label} reanalysis lacks source config provenance")
        else:
            source_config = Path(source_run) / "config.yaml"
            if not source_config.is_file() or _sha256(source_config) != source_config_hash:
                errors.append(f"{label} source config does not match its recorded hash")
            errors.extend(_source_manifest_errors(label, Path(source_run), manifest))
        if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_hash, str):
            errors.append(f"{label} reanalysis lacks checkpoint file provenance")
        else:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_hash:
                errors.append(f"{label} checkpoint does not match its recorded hash")
        if manifest.get("feature_cache_disposition") not in {
            "verified",
            "source_cache_replay_verified",
        }:
            errors.append(f"{label} reanalysis feature cache is not hash-verified")
        evaluated_cache_path = manifest.get("evaluated_feature_cache_path")
        evaluated_cache_hash = manifest.get("feature_cache_file_sha256")
        if not isinstance(evaluated_cache_path, str) or not isinstance(evaluated_cache_hash, str):
            errors.append(f"{label} evaluated feature cache provenance is missing")
        else:
            evaluated_cache = Path(evaluated_cache_path)
            if not evaluated_cache.is_file() or _sha256(evaluated_cache) != evaluated_cache_hash:
                errors.append(f"{label} evaluated feature cache does not match its recorded hash")
        cache_path = manifest.get("source_feature_cache_replay_path")
        cache_hash = manifest.get("feature_cache_file_sha256")
        if cache_path is not None:
            if not isinstance(cache_path, str) or not isinstance(cache_hash, str):
                errors.append(f"{label} source feature cache provenance is malformed")
            else:
                cache_file = Path(cache_path)
                if not cache_file.is_file() or _sha256(cache_file) != cache_hash:
                    errors.append(f"{label} source feature cache does not match its recorded hash")

    artifacts: list[dict[str, Any]] = []
    for split in SPLITS:
        for name in NUMERICAL_ARTIFACTS:
            first_path = first / split / name
            second_path = second / split / name
            left_hash = _sha256(first_path) if first_path.is_file() else None
            right_hash = _sha256(second_path) if second_path.is_file() else None
            same = left_hash is not None and left_hash == right_hash
            artifacts.append(
                {
                    "split": split,
                    "name": name,
                    "first_sha256": left_hash,
                    "second_sha256": right_hash,
                    "identical": same,
                }
            )
            if left_hash is None or right_hash is None:
                errors.append(f"required numerical artifact missing: {split}/{name}")
            elif not same:
                errors.append(f"numerical artifact differs: {split}/{name}")

    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "artifact_role": "deterministic_reanalysis_repeatability",
        "first_reanalysis": str(first),
        "second_reanalysis": str(second),
        "identity": identity,
        "evaluated_feature_caches": {
            "first": {
                "path": first_manifest.get("evaluated_feature_cache_path"),
                "sha256": first_manifest.get("feature_cache_file_sha256"),
            },
            "second": {
                "path": second_manifest.get("evaluated_feature_cache_path"),
                "sha256": second_manifest.get("feature_cache_file_sha256"),
            },
        },
        "config_sha256": config_hashes,
        "numerical_artifacts": artifacts,
        "errors": errors,
        "interpretation_limit": (
            "Repeatability of deterministic reanalysis artifacts does not establish "
            "strategy validity, out-of-sample generalization, or confirmatory evidence."
        ),
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    """Create a new immutable JSON report without overwriting prior evidence."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First deterministic reanalysis directory")
    parser.add_argument("second", type=Path, help="Second deterministic reanalysis directory")
    parser.add_argument("--output", type=Path, required=True, help="New comparison JSON report")
    args = parser.parse_args()
    try:
        report = compare_reanalyses(args.first, args.second)
        write_report(report, args.output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "status": "failed",
            "artifact_role": "deterministic_reanalysis_repeatability",
            "errors": [f"comparison could not complete: {type(exc).__name__}: {exc}"],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
