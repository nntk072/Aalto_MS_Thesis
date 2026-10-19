#!/usr/bin/env python3
"""Validate thesis result-manifest structure, references, and artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "docs/thesis/result_manifest.schema.json"
DEFAULT_TRIAL_LEDGER_SCHEMA = ROOT / "docs/thesis/trial_ledger.schema.json"
FROZEN_SUPPORTING_ROLES = {
    "protocol",
    "trial_ledger",
    "results_summary",
    "reproduction_notes",
    "source_hash_manifest",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "role"} <= value.keys():
            yield value
        for child in value.values():
            yield from _artifact_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_records(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(path_value: str, repo_root: Path) -> Path | None:
    relative = Path(path_value)
    if relative.is_absolute():
        return None
    root = repo_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def validate_result_manifest(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    *,
    repo_root: Path = ROOT,
    verify_artifact_hashes: bool = True,
) -> list[str]:
    """Return schema, cross-reference, and optional on-disk integrity errors."""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(
        validator.iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    errors = [
        f"schema at {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in schema_errors
    ]
    if schema_errors:
        return errors

    runs = manifest["runs"]
    comparisons = manifest["comparisons"]
    claims = manifest["claims"]
    run_ids = [run["run_id"] for run in runs]
    comparison_ids = [comparison["comparison_id"] for comparison in comparisons]
    claim_ids = [claim["claim_id"] for claim in claims]
    for label, values in (
        ("run_id", run_ids),
        ("comparison_id", comparison_ids),
        ("claim_id", claim_ids),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label} values")
    known_runs = set(run_ids)
    known_comparisons = set(comparison_ids)
    for comparison in comparisons:
        for run_id in comparison["run_ids"]:
            if run_id not in known_runs:
                errors.append(
                    f"comparison {comparison['comparison_id']} references unknown run {run_id}"
                )
    for claim in claims:
        for comparison_id in claim["comparison_ids"]:
            if comparison_id not in known_comparisons:
                errors.append(
                    f"claim {claim['claim_id']} references unknown comparison {comparison_id}"
                )

    if manifest["manifest_status"] == "frozen":
        if not claims or not comparisons or not runs:
            errors.append("frozen manifest must contain claims, comparisons, and runs")
        present_roles = {item["role"] for item in manifest["supporting_artifacts"]}
        missing_roles = sorted(FROZEN_SUPPORTING_ROLES - present_roles)
        if missing_roles:
            errors.append(f"frozen manifest lacks supporting artifact roles: {missing_roles}")
        if not manifest.get("frozen_at_utc"):
            errors.append("frozen manifest lacks frozen_at_utc")
        review = manifest.get("review_disposition", {})
        if review.get("status") not in {"resolved", "resolved_with_disclosures"}:
            errors.append("frozen manifest lacks a resolved review disposition")
        if not review.get("reviewer") or not review.get("artifact"):
            errors.append("frozen manifest lacks reviewer identity or review artifact")
        if any(not claim["comparison_ids"] for claim in claims):
            errors.append("frozen manifest contains a claim with no comparison")

    supporting_by_path: dict[str, list[dict[str, Any]]] = {}
    for artifact in manifest["supporting_artifacts"]:
        supporting_by_path.setdefault(artifact["path"], []).append(artifact)
    for run in runs:
        identity = run["identity"]
        source_manifest_path = identity.get("source_hash_manifest_path")
        if manifest["manifest_status"] == "frozen" and not source_manifest_path:
            errors.append(f"frozen run {run['run_id']} lacks source hash manifest path")
        if source_manifest_path:
            matches = [
                artifact
                for artifact in supporting_by_path.get(source_manifest_path, [])
                if artifact["role"] == "source_hash_manifest"
                and artifact["sha256"] == identity["source_hash_manifest_sha256"]
            ]
            if not matches:
                errors.append(
                    f"run {run['run_id']} source hash manifest path/digest does not match "
                    "a supporting source_hash_manifest artifact"
                )

    if verify_artifact_hashes:
        root = repo_root.resolve()
        for artifact in _artifact_records(manifest):
            relative = artifact["path"]
            path = _contained_path(relative, root)
            if path is None:
                errors.append(f"artifact path is absolute or escapes repository root: {relative}")
                continue
            if not path.is_file():
                errors.append(f"artifact is missing or not a file: {relative}")
                continue
            actual = _sha256(path)
            if actual != artifact["sha256"]:
                errors.append(f"artifact SHA-256 mismatch: {relative}")

        trial_ledger_artifacts = [
            item for item in manifest["supporting_artifacts"] if item["role"] == "trial_ledger"
        ]
        if len(trial_ledger_artifacts) > 1:
            errors.append("manifest has multiple trial_ledger supporting artifacts")
        if len(trial_ledger_artifacts) == 1:
            ledger_path = _contained_path(trial_ledger_artifacts[0]["path"], root)
            if ledger_path is not None and ledger_path.is_file():
                try:
                    ledger = _load_json(ledger_path)
                    ledger_schema = _load_json(DEFAULT_TRIAL_LEDGER_SCHEMA)
                    Draft202012Validator.check_schema(ledger_schema)
                    ledger_errors = sorted(
                        Draft202012Validator(
                            ledger_schema, format_checker=FormatChecker()
                        ).iter_errors(ledger),
                        key=lambda error: tuple(str(part) for part in error.absolute_path),
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"trial ledger could not be read: {exc}")
                else:
                    if ledger_errors:
                        errors.append("linked trial ledger fails its schema")
                    else:
                        ledger_runs = {
                            (record["campaign_id"], record["run_id"])
                            for record in ledger["records"]
                            if record["record_type"] == "run"
                        }
                        for run in runs:
                            key = (run["campaign_id"], run["run_id"])
                            if key not in ledger_runs:
                                errors.append(
                                    "result run is absent from linked trial ledger: "
                                    f"{run['campaign_id']}/{run['run_id']}"
                                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--skip-artifact-hashes", action="store_true")
    args = parser.parse_args()

    errors = validate_result_manifest(
        _load_json(args.manifest),
        _load_json(args.schema),
        repo_root=args.repo_root,
        verify_artifact_hashes=not args.skip_artifact_hashes,
    )
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
