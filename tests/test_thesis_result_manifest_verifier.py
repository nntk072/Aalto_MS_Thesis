"""Tests for the thesis-facing result-manifest verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from scripts.research.thesis.evidence.verify_result_manifest import validate_result_manifest


def _artifact(path: str, content: bytes) -> dict[str, str]:
    return {
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "role": "metrics",
    }


def _manifest(artifact: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "aalto.thesis_result_manifest.v1",
        "manifest_status": "draft",
        "created_at_utc": "2026-10-18T00:00:00Z",
        "frozen_at_utc": None,
        "claims": [
            {
                "claim_id": "rq1-screen",
                "rq": "RQ1",
                "statement": "Screen result remains exploratory.",
                "evidence_role": "screening",
                "allowed_interpretation": "Summary-derived descriptive evidence only.",
                "comparison_ids": ["screen"],
                "artifacts": [artifact],
                "limitations": ["Not a matched comparison."],
            }
        ],
        "comparisons": [
            {
                "comparison_id": "screen",
                "question": "What did the screen record?",
                "run_ids": ["run-1"],
                "evidence_role": "screening",
                "status": "complete",
                "limitations": ["Contaminated for selection."],
            }
        ],
        "runs": [
            {
                "run_id": "run-1",
                "campaign_id": "campaign-1",
                "variant": "variant-1",
                "seed": 50,
                "role": "selection",
                "status": "excluded",
                "identity": {
                    "git_sha": "a" * 40,
                    "source_hash_manifest_sha256": "b" * 64,
                    "config_sha256": "c" * 64,
                    "data_sha256": ["d" * 64],
                    "cache_sha256": ["e" * 64],
                    "checkpoint_sha256": None,
                },
                "artifacts": [artifact],
            }
        ],
        "supporting_artifacts": [],
        "review_disposition": {
            "status": "not_started",
            "reviewer": None,
            "reviewed_at_utc": None,
            "artifact": None,
        },
        "limitations": ["Draft only."],
    }


def _schema() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "docs/thesis/result_manifest.schema.json"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_result_manifest_verifier_accepts_valid_draft_and_artifact_hash(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_bytes(b"{}")
    manifest = _manifest(_artifact("metrics.json", b"{}"))

    assert validate_result_manifest(manifest, _schema(), repo_root=tmp_path) == []


def test_result_manifest_verifier_rejects_dangling_references(tmp_path: Path) -> None:
    manifest = _manifest(_artifact("missing.json", b"{}"))
    manifest["comparisons"][0]["run_ids"] = ["missing-run"]
    manifest["claims"][0]["comparison_ids"] = ["missing-comparison"]

    errors = validate_result_manifest(manifest, _schema(), repo_root=tmp_path)

    assert any("unknown run missing-run" in error for error in errors)
    assert any("unknown comparison missing-comparison" in error for error in errors)
    assert any("artifact is missing" in error for error in errors)


def test_result_manifest_verifier_rejects_duplicate_run_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_bytes(b"changed")
    manifest = _manifest(_artifact("metrics.json", b"expected"))
    manifest["runs"].append(dict(manifest["runs"][0]))

    errors = validate_result_manifest(manifest, _schema(), repo_root=tmp_path)

    assert "duplicate run_id values" in errors
    assert any("SHA-256 mismatch" in error for error in errors)


def test_frozen_result_manifest_requires_review_and_claim_comparisons(tmp_path: Path) -> None:
    manifest = _manifest(_artifact("missing.json", b"{}"))
    manifest["manifest_status"] = "frozen"
    manifest["frozen_at_utc"] = "2026-10-18T00:00:00Z"
    manifest["claims"][0]["comparison_ids"] = []

    errors = validate_result_manifest(
        manifest, _schema(), repo_root=tmp_path, verify_artifact_hashes=False
    )

    assert "frozen manifest lacks a resolved review disposition" in errors
    assert "frozen manifest lacks reviewer identity or review artifact" in errors
    assert "frozen manifest contains a claim with no comparison" in errors


def test_frozen_result_manifest_cannot_be_empty(tmp_path: Path) -> None:
    manifest = _manifest(_artifact("unused.json", b"{}"))
    manifest["manifest_status"] = "frozen"
    manifest["frozen_at_utc"] = "2026-10-18T00:00:00Z"
    manifest["claims"] = []
    manifest["comparisons"] = []
    manifest["runs"] = []
    manifest["review_disposition"] = {
        "status": "resolved",
        "reviewer": "reviewer",
        "reviewed_at_utc": "2026-10-18T00:00:00Z",
        "artifact": _artifact("review.md", b"review"),
    }

    errors = validate_result_manifest(
        manifest, _schema(), repo_root=tmp_path, verify_artifact_hashes=False
    )

    assert "frozen manifest must contain claims, comparisons, and runs" in errors
    assert any("supporting artifact roles" in error for error in errors)


def _frozen_supporting_artifacts() -> list[dict[str, str]]:
    return [
        {"path": f"{role}.txt", "sha256": "a" * 64, "role": role}
        for role in (
            "protocol",
            "trial_ledger",
            "results_summary",
            "reproduction_notes",
            "source_hash_manifest",
        )
    ]


def _ledger_record() -> dict[str, Any]:
    return {
        "record_id": "ledger-run-1",
        "record_type": "run",
        "campaign_id": "campaign-1",
        "run_id": "run-1",
        "status": "excluded",
        "evidence_role": "selection",
        "source_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "artifacts": [],
        "disposition": "Summary-derived screening record.",
        "identity_completeness": "partial",
    }


def test_frozen_manifest_requires_supporting_roles(tmp_path: Path) -> None:
    manifest = _manifest(_artifact("unused.json", b"{}"))
    manifest["manifest_status"] = "frozen"
    manifest["frozen_at_utc"] = "2026-10-18T00:00:00Z"
    manifest["supporting_artifacts"] = _frozen_supporting_artifacts()
    manifest["review_disposition"] = {
        "status": "resolved",
        "reviewer": "independent reviewer",
        "reviewed_at_utc": "2026-10-18T00:00:00Z",
        "artifact": _artifact("review.md", b"review"),
    }
    manifest["runs"][0]["identity"]["source_hash_manifest_path"] = "source_hash_manifest.txt"
    manifest["runs"][0]["identity"]["source_hash_manifest_sha256"] = "a" * 64

    assert (
        validate_result_manifest(
            manifest, _schema(), repo_root=tmp_path, verify_artifact_hashes=False
        )
        == []
    )


def test_result_manifest_checks_runs_against_linked_trial_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger = {
        "schema": "aalto.thesis_trial_ledger.v1",
        "updated_at_utc": "2026-10-18T00:00:00Z",
        "records": [_ledger_record()],
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    artifact = _artifact("ledger.json", ledger_path.read_bytes())
    artifact["role"] = "trial_ledger"
    manifest = _manifest(_artifact("ledger.json", ledger_path.read_bytes()))
    manifest["supporting_artifacts"] = [artifact]

    assert validate_result_manifest(manifest, _schema(), repo_root=tmp_path) == []

    ledger["records"] = []
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    errors = validate_result_manifest(manifest, _schema(), repo_root=tmp_path)
    assert any("absent from linked trial ledger" in error for error in errors)


def test_result_manifest_requires_source_hash_to_match_supporting_artifact() -> None:
    manifest = _manifest(_artifact("unused.json", b"{}"))
    manifest["runs"][0]["identity"]["source_hash_manifest_path"] = "source.json"
    manifest["supporting_artifacts"] = [
        {"path": "source.json", "sha256": "f" * 64, "role": "source_hash_manifest"}
    ]

    errors = validate_result_manifest(manifest, _schema(), verify_artifact_hashes=False)

    assert any("source hash manifest path/digest does not match" in error for error in errors)


def test_result_manifest_rejects_artifact_path_escape(tmp_path: Path) -> None:
    manifest = _manifest(_artifact("../outside.json", b"{}"))

    errors = validate_result_manifest(manifest, _schema(), repo_root=tmp_path)

    assert any("escapes repository root" in error for error in errors)


def test_result_manifest_cli_reports_pass_and_hash_failure(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    artifact = tmp_path / "metrics.json"
    artifact.write_bytes(b"{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(_artifact("metrics.json", b"{}"))), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(repo / "scripts/research/thesis/evidence/verify_result_manifest.py"),
        str(manifest_path),
        "--repo-root",
        str(tmp_path),
    ]

    passed = subprocess.run(command, cwd=repo, capture_output=True, check=False, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "passed"

    artifact.write_bytes(b"tampered")
    failed = subprocess.run(command, cwd=repo, capture_output=True, check=False, text=True)
    assert failed.returncode == 1
    assert "SHA-256 mismatch" in json.loads(failed.stdout)["errors"][0]
