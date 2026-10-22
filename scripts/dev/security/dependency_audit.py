#!/usr/bin/env python3
"""Audit the frozen Linux runtime and development dependencies."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXCEPTIONS = ROOT / "docs/security/vulnerability_exceptions.toml"
AUDITOR_VERSION = "2.10.1"
ALLOWED_AUDITOR_EXIT_CODES = {0, 1}
ALLOWED_UNAUDITABLE_DEPENDENCIES = {("torch", "2.14.0+cu130")}


def normalize_package(name: str) -> str:
    """Normalize Python package names as specified by PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load_exceptions(path: Path, *, today: dt.date | None = None) -> set[tuple[str, str]]:
    """Load current advisory/package exceptions, rejecting invalid or expired entries."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Exception file not found: {path}", file=sys.stderr)
        raise
    entries = payload.get("exception", [])
    if not isinstance(entries, list):
        raise ValueError("'exception' must be an array of tables")

    today = today or dt.date.today()
    accepted: set[tuple[str, str]] = set()
    required = {"advisory", "package", "reason", "impact", "review_after"}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"exception {index} must be a TOML table")
        missing = sorted(required - entry.keys())
        if missing:
            raise ValueError(f"exception {index} missing required fields: {', '.join(missing)}")
        advisory = entry["advisory"]
        package = entry["package"]
        reason = entry["reason"]
        impact = entry["impact"]
        review_after = entry["review_after"]
        if not all(isinstance(value, str) and value.strip() for value in (advisory, package)):
            raise ValueError(f"exception {index} advisory and package must be nonempty strings")
        if not all(isinstance(value, str) and value.strip() for value in (reason, impact)):
            raise ValueError(f"exception {index} reason and impact must be nonempty strings")
        if isinstance(review_after, dt.datetime) or not isinstance(review_after, dt.date):
            raise ValueError(f"exception {index} review_after must be a TOML date")
        if review_after < today:
            raise ValueError(
                f"exception {index} expired on {review_after.isoformat()}: {advisory} for {package}"
            )
        key = (normalize_package(package), advisory.strip().upper())
        if key in accepted:
            raise ValueError(f"duplicate exception for {key[1]} / {key[0]}")
        accepted.add(key)
    return accepted


def findings_from_report(payload: Any) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Return (package, advisory, version) findings and explicit scanner skips."""
    if not isinstance(payload, dict) or not isinstance(payload.get("dependencies"), list):
        raise ValueError("pip-audit returned JSON without a dependencies list")
    findings: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    for dependency in payload["dependencies"]:
        if not isinstance(dependency, dict):
            raise ValueError("pip-audit dependency entry must be an object")
        name = dependency.get("name")
        version = dependency.get("version")
        if not isinstance(name, str) or not name:
            raise ValueError("pip-audit dependency entry has no package name")
        skip_reason = dependency.get("skip_reason")
        if skip_reason:
            skipped.append(f"{name}@{version or 'unknown'}: {skip_reason}")
        vulnerabilities = dependency.get("vulns", [])
        if not isinstance(vulnerabilities, list):
            raise ValueError(f"pip-audit vulnerabilities for {name} must be a list")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                raise ValueError(f"pip-audit vulnerability for {name} must be an object")
            advisory = vulnerability.get("id")
            if not isinstance(advisory, str) or not advisory.strip():
                raise ValueError(f"pip-audit vulnerability for {name} has no advisory ID")
            findings.append(
                (normalize_package(name), advisory.strip().upper(), str(version or "?"))
            )
    return findings, skipped


def unexpected_skips(skipped: list[str]) -> list[str]:
    """Return unauditable dependencies outside the explicitly reviewed allowlist."""
    unexpected: list[str] = []
    for item in skipped:
        package_version, _, reason = item.partition(": ")
        package, separator, version = package_version.partition("@")
        normalized = normalize_package(package)
        allowed = (normalized, version) in ALLOWED_UNAUDITABLE_DEPENDENCIES
        if normalized == "torch" and version in {"", "unknown"}:
            # pip-audit's requirements-file parser omits the version field but includes
            # the exact local CUDA version in its skip reason.
            allowed = "torch (2.14.0+cu130)" in reason and separator == "@"
        if not allowed:
            unexpected.append(item)
    return unexpected


def uv_command() -> str:
    configured = os.environ.get("UV_BIN")
    command = configured or shutil.which("uv")
    if not command:
        raise RuntimeError("uv is required to export the frozen lockfile")
    return command


def run_audit(*, exceptions_path: Path, report_out: Path | None = None) -> int:
    exceptions = load_exceptions(exceptions_path)
    uv = uv_command()
    with tempfile.TemporaryDirectory(prefix="thesis-security-audit-") as temp_dir:
        requirements = Path(temp_dir) / "requirements.txt"
        exported = subprocess.run(
            [
                uv,
                "export",
                "--frozen",
                "--no-hashes",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--extra",
                "dev",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if exported.returncode != 0:
            print(exported.stderr, file=sys.stderr, end="")
            print("Could not export the frozen Linux runtime+dev lock set.", file=sys.stderr)
            return exported.returncode or 2
        requirements.write_text(exported.stdout, encoding="utf-8")
        audited = subprocess.run(
            [
                uv,
                "tool",
                "run",
                "--from",
                f"pip-audit=={AUDITOR_VERSION}",
                "pip-audit",
                "--no-deps",
                "--disable-pip",
                "--format",
                "json",
                "-r",
                str(requirements),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(audited.stdout, encoding="utf-8")
    if audited.returncode not in ALLOWED_AUDITOR_EXIT_CODES:
        print(audited.stderr, file=sys.stderr, end="")
        print(f"pip-audit failed with exit code {audited.returncode}.", file=sys.stderr)
        return audited.returncode or 2
    try:
        payload = json.loads(audited.stdout)
        findings, skipped = findings_from_report(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Could not interpret pip-audit output: {exc}", file=sys.stderr)
        print(audited.stderr, file=sys.stderr, end="")
        return 2

    unresolved: list[tuple[str, str, str]] = []
    for package, advisory, version in findings:
        if (package, advisory) not in exceptions:
            unresolved.append((package, advisory, version))

    unique_findings = sorted(set((package, advisory) for package, advisory, _ in findings))
    unique_unresolved = sorted(set((package, advisory) for package, advisory, _ in unresolved))
    print(
        f"pip-audit {AUDITOR_VERSION}: {len(findings)} finding records, "
        f"{len(unique_findings)} unique package/advisory pairs, "
        f"{len({package for package, _ in unique_findings})} affected packages."
    )
    if skipped:
        for item in skipped:
            print(f"SKIPPED (not available to PyPI audit): {item}")
    unreviewed_skips = unexpected_skips(skipped)
    if unreviewed_skips:
        print("Unreviewed unauditable dependencies:", file=sys.stderr)
        for item in unreviewed_skips:
            print(f"  {item}", file=sys.stderr)
        return 2
    if unique_unresolved:
        print("Unreviewed dependency vulnerabilities:", file=sys.stderr)
        for package, advisory in unique_unresolved:
            versions = sorted(
                {
                    version
                    for found_package, found_advisory, version in unresolved
                    if found_package == package and found_advisory == advisory
                }
            )
            print(f"  {package} {','.join(versions)} {advisory}", file=sys.stderr)
        return 1
    print(f"All findings are covered by {len(exceptions)} active reviewed exceptions.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    try:
        return run_audit(exceptions_path=args.exceptions, report_out=args.report_out)
    except (OSError, RuntimeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"dependency audit setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
