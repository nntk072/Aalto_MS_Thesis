#!/usr/bin/env python3
"""Scan Git-visible repository files for credential-like secrets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SCANNER_VERSION = "1.5.0"
DISABLED_PLUGINS = ("Base64HighEntropyString", "HexHighEntropyString")


def uv_command() -> str:
    configured = os.environ.get("UV_BIN")
    command = configured or shutil.which("uv")
    if not command:
        raise RuntimeError("uv is required to run the isolated secret scanner")
    return command


def findings_from_baseline(payload: Any) -> list[tuple[str, int, str]]:
    """Return safe finding metadata without exposing matched secret values."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError("detect-secrets returned JSON without a results mapping")
    findings: list[tuple[str, int, str]] = []
    for filename, entries in payload["results"].items():
        if not isinstance(filename, str) or not isinstance(entries, list):
            raise ValueError("detect-secrets result entries have an invalid shape")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("detect-secrets finding must be an object")
            line = entry.get("line_number")
            detector = entry.get("type")
            if not isinstance(line, int) or not isinstance(detector, str):
                raise ValueError(
                    f"detect-secrets finding in {filename} lacks safe location metadata"
                )
            findings.append((filename, line, detector))
    return findings


def main() -> int:
    command = [
        uv_command(),
        "tool",
        "run",
        "--from",
        f"detect-secrets=={SCANNER_VERSION}",
        "detect-secrets",
        "scan",
    ]
    for plugin in DISABLED_PLUGINS:
        command.extend(("--disable-plugin", plugin))
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        print(f"detect-secrets failed with exit code {result.returncode}.", file=sys.stderr)
        return result.returncode or 2
    try:
        payload = json.loads(result.stdout)
        findings = findings_from_baseline(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Could not interpret detect-secrets output: {exc}", file=sys.stderr)
        return 2
    if findings:
        print(f"Credential secret scan found {len(findings)} finding(s):", file=sys.stderr)
        for filename, line, detector in findings:
            print(f"  {filename}:{line} [{detector}]", file=sys.stderr)
        print(
            "Review findings and use an inline allowlist only for confirmed fixtures.",
            file=sys.stderr,
        )
        return 1
    print("Credential secret scan passed; entropy-only Base64 and Hex detectors are disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
