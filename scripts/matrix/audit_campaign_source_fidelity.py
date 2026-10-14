#!/usr/bin/env python3
"""Compare a campaign's recorded source hashes with the current repository tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _repo_root(campaign_dir: Path) -> Path:
    for parent in (campaign_dir, *campaign_dir.parents):
        if (parent / ".git").exists():
            return parent
    raise ValueError(f"No Git repository found above campaign directory: {campaign_dir}")


def build_audit(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    root = _repo_root(campaign_dir)
    attempts = sorted(
        (campaign_dir / "runs" / "attempts").glob("*/attempt*/attempt_provenance.json")
    )
    if not attempts:
        raise ValueError(f"No attempt_provenance.json files found under {campaign_dir}")

    attempt_rows: list[dict[str, Any]] = []
    inventories: Counter[str] = Counter()
    inventory_by_digest: dict[str, dict[str, str]] = {}
    for path in attempts:
        try:
            provenance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            attempt_rows.append({"attempt_path": str(path), "error": str(exc)})
            continue
        identity = provenance.get("identity", {})
        source_hashes = identity.get("source_hashes", {}) if isinstance(identity, dict) else {}
        if not isinstance(source_hashes, dict) or not source_hashes:
            attempt_rows.append({"attempt_path": str(path), "error": "source_hashes missing"})
            continue
        hashes = {str(key): str(value) for key, value in source_hashes.items()}
        inventory_digest = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        inventories[inventory_digest] += 1
        inventory_by_digest[inventory_digest] = hashes
        attempt_rows.append(
            {
                "attempt_path": str(path.relative_to(campaign_dir)),
                "run_key": provenance.get("run_key"),
                "git_sha": identity.get("git_sha"),
                "source_file_count": len(hashes),
                "source_inventory_sha256": inventory_digest,
            }
        )

    if not inventories:
        raise ValueError("No valid source-hash inventories found in attempt provenance")
    most_common_digest, attempt_count = inventories.most_common(1)[0]
    expected = inventory_by_digest[most_common_digest]
    matching: list[str] = []
    changed: list[dict[str, str]] = []
    missing: list[str] = []
    for relative_path, expected_hash in sorted(expected.items()):
        current_hash = _sha256(root / relative_path)
        if current_hash is None:
            missing.append(relative_path)
        elif current_hash == expected_hash:
            matching.append(relative_path)
        else:
            changed.append(
                {
                    "path": relative_path,
                    "recorded_sha256": expected_hash,
                    "current_sha256": current_hash,
                }
            )

    replay_reproducible = (
        len(inventories) == 1
        and len(attempt_rows) == len(attempts)
        and all("error" not in row for row in attempt_rows)
        and not changed
        and not missing
    )
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "campaign_dir": str(campaign_dir),
        "repository": str(root),
        "attempt_count": len(attempts),
        "valid_attempt_count": sum("error" not in row for row in attempt_rows),
        "unique_source_inventories": len(inventories),
        "most_common_inventory_attempts": attempt_count,
        "source_file_count": len(expected),
        "current_match_count": len(matching),
        "current_changed_count": len(changed),
        "current_missing_count": len(missing),
        "changed_files": changed,
        "missing_files": missing,
        "replay_reproducible_from_current_tree": replay_reproducible,
        "attempts": attempt_rows,
        "interpretation": (
            "The recorded hashes identify launch-time source files but do not archive their contents. "
            "Any changed file therefore prevents exact source replay from the current tree. "
            "Saved metrics and equity remain separately checkable against their recorded artifacts."
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Campaign source-fidelity audit",
        "",
        f"**Generated UTC:** {audit['generated_utc']}  ",
        f"**Campaign:** `{audit['campaign_dir']}`",
        "",
        "## Result",
        "",
        f"- Attempts with provenance: **{audit['valid_attempt_count']}/{audit['attempt_count']}**",
        f"- Unique recorded source inventories: **{audit['unique_source_inventories']}**",
        f"- Files in the common inventory: **{audit['source_file_count']}**",
        f"- Current files matching recorded hashes: **{audit['current_match_count']}**",
        f"- Changed since the run: **{audit['current_changed_count']}**",
        f"- Missing from the current tree: **{audit['current_missing_count']}**",
        f"- Exact replay from the current tree established: **{'yes' if audit['replay_reproducible_from_current_tree'] else 'no'}**",
        "",
        "Recorded hashes are integrity identifiers, not archived source contents. The run metrics can still be inspected and reconciled to equity; this audit does not claim a reproducible retraining or strategy result.",
        "",
        "## Changed files",
        "",
    ]
    lines.extend(f"- `{item['path']}`" for item in audit["changed_files"])
    if not audit["changed_files"]:
        lines.append("- None")
    lines.extend(["", "## Missing files", ""])
    lines.extend(f"- `{item}`" for item in audit["missing_files"])
    if not audit["missing_files"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out-stem", type=Path, default=None)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir.resolve()
    out_stem = args.out_stem or campaign_dir / "reports" / "source_fidelity_audit"
    audit = build_audit(campaign_dir)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    out_stem.with_suffix(".md").write_text(render_markdown(audit), encoding="utf-8")
    print(out_stem.with_suffix(".md"))
    print(out_stem.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
