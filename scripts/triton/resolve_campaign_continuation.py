#!/usr/bin/env python3
"""Validate a preserved campaign's explicit handoff to a checkpoint continuation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


def resolve_handoff(repo: Path, source_campaign: Path, handoff_path: Path) -> tuple[str, str, str]:
    repo = repo.resolve()
    source_campaign = source_campaign.resolve()
    handoff_path = handoff_path.resolve()
    try:
        source_campaign.relative_to(repo)
        handoff_path.relative_to(source_campaign)
    except ValueError as exc:
        raise ValueError("handoff paths must remain inside the repository/source campaign") from exc

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if (
        not isinstance(handoff, dict)
        or handoff.get("schema") != "aalto.campaign_continuation_handoff.v1"
    ):
        raise ValueError("unsupported campaign continuation handoff schema")
    source_manifest = json.loads((source_campaign / "campaign.json").read_text(encoding="utf-8"))
    if handoff.get("source_campaign_id") != source_campaign.name or handoff.get(
        "source_campaign_fingerprint"
    ) != source_manifest.get("fingerprint"):
        raise ValueError("handoff does not identify the preserved source campaign fingerprint")

    target_id = handoff.get("target_campaign_id")
    spec_rel = handoff.get("campaign_spec")
    map_rel = handoff.get("checkpoint_map")
    if (
        not isinstance(target_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target_id) is None
    ):
        raise ValueError("handoff target campaign ID is invalid")
    if not isinstance(spec_rel, str) or not isinstance(map_rel, str):
        raise ValueError("handoff campaign spec or checkpoint map is missing")
    spec_path = (repo / spec_rel).resolve()
    map_path = (repo / map_rel).resolve()
    for path in (spec_path, map_path):
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise ValueError("handoff target paths must remain inside the repository") from exc
    if not spec_path.is_file() or not map_path.is_file():
        raise ValueError("handoff target campaign spec or checkpoint map is absent")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    checkpoint_map: Any = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("campaign_id") != target_id:
        raise ValueError("target campaign spec ID does not match the handoff")
    if not isinstance(checkpoint_map, dict) or checkpoint_map.get("schema_version") != 1:
        raise ValueError("target checkpoint map has an unsupported schema")
    entries = checkpoint_map.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("target checkpoint map has no checkpoints list")
    expected_keys = {
        f"{variant}__seed{seed}"
        for variant in spec.get("variants", [])
        for seed in spec.get("seeds", [])
    }
    observed_keys = {entry.get("key") for entry in entries if isinstance(entry, dict)}
    if observed_keys != expected_keys or len(entries) != len(expected_keys):
        raise ValueError("target checkpoint map does not cover the campaign run keys exactly")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or entry.get("source_campaign_id") != source_campaign.name
            or entry.get("source_campaign_fingerprint") != source_manifest.get("fingerprint")
        ):
            raise ValueError("checkpoint map contains a different source campaign lineage")

    return target_id, spec_rel, str(source_campaign)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source-campaign", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    args = parser.parse_args()
    target_id, spec, source = resolve_handoff(args.repo, args.source_campaign, args.handoff)
    print(target_id)
    print(spec)
    print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
