from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.triton.resolve_campaign_continuation import resolve_handoff


def _handoff_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "outputs/gpu_campaigns/parent"
    (source / "reports").mkdir(parents=True)
    (source / "campaign.json").write_text(json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")
    spec = repo / "config/campaigns/child.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "campaign_id: child_recovery\nvariants: [variant]\nseeds: [50]\n", encoding="utf-8"
    )
    checkpoint_map = repo / "outputs/gpu_campaigns/child_recovery/initial_checkpoints.json"
    checkpoint_map.parent.mkdir(parents=True)
    checkpoint_map.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoints": [
                    {
                        "key": "variant__seed50",
                        "source_campaign_id": "parent",
                        "source_campaign_fingerprint": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    handoff = source / "reports/continuation_handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema": "aalto.campaign_continuation_handoff.v1",
                "source_campaign_id": "parent",
                "source_campaign_fingerprint": "a" * 64,
                "target_campaign_id": "child_recovery",
                "campaign_spec": "config/campaigns/child.yaml",
                "checkpoint_map": "outputs/gpu_campaigns/child_recovery/initial_checkpoints.json",
            }
        ),
        encoding="utf-8",
    )
    return repo, source, handoff


def test_resolve_handoff_requires_valid_source_and_complete_target_map(tmp_path: Path) -> None:
    repo, source, handoff = _handoff_fixture(tmp_path)

    assert resolve_handoff(repo, source, handoff) == (
        "child_recovery",
        "config/campaigns/child.yaml",
        str(source.resolve()),
    )

    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["source_campaign_fingerprint"] = "b" * 64
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source campaign fingerprint"):
        resolve_handoff(repo, source, handoff)
