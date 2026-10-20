from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.research.campaigns.audit_campaign_source_fidelity import build_audit, render_markdown


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_source_fidelity_audit_counts_current_matches_and_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    (repository / "src").mkdir()
    (repository / "src" / "same.py").write_bytes(b"same")
    (repository / "src" / "changed.py").write_bytes(b"new")
    campaign_dir = repository / "outputs" / "campaign"
    source_hashes = {
        "src/same.py": _sha256(b"same"),
        "src/changed.py": _sha256(b"old"),
    }
    for run_key in ("run_a", "run_b"):
        attempt = campaign_dir / "runs" / "attempts" / run_key / "attempt1"
        attempt.mkdir(parents=True)
        (attempt / "attempt_provenance.json").write_text(
            json.dumps(
                {
                    "run_key": run_key,
                    "identity": {"git_sha": "abc", "source_hashes": source_hashes},
                }
            ),
            encoding="utf-8",
        )

    audit = build_audit(campaign_dir)
    report = render_markdown(audit)

    assert audit["attempt_count"] == 2
    assert audit["valid_attempt_count"] == 2
    assert audit["unique_source_inventories"] == 1
    assert audit["current_match_count"] == 1
    assert audit["current_changed_count"] == 1
    assert audit["current_missing_count"] == 0
    assert audit["replay_reproducible_from_current_tree"] is False
    assert "`src/changed.py`" in report
    assert "Exact replay from the current tree established: **no**" in report
