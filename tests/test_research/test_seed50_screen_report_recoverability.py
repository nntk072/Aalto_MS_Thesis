import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import scripts.research.audit_seed50_screen_report_recoverability as audit
from scripts.research.audit_seed50_screen_report_recoverability import (
    mainplan_checkpoints,
)


def test_mainplan_checkpoint_inventory_handles_nested_run_directories(
    tmp_path: Path,
) -> None:
    checkpoint = (
        tmp_path
        / "runs"
        / "attempts"
        / "variant"
        / "attempt1"
        / "variant"
        / "timestamped_run"
        / "model"
        / "ppo_ckpt_1000000_steps.zip"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert mainplan_checkpoints(tmp_path) == [checkpoint]


def test_source_state_records_head_and_dirty_paths() -> None:
    responses = iter(("abc123\n", " M tracked.py\n?? untracked.py\n"))
    with patch(
        "scripts.research.audit_seed50_screen_report_recoverability.subprocess.check_output",
        side_effect=responses,
    ):
        state = audit.source_state()

    assert state == {
        "head": "abc123",
        "dirty_paths": [" M tracked.py", "?? untracked.py"],
    }


def test_build_report_accounts_for_missing_runs_and_matching_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = tmp_path / "outputs"
    recovery_dir = outputs / "recovery"
    source_dir = outputs / "source"
    mainplan_dir = outputs / "mainplan"
    summary_path = recovery_dir / "reports" / "campaign_results_summary_v2.json"
    campaign_path = recovery_dir / "campaign.json"
    report_only_archive = outputs / "restored-checkpoint.zip"
    archive_bytes = b"checkpoint archive fixture"
    outputs.mkdir()
    report_only_archive.write_bytes(archive_bytes)
    source_hashes = {
        f"variant_{index}": hashlib.sha256(f"source-{index}".encode()).hexdigest()
        for index in range(18)
    }
    source_hashes["variant_0"] = hashlib.sha256(archive_bytes).hexdigest()
    final_hashes = {
        f"variant_{index}": hashlib.sha256(f"final-{index}".encode()).hexdigest()
        for index in range(18)
    }
    campaign = {
        "initial_checkpoint_lineage": {
            key: {"checkpoint_sha256": digest} for key, digest in source_hashes.items()
        },
        "jobs": [
            {
                "key": key,
                "artifact_hashes": {"model/ppo_final.zip": final_hashes[key]},
            }
            for key in source_hashes
        ],
    }
    summary = {
        "campaign_dir": str(recovery_dir),
        "jobs": [
            {
                "variant": key,
                "controller_status": "succeeded",
                "run_artifact_dir": f"runs/{key}",
                "train_metrics": {"return": 0.0},
                "test_metrics_report_only": {"return": 0.0},
                "evidence_sha256": {"manifest.json": "0" * 64},
            }
            for key in source_hashes
        ],
    }
    campaign_path.parent.mkdir(parents=True)
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for base in (recovery_dir, source_dir):
        manifest = base / "reports" / "replay" / "reanalysis_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}", encoding="utf-8")
    for index in range(18):
        checkpoint = (
            mainplan_dir / f"variant-{index}" / "attempt1" / "model" / audit.MAINPLAN_CHECKPOINT
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"mainplan-{index}".encode())

    monkeypatch.setattr(audit, "OUTPUTS", outputs)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "RECOVERY_DIR", recovery_dir)
    monkeypatch.setattr(audit, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(audit, "MAINPLAN_DIR", mainplan_dir)
    monkeypatch.setattr(audit, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(audit, "source_state", lambda: {"head": "test", "dirty_paths": []})

    report = audit.build_report()

    assert report["scope"]["archive_count"] == 19
    assert report["inputs"]["audit_script_sha256"] == audit.sha256(Path(audit.__file__).resolve())
    assert report["screen"]["present_1m_mainplan_checkpoints"] == 18
    assert report["screen"]["present_run_artifact_directories"] == 0
    assert report["screen"]["available_replay_manifest_count"] == 2
    assert report["checkpoint_recoverability"]["source_attempt3_hash_matches_in_outputs"] == 1
    assert report["checkpoint_recoverability"]["recovery_final_hash_matches_in_outputs"] == 0
    assert report["checkpoint_recoverability"]["disposition"].startswith("partial or exact")


def test_main_writes_report_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recoverability.json"
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit", "--out", str(output)])
    monkeypatch.setattr(
        audit,
        "build_report",
        lambda: {"checkpoint_recoverability": {"disposition": "fixture"}},
    )

    assert audit.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "checkpoint_recoverability": {"disposition": "fixture"}
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.main()
