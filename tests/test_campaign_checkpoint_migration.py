"""Tests for checksummed checkpoint handoff between campaign identities."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.research.campaigns import audit_campaign_checkpoint_loads

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/campaigns/execution.py"
SPEC = importlib.util.spec_from_file_location("campaign_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_checkpoint_map_cli_loads_controller_from_its_own_worktree(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "experiments/campaigns/checkpoints.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Create an immutable checkpoint-lineage map" in result.stdout


def _source_campaign(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    campaign_dir = tmp_path / "parent_campaign"
    attempt_dir = campaign_dir / "runs/attempts/variant__seed42/attempt1"
    model_dir = attempt_dir / "model"
    model_dir.mkdir(parents=True)
    source_file = campaign_dir / "source.txt"
    source_file.write_text("parent source\n", encoding="utf-8")
    source_hash = controller.sha256_file(source_file)
    source_snapshot = controller.write_campaign_source_snapshot(
        campaign_dir,
        [source_file],
        campaign_dir / "source_snapshot.zip",
        git_sha="b" * 40,
        expected_source_hashes={"source.txt": source_hash},
    )
    checkpoint = model_dir / "ppo_ckpt_1000000_steps.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 1_000_000}))
    source_identity = {"source": "test fixture"}
    fingerprint = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
    provenance = {
        "campaign_fingerprint": fingerprint,
        "identity": source_identity,
        "run_key": "variant__seed42",
        "variant": "variant",
        "seed": 42,
        "attempt": 1,
        "requested_timesteps": 20_000_000,
    }
    provenance_path = attempt_dir / "attempt_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    manifest = {
        "fingerprint": fingerprint,
        "git_sha": "b" * 40,
        "timesteps": 20_000_000,
        "source_hashes": {"source.txt": source_hash},
        "source_snapshot": source_snapshot,
        "jobs": [
            {
                "key": "variant__seed42",
                "variant": "variant",
                "seed": 42,
                "status": "running",
                "attempts": 1,
            }
        ],
    }
    (campaign_dir / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")
    record = {
        "key": "variant__seed42",
        "source_campaign_dir": str(campaign_dir),
        "source_campaign_fingerprint": fingerprint,
        "source_attempt": 1,
        "source_attempt_provenance_sha256": controller.sha256_file(provenance_path),
        "source_num_timesteps": 1_000_000,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": controller.sha256_file(checkpoint),
    }
    return campaign_dir, checkpoint, record


def test_build_and_validate_initial_checkpoint_lineage(tmp_path: Path) -> None:
    campaign_dir, _checkpoint, expected_record = _source_campaign(tmp_path)

    checkpoint_map = controller.build_initial_checkpoint_map(
        [campaign_dir], {"variant__seed42"}, 20_000_000
    )
    assert checkpoint_map["schema_version"] == 1
    assert checkpoint_map["checkpoints"] == [
        {
            **expected_record,
            "source_campaign_id": campaign_dir.name,
            "source_campaign_git_sha": "b" * 40,
            "source_requested_timesteps": 20_000_000,
        }
    ]

    map_path = tmp_path / "checkpoint_map.json"
    map_path.write_text(json.dumps(checkpoint_map), encoding="utf-8")
    validated = controller.validate_initial_checkpoints(
        map_path,
        [{"key": "variant__seed42", "variant": "variant", "seed": 42}],
        20_000_000,
    )
    assert validated["variant__seed42"]["checkpoint_sha256"] == expected_record["checkpoint_sha256"]

    successor_dir = tmp_path / "successor"
    archived = controller.archive_initial_checkpoints(successor_dir, validated)
    assert (
        controller.sha256_file(archived["variant__seed42"]) == expected_record["checkpoint_sha256"]
    )
    selected, origin = controller.choose_resume_checkpoint(
        successor_dir,
        "variant__seed42",
        None,
        "e" * 64,
        archived,
    )
    assert selected == archived["variant__seed42"]
    assert origin == "parent_campaign_checkpoint"

    attempt_dir = successor_dir / "runs/attempts/variant__seed42/attempt1"
    model_dir = attempt_dir / "model"
    model_dir.mkdir(parents=True)
    newer = model_dir / "ppo_ckpt_2000000_steps.zip"
    with zipfile.ZipFile(newer, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 2_000_000}))
    (attempt_dir / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "campaign_fingerprint": "e" * 64,
                "run_key": "variant__seed42",
                "attempt": 1,
                "requested_timesteps": 20_000_000,
            }
        ),
        encoding="utf-8",
    )
    selected, origin = controller.choose_resume_checkpoint(
        successor_dir,
        "variant__seed42",
        1,
        "e" * 64,
        archived,
    )
    assert selected == newer
    assert origin == "interrupted_campaign_attempt"


def test_initial_checkpoint_validation_rejects_lineage_or_hash_mismatch(
    tmp_path: Path,
) -> None:
    campaign_dir, _checkpoint, record = _source_campaign(tmp_path)
    map_path = tmp_path / "checkpoint_map.json"
    source: dict[str, Any] = {"schema_version": 1, "checkpoints": [record]}
    expected_jobs = [{"key": "variant__seed42", "variant": "variant", "seed": 42}]

    source["checkpoints"][0] = {**record, "source_campaign_fingerprint": "c" * 64}
    map_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        controller.validate_initial_checkpoints(map_path, expected_jobs, 20_000_000)

    source["checkpoints"][0] = {**record, "checkpoint_sha256": "d" * 64}
    map_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        controller.validate_initial_checkpoints(map_path, expected_jobs, 20_000_000)

    assert campaign_dir.is_dir()


def test_initial_checkpoint_map_falls_back_to_earlier_valid_attempt(tmp_path: Path) -> None:
    campaign_dir, _checkpoint, expected_record = _source_campaign(tmp_path)
    manifest_path = campaign_dir / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["attempts"] = 2
    manifest["jobs"][0]["status"] = "interrupted"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (campaign_dir / "runs/attempts/variant__seed42/attempt2").mkdir()

    checkpoint_map = controller.build_initial_checkpoint_map(
        [campaign_dir], {"variant__seed42"}, 20_000_000
    )
    assert checkpoint_map["checkpoints"] == [
        {
            **expected_record,
            "source_campaign_id": campaign_dir.name,
            "source_campaign_git_sha": "b" * 40,
            "source_requested_timesteps": 20_000_000,
        }
    ]
    map_path = tmp_path / "checkpoint_map.json"
    map_path.write_text(json.dumps(checkpoint_map), encoding="utf-8")
    validated = controller.validate_initial_checkpoints(
        map_path,
        [{"key": "variant__seed42", "variant": "variant", "seed": 42}],
        20_000_000,
    )
    assert validated["variant__seed42"]["source_attempt"] == 1


def test_explicit_checkpoint_map_keeps_selected_attempt_after_newer_attempt_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir, _checkpoint, _record = _source_campaign(tmp_path)
    frozen_map = controller.build_initial_checkpoint_map(
        [campaign_dir], {"variant__seed42"}, 20_000_000
    )
    frozen_map.pop("schema_version")
    frozen_map_path = campaign_dir / "initial_checkpoints.json"
    frozen_map_path.write_text(json.dumps(frozen_map), encoding="utf-8")

    attempt2 = campaign_dir / "runs/attempts/variant__seed42/attempt2"
    model_dir = attempt2 / "model"
    model_dir.mkdir(parents=True)
    checkpoint2 = model_dir / "ppo_ckpt_2000000_steps.zip"
    with zipfile.ZipFile(checkpoint2, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 2_000_000}))
    source_identity = {"source": "test fixture"}
    fingerprint = hashlib.sha256(json.dumps(source_identity, sort_keys=True).encode()).hexdigest()
    provenance = {
        "campaign_fingerprint": fingerprint,
        "identity": source_identity,
        "run_key": "variant__seed42",
        "variant": "variant",
        "seed": 42,
        "attempt": 2,
        "requested_timesteps": 20_000_000,
    }
    (attempt2 / "attempt_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    manifest_path = campaign_dir / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["attempts"] = 2
    manifest["jobs"][0]["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(audit_campaign_checkpoint_loads, "ROOT", tmp_path)
    records, provenance_record = audit_campaign_checkpoint_loads._load_explicit_checkpoint_map(
        frozen_map_path,
        [{"key": "variant__seed42", "variant": "variant", "seed": 42}],
        20_000_000,
    )

    assert records["variant__seed42"]["source_attempt"] == 1
    assert records["variant__seed42"]["source_num_timesteps"] == 1_000_000
    assert provenance_record["schema_version_added"] is True
    assert provenance_record["input_sha256"] == controller.sha256_file(frozen_map_path)


def test_full_budget_checkpoint_can_be_migrated_for_policy_reevaluation(
    tmp_path: Path,
) -> None:
    campaign_dir, checkpoint, record = _source_campaign(tmp_path)
    full_checkpoint = checkpoint.with_name("ppo_ckpt_20000000_steps.zip")
    with zipfile.ZipFile(full_checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 20_000_000}))
    manifest_path = campaign_dir / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["status"] = "succeeded"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = {
        **record,
        "checkpoint_path": str(full_checkpoint),
        "checkpoint_sha256": controller.sha256_file(full_checkpoint),
        "source_num_timesteps": 20_000_000,
    }
    map_path = tmp_path / "checkpoint_map.json"
    map_path.write_text(
        json.dumps({"schema_version": 1, "checkpoints": [record]}),
        encoding="utf-8",
    )

    validated = controller.validate_initial_checkpoints(
        map_path,
        [{"key": "variant__seed42", "variant": "variant", "seed": 42}],
        20_000_000,
    )
    assert validated["variant__seed42"]["source_num_timesteps"] == 20_000_000
    assert full_checkpoint.is_file()


def test_checkpoint_selection_caps_at_the_target_timestep(tmp_path: Path) -> None:
    campaign_dir, checkpoint, _record = _source_campaign(tmp_path)
    newer = checkpoint.with_name("ppo_ckpt_3000000_steps.zip")
    with zipfile.ZipFile(newer, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 3_000_000}))
    manifest = json.loads((campaign_dir / "campaign.json").read_text(encoding="utf-8"))
    fingerprint = manifest["fingerprint"]

    selected = controller.latest_resumable_checkpoint(
        campaign_dir, "variant__seed42", 1, fingerprint, max_steps=2_000_000
    )
    assert selected == checkpoint
    latest = newer.with_name("ppo_latest.zip")
    with zipfile.ZipFile(latest, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": 3_000_000}))
    assert (
        controller.latest_resumable_checkpoint(
            campaign_dir, "variant__seed42", 1, fingerprint, max_steps=500_000
        )
        is None
    )


def test_migrated_checkpoint_model_load_checks_embedded_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePPO:
        @staticmethod
        def load(_path: Path, device: str) -> Any:
            assert device == "cpu"
            return SimpleNamespace(num_timesteps=1_000_000)

    from learning.rl import checkpoint as checkpoint_loading

    monkeypatch.setattr(
        checkpoint_loading,
        "load_sb3_checkpoint",
        lambda _path, *, device: FakePPO.load(_path, device=device),
    )
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.touch()
    loaded_steps = controller.verify_initial_checkpoint_models(
        {"variant__seed42": checkpoint},
        {"variant__seed42": {"source_num_timesteps": 1_000_000}},
    )
    assert loaded_steps == {"variant__seed42": 1_000_000}

    with pytest.raises(ValueError, match="loaded checkpoint step mismatch"):
        controller.verify_initial_checkpoint_models(
            {"variant__seed42": checkpoint},
            {"variant__seed42": {"source_num_timesteps": 2_000_000}},
        )
