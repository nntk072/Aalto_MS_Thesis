"""Regression checks for campaign artifact admission."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/campaigns/execution.py"
SPEC = importlib.util.spec_from_file_location("campaign_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_worker_command_binds_one_gpu_from_shared_allocation() -> None:
    command = controller.worker_command(
        ["bash", "scripts/research/frozen-launches/train_one_variant.sh"],
        node="gpuarm2",
        gpu_type="gh200",
        cpus=7,
        gpu_count=2,
        gpu_id=1,
        step_cpus=48,
    )

    assert command[:9] == [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist=gpuarm2",
        "--cpus-per-task=48",
        "--gpus=gh200:2",
        "--gpu-bind=map_gpu:1",
        "--cpu-bind=cores",
    ]
    assert command[9:11] == ["--chdir", str(controller.ROOT)]


def test_worker_command_uses_explicit_cpu_mask_when_provided() -> None:
    command = controller.worker_command(
        ["python", "train.py"],
        node="gpuarm1",
        gpu_type="gh200",
        cpus=5,
        cpu_mask="0x3e0",
        expected_cpu_ids=list(range(5, 10)),
        step_cpus=48,
    )

    assert "--cpu-bind=mask_cpu:0x3e0" in command
    assert "--exact" not in command
    assert "--expected-cpus" in command
    assert ",".join(str(cpu) for cpu in range(5, 10)) in command
    assert command[-2:] == ["python", "train.py"]


def test_worker_step_envelope_must_cover_the_cpu_mask() -> None:
    with pytest.raises(ValueError, match="cannot be smaller"):
        controller.worker_command(
            ["python", "train.py"],
            node="gpuarm1",
            gpu_type="gh200",
            cpus=5,
            step_cpus=4,
        )


def test_worker_cpu_masks_are_disjoint_and_reject_oversubscription() -> None:
    cpu_ids = list(range(48))
    masks = [controller.cpu_mask_for_slot(cpu_ids, slot, 5) for slot in range(9)]

    assert all(len(ids) == 5 for _, ids in masks)
    assert len({cpu for _, ids in masks for cpu in ids}) == 45
    assert masks[0][0] == "0x1f"
    assert masks[1][0] == "0x3e0"
    assert controller.cpu_mask_for_slot(cpu_ids, 45, 1)[1] == [45]
    with pytest.raises(ValueError, match="only 3 remain"):
        controller.cpu_mask_for_slot(cpu_ids, 9, 5)


def test_resource_sampler_does_not_reserve_a_gpu() -> None:
    command = controller.worker_command(
        ["python", "sample.py"],
        node="gpuarm1",
        gpu_type="gh200",
        cpus=1,
        gpu_count=0,
    )
    assert not any(argument.startswith("--gpus=") for argument in command)
    assert not any(argument.startswith("--gpu-bind=") for argument in command)


def test_action_space_width_validation_supports_legacy_discrete_and_box() -> None:
    assert controller.action_space_matches_width(SimpleNamespace(shape=(), n=20), 20)
    assert controller.action_space_matches_width(SimpleNamespace(shape=(), n=np.int64(20)), 20)
    assert controller.action_space_matches_width(SimpleNamespace(shape=(5,)), 5)
    assert not controller.action_space_matches_width(SimpleNamespace(shape=(), n=19), 20)
    assert not controller.action_space_matches_width(SimpleNamespace(shape=(4,)), 5)


def test_campaign_source_snapshot_archives_and_verifies_exact_hashed_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    source_dir = repository / "quant_rl"
    source_dir.mkdir(parents=True)
    source = source_dir / "trainer.py"
    source.write_text("value = 1\n", encoding="utf-8")
    sources = [source]
    expected_hashes = {"quant_rl/trainer.py": controller.sha256_file(source)}
    archive_path = tmp_path / "campaign" / "source_snapshot.zip"

    record = controller.write_campaign_source_snapshot(
        repository,
        sources,
        archive_path,
        git_sha="a" * 40,
        expected_source_hashes=expected_hashes,
    )

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("quant_rl/trainer.py") == b"value = 1\n"
        snapshot_manifest = json.loads(archive.read("source_manifest.json"))
    assert snapshot_manifest == {
        "schema_version": 1,
        "git_sha": "a" * 40,
        "files": expected_hashes,
    }
    assert record["file_count"] == 1
    assert controller.verify_campaign_source_snapshot(archive_path, record) == (True, "verified")

    duplicate_path = tmp_path / "campaign-copy" / "source_snapshot.zip"
    duplicate_record = controller.write_campaign_source_snapshot(
        repository,
        sources,
        duplicate_path,
        git_sha="a" * 40,
        expected_source_hashes=expected_hashes,
    )
    assert duplicate_record["sha256"] == record["sha256"]

    source.write_text("value = 2\n", encoding="utf-8")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("quant_rl/trainer.py") == b"value = 1\n"

    archive_bytes = archive_path.read_bytes()
    archive_path.write_bytes(archive_bytes[:-1] + bytes([archive_bytes[-1] ^ 1]))
    valid, reason = controller.verify_campaign_source_snapshot(archive_path, record)
    assert not valid
    assert reason == "source snapshot archive hash mismatch"


def test_campaign_source_snapshot_fails_if_source_changes_during_snapshot(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "trainer.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed while creating"):
        controller.write_campaign_source_snapshot(
            repository,
            [source],
            tmp_path / "campaign" / "source_snapshot.zip",
            git_sha="a" * 40,
            expected_source_hashes={"trainer.py": "b" * 64},
        )


@pytest.mark.parametrize(
    "steps,checkpoint,train_ok,gate_reason,expected",
    [
        (8192, b"checkpoint", True, None, True),
        (8192, None, True, None, False),
        (8192, b"", True, None, False),
        (0, b"checkpoint", True, None, False),
        (-1, b"checkpoint", True, None, False),
        ("8192", b"checkpoint", True, None, False),
        (8192, b"checkpoint", False, "peak_trailing_dd,max_loss_stopped_trading", False),
    ],
)
def test_verify_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: object,
    checkpoint: bytes | None,
    train_ok: bool,
    gate_reason: str | None,
    expected: bool,
) -> None:
    run = tmp_path / "one" / "timestamped-run"
    run.mkdir(parents=True)
    config_text = '{"env":{"normalize":true}}\n'
    (run / "config.yaml").write_text(config_text, encoding="utf-8")
    cache_path = tmp_path / "feature-cache.parquet"
    cache_path.write_bytes(b"verified feature cache bytes")
    cache_hash = "a" * 16
    Path(str(cache_path) + ".hash").write_text(cache_hash + "\n", encoding="utf-8")
    (tmp_path / "one" / "resolved_config.json").write_text(
        json.dumps({"variant": "variant", "seed": 7, "steps": 8192}), encoding="utf-8"
    )
    identity = {"jobs": ["variant__seed7"], "steps": 8192}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (tmp_path / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "run_key": "variant__seed7",
                "variant": "variant",
                "seed": 7,
                "requested_timesteps": 8192,
                "attempt": 1,
                "campaign_fingerprint": fingerprint,
                "identity": identity,
            }
        ),
        encoding="utf-8",
    )
    feature_columns = ["close", "volume"]
    payload = {
        "timesteps_completed": steps,
        "timesteps": 8192,
        "seed": 7,
        "algo": "ppo",
        "test_sharpe": 0.5,
        "test_max_dd": 0.1,
        "test_trades": 12,
        "test_return": 0.2,
        "test_survived_full_year": True,
        "train_equity_ok": train_ok,
        "feature_cache_hash": cache_hash,
        "feature_cache_name": cache_path.name,
        "run_manifests": ["run_manifest.json"],
        "feature_columns": feature_columns,
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(feature_columns, separators=(",", ":")).encode()
        ).hexdigest(),
        "effective_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "action_space_width": 2,
        "observation_space_shapes": {"observation": [3]},
        "env_flags": {"normalize": True},
    }
    (run / "training_log.json").write_text(json.dumps(payload), encoding="utf-8")
    if gate_reason is not None:
        (run / "train_equity_gate.json").write_text(
            json.dumps({"ok": False, "reason": gate_reason}), encoding="utf-8"
        )
    if checkpoint is not None:
        (run / "model").mkdir()
        model_file = run / "model" / "ppo_final.zip"
        if checkpoint:
            with zipfile.ZipFile(model_file, "w") as archive:
                archive.writestr("checkpoint", checkpoint)
        else:
            model_file.write_bytes(checkpoint)
    model_file = run / "model" / "ppo_final.zip"
    checkpoint_hash = (
        hashlib.sha256(model_file.read_bytes()).hexdigest() if model_file.is_file() else "b" * 64
    )
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "feature_cache": {
                    "path": str(cache_path),
                    "identity": cache_path.name,
                    "sha256": cache_hash,
                    "file_sha256": cache_sha256,
                },
                "checkpoint": {"sha256": checkpoint_hash},
                "config": {"sha256": hashlib.sha256(config_text.encode()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    fake_model = SimpleNamespace(
        num_timesteps=8192,
        action_space=SimpleNamespace(shape=(2,)),
        observation_space=SimpleNamespace(spaces={"observation": SimpleNamespace(shape=(3,))}),
    )
    from stable_baselines3 import PPO

    monkeypatch.setattr(PPO, "load", staticmethod(lambda *args, **kwargs: fake_model))
    success, reason = controller.verify_run_artifacts(
        tmp_path,
        "variant__seed7",
        8192,
        expected_fingerprint=fingerprint,
        expected_attempt=1,
    )
    assert success is expected
    if gate_reason is not None:
        assert reason == f"training risk-survival gate failed: {gate_reason}"
    if expected:
        Path(str(cache_path) + ".hash").write_text("c" * 16 + "\n", encoding="utf-8")
        assert not controller.verify_run_artifacts(
            tmp_path,
            "variant__seed7",
            8192,
            expected_fingerprint=fingerprint,
            expected_attempt=1,
        )[0]
        Path(str(cache_path) + ".hash").write_text(cache_hash + "\n", encoding="utf-8")
        cache_path.write_bytes(b"tampered cache bytes with same sidecar")
        assert not controller.verify_run_artifacts(
            tmp_path,
            "variant__seed7",
            8192,
            expected_fingerprint=fingerprint,
            expected_attempt=1,
        )[0]


def test_verify_run_artifacts_invalid_json(tmp_path: Path) -> None:
    run = tmp_path / "one" / "timestamped-run"
    run.mkdir(parents=True)
    (run / "training_log.json").write_text("{")
    assert (
        controller.verify_run_artifacts(
            tmp_path,
            "variant__seed7",
            8192,
            expected_fingerprint="b" * 64,
            expected_attempt=1,
        )[0]
        is False
    )


def test_verify_run_artifacts_rejects_attempt_provenance_mismatch(tmp_path: Path) -> None:
    (tmp_path / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "run_key": "variant__seed7",
                "variant": "variant",
                "seed": 7,
                "requested_timesteps": 8192,
                "attempt": 1,
                "campaign_fingerprint": "a" * 64,
                "identity": {"jobs": ["variant__seed7"], "steps": 8192},
            }
        ),
        encoding="utf-8",
    )
    success, reason = controller.verify_run_artifacts(
        tmp_path,
        "variant__seed7",
        8192,
        expected_fingerprint="b" * 64,
        expected_attempt=1,
    )
    assert success is False
    assert "attempt provenance" in reason


def test_verify_run_artifacts_rejects_identity_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "run_key": "variant__seed7",
                "variant": "variant",
                "seed": 7,
                "requested_timesteps": 8192,
                "attempt": 1,
                "campaign_fingerprint": "a" * 64,
                "identity": {"jobs": ["variant__seed7"], "steps": 8192},
            }
        ),
        encoding="utf-8",
    )
    success, reason = controller.verify_run_artifacts(
        tmp_path,
        "variant__seed7",
        8192,
        expected_fingerprint="a" * 64,
        expected_attempt=1,
    )
    assert success is False
    assert "identity does not match" in reason


def test_failed_attempt_artifacts_are_hashed_and_remain_terminal_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "config" / "base").mkdir(parents=True)
    (root / "config" / "campaigns").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "config" / "base" / "default.yaml").write_text(
        "data:\n  split:\n    train_end: '2025-12-31'\n    test_start: '2026-01-01'\n"
        "  raw_dir: data\n  m1_files: {}\n  tick_files: {}\n",
        encoding="utf-8",
    )
    (root / "config" / "experiments.yaml").write_text(
        "variants:\n  - name: test_variant\n", encoding="utf-8"
    )
    (root / "config" / "campaigns" / "variant_list.txt").write_text(
        "test_variant\n", encoding="utf-8"
    )
    spec_path = root / "config" / "campaigns" / "test.yaml"
    spec_path.write_text(
        "schema_version: 1\n"
        "campaign_id: test_failed_attempt_resume\n"
        "variant_file: config/campaigns/variant_list.txt\n"
        "variants: [test_variant]\n"
        "seeds: [7]\n"
        "timesteps: 2\n"
        "data_source: frames\n"
        "split:\n  train_end: '2025-12-31'\n  test_start: '2026-01-01'\n"
        "evaluation:\n  locked_oos_role: final_report_only\n  selection_allowed: false\n  deterministic: true\n",
        encoding="utf-8",
    )
    worker = root / "fake_worker.py"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import os\nfrom pathlib import Path\nimport sys\n"
        "out = Path(os.environ['OUT_DIR'])\n"
        "(out / 'training_log.json').write_text('{\\\"attempt\\\":true}')\n"
        "(out / 'model').mkdir()\n"
        "(out / 'model' / 'ppo_final.zip').write_bytes(b'preserved diagnostic bytes')\n"
        "sys.exit(int((Path(__file__).parent.parent / 'worker_exit_code').read_text()))\n",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    exit_code = root.parent / "worker_exit_code"
    exit_code.write_text("7", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Codex Test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "codex-test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

    monkeypatch.setattr(controller, "ROOT", root)
    monkeypatch.setattr(
        controller,
        "available_resources",
        lambda: {"cpu_affinity": 16, "cpu_ids": list(range(16)), "memory_limit_bytes": 0},
    )
    monkeypatch.setattr(
        controller,
        "runtime_metadata",
        lambda: {"python": "3.12", "machine": "x86_64", "packages": {}, "gpu": "test"},
    )
    monkeypatch.setattr(controller.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["campaign_controller.py", "--campaign-spec", str(spec_path), "--test-worker", str(worker)],
    )
    setattr(controller, "STOP", False)
    assert controller.main() == 1

    out = root / "outputs" / "gpu_campaigns" / "test_failed_attempt_resume"
    assert (out / "reports" / "campaign_results_summary.json").is_file()
    assert (out / "reports" / "campaign_results_summary.md").is_file()
    first_manifest = controller.read_manifest(out / "campaign.json")
    source_snapshot = first_manifest["source_snapshot"]
    snapshot_valid, snapshot_reason = controller.verify_campaign_source_snapshot(
        out / source_snapshot["path"], source_snapshot
    )
    assert snapshot_valid, snapshot_reason
    first = first_manifest["jobs"][0]
    assert first["status"] == "failed"
    assert first["exit_code"] == 7
    assert first["attempts"] == 1
    assert (
        first["artifact_hashes"]["training_log.json"]
        == hashlib.sha256(b'{"attempt":true}').hexdigest()
    )
    assert "model/ppo_final.zip" in first["artifact_hashes"]
    failure = json.loads((out / "logs" / "failures" / "summary.jsonl").read_text().splitlines()[0])
    assert "worker exited with code 7" in failure["reason"]
    assert failure["artifact_hashes"] == first["artifact_hashes"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_controller.py",
            "--campaign-spec",
            str(spec_path),
            "--test-worker",
            str(worker),
            "--resume",
        ],
    )
    setattr(controller, "STOP", False)
    assert controller.main() == 1
    resumed = controller.read_manifest(out / "campaign.json")["jobs"][0]
    assert resumed["status"] == "failed"
    assert resumed["attempts"] == 1
    assert (
        out / "runs" / "attempts" / "test_variant__seed7" / "attempt1" / "attempt_provenance.json"
    ).is_file()
    assert not (out / "runs" / "attempts" / "test_variant__seed7" / "attempt2").exists()

    snapshot_path = out / source_snapshot["path"]
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_path.write_bytes(snapshot_bytes[:-1] + bytes([snapshot_bytes[-1] ^ 1]))
    with pytest.raises(RuntimeError, match="campaign source snapshot invalid"):
        controller.main()


def test_latest_resumable_checkpoint_requires_matching_attempt_and_fingerprint(
    tmp_path: Path,
) -> None:
    key = "variant__seed7"
    fingerprint = "a" * 64
    attempt = tmp_path / "runs" / "attempts" / key / "attempt2"
    model = attempt / "model"
    model.mkdir(parents=True)
    (attempt / "attempt_provenance.json").write_text(
        json.dumps(
            {
                "campaign_fingerprint": fingerprint,
                "run_key": key,
                "attempt": 2,
            }
        ),
        encoding="utf-8",
    )
    for steps in (1_000_000, 3_000_000):
        with zipfile.ZipFile(model / f"ppo_ckpt_{steps}_steps.zip", "w") as archive:
            archive.writestr("checkpoint", str(steps))

    checkpoint = controller.latest_resumable_checkpoint(tmp_path, key, 2, fingerprint)

    assert checkpoint == model / "ppo_ckpt_3000000_steps.zip"
    assert controller.latest_resumable_checkpoint(tmp_path, key, 2, "b" * 64) is None


def test_campaign_source_files_include_untracked_runtime_modules(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    tracked = root / "configuration" / "tracked.py"
    untracked = root / "configuration" / "runtime_module.py"
    documentation = root / "docs" / "mutable_status.md"
    tracked.parent.mkdir()
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    untracked.write_text("VALUE = 2\n", encoding="utf-8")
    documentation.parent.mkdir()
    documentation.write_text("status updates do not alter runtime identity\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "configuration/tracked.py"], check=True)

    sources = controller.campaign_source_files(root)

    assert tracked.resolve() in sources
    assert untracked.resolve() in sources
    assert documentation.resolve() not in sources


def test_campaign_evaluation_requires_exposure_before_selection() -> None:
    evaluation = {
        "locked_oos_role": "selection_acknowledged",
        "locked_oos_exposure": "previously_inspected_holdout",
        "selection_allowed": True,
        "deterministic": True,
    }

    assert controller.validate_campaign_evaluation(evaluation)["selection_allowed"] is True


def test_campaign_evaluation_rejects_selection_on_unexposed_holdout() -> None:
    evaluation = {
        "locked_oos_role": "selection_acknowledged",
        "selection_allowed": True,
        "deterministic": True,
    }

    with pytest.raises(ValueError, match="only on explicitly exposed"):
        controller.validate_campaign_evaluation(evaluation)
