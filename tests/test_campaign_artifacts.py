"""Regression checks for campaign artifact admission."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/matrix/campaign_controller.py"
SPEC = importlib.util.spec_from_file_location("campaign_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


@pytest.mark.parametrize(
    "steps,checkpoint,expected",
    [
        (8192, b"checkpoint", True),
        (8192, None, False),
        (8192, b"", False),
        (0, b"checkpoint", False),
        (-1, b"checkpoint", False),
        ("8192", b"checkpoint", False),
    ],
)
def test_verify_run_artifacts(
    tmp_path: Path, steps: object, checkpoint: bytes | None, expected: bool
) -> None:
    run = tmp_path / "one" / "timestamped-run"
    run.mkdir(parents=True)
    (run / "training_log.json").write_text(
        json.dumps({"timesteps_completed": steps}), encoding="utf-8"
    )
    if checkpoint is not None:
        (run / "model").mkdir()
        (run / "model" / "ppo_final.zip").write_bytes(checkpoint)
    success, _ = controller.verify_run_artifacts(tmp_path, "one")
    assert success is expected


def test_verify_run_artifacts_invalid_json(tmp_path: Path) -> None:
    run = tmp_path / "one" / "timestamped-run"
    run.mkdir(parents=True)
    (run / "training_log.json").write_text("{")
    assert controller.verify_run_artifacts(tmp_path, "one")[0] is False
