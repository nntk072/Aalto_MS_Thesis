from pathlib import Path

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
