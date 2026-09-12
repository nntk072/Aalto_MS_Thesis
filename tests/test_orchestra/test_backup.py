"""Tests for orchestra workspace snapshots."""

from __future__ import annotations

from orchestra.backup import create_snapshot, list_slots, resolve_slot, restore_snapshot


def test_snapshot_restore_reverts_edits_and_new_files(tmp_path) -> None:
    src = tmp_path / "quant_rl"
    src.mkdir()
    target = src / "plots.py"
    target.write_text("original\n")
    dest = tmp_path / "orchestra" / "state" / "backups" / "task-1" / "00-start"
    meta = create_snapshot(tmp_path, dest)
    assert meta["files"] >= 1
    target.write_text("changed by orchestra\n")
    (src / "extra.py").write_text("new\n")
    n = restore_snapshot(tmp_path, dest)
    assert n >= 1
    assert target.read_text() == "original\n"
    assert not (src / "extra.py").exists()


def test_per_step_slots_restore_implementation_checkpoint(tmp_path) -> None:
    src = tmp_path / "quant_rl"
    src.mkdir()
    target = src / "plots.py"
    target.write_text("before\n")
    root = tmp_path / "orchestra" / "state" / "backups" / "task-1"
    create_snapshot(tmp_path, root / "00-start")
    target.write_text("after-implement\n")
    create_snapshot(tmp_path, root / "05-implementation")
    target.write_text("after-fix\n")
    assert list_slots(root) == ["00-start", "05-implementation"]
    restore_snapshot(tmp_path, resolve_slot(root, "5"))
    assert target.read_text() == "after-implement\n"
    restore_snapshot(tmp_path, resolve_slot(root, "start"))
    assert target.read_text() == "before\n"
