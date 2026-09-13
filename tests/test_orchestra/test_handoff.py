"""Tests for handoff bundles."""

from __future__ import annotations

import subprocess

from orchestra import handoff
from orchestra.handoff import HandoffBundle, _cap, render_handoff_prompt


def test_handoff_caps() -> None:
    assert len(_cap("x" * 100, 50)) < 100
    assert "[truncated]" in _cap("x" * 100, 50)


def test_render_handoff_prompt_minimal() -> None:
    bundle = HandoffBundle(
        task_id="task-1",
        phase="implementation",
        tier="T2",
        goal="fix bug",
        plan_digest="plan here",
        resume_hint="continue step 3",
        source_model="opencode/default",
    )
    text = render_handoff_prompt(bundle)
    assert "continue step 3" in text
    assert "Do not re-plan" in text.lower() or "do not re-plan" in text.lower()
    assert "fix bug" in text


def test_git_diff_helpers_are_empty_when_git_is_unavailable(tmp_path, monkeypatch) -> None:
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing_git)

    assert handoff.git_diff_stat(tmp_path) == ""
    assert handoff.git_diff_excerpt(tmp_path) == ""
