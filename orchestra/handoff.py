"""Checkpoint handoff bundles for mid-task model switching."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .prompts import load_prompt

_PLAN_CAP = 4000
_DIFF_CAP = 8000
_TEST_LINES = 40


@dataclass
class HandoffBundle:
    task_id: str
    phase: str
    tier: str
    goal: str
    plan_digest: str = ""
    diff_stat: str = ""
    diff_excerpt: str = ""
    last_test_tail: str = ""
    open_items: list[str] = field(default_factory=list)
    source_model: str = ""
    source_tmux: str = ""
    native_session_id: str | None = None
    resume_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffBundle:
        return cls(
            task_id=data.get("task_id", ""),
            phase=data.get("phase", ""),
            tier=data.get("tier", "T2"),
            goal=data.get("goal", ""),
            plan_digest=data.get("plan_digest", ""),
            diff_stat=data.get("diff_stat", ""),
            diff_excerpt=data.get("diff_excerpt", ""),
            last_test_tail=data.get("last_test_tail", ""),
            open_items=list(data.get("open_items", [])),
            source_model=data.get("source_model", ""),
            source_tmux=data.get("source_tmux", ""),
            native_session_id=data.get("native_session_id"),
            resume_hint=data.get("resume_hint", ""),
        )


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def git_diff_stat(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )
    return (result.stdout or "").strip()


def git_diff_excerpt(workspace: Path, limit: int = _DIFF_CAP) -> str:
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            cwd=str(workspace),
        )
    return _cap(result.stdout or "", limit)


def build_handoff_bundle(
    state_data: dict[str, Any],
    workspace: Path,
    source_model: str,
    source_tmux: str = "",
    native_session_id: str | None = None,
    resume_hint: str = "",
) -> HandoffBundle:
    """Build a handoff bundle from task state and workspace."""
    verification = state_data.get("verification_result") or {}
    tests = verification.get("tests") or {}
    test_out = tests.get("output", "") or ""
    test_lines = test_out.splitlines()[-_TEST_LINES:]
    plan = state_data.get("final_plan") or state_data.get("synthesis_output") or ""
    return HandoffBundle(
        task_id=state_data.get("task_id", ""),
        phase=state_data.get("phase", ""),
        tier=state_data.get("tier") or "T2",
        goal=state_data.get("task_description", ""),
        plan_digest=_cap(plan, _PLAN_CAP),
        diff_stat=git_diff_stat(workspace),
        diff_excerpt=git_diff_excerpt(workspace),
        last_test_tail="\n".join(test_lines),
        open_items=list(state_data.get("open_items", [])),
        source_model=source_model,
        source_tmux=source_tmux,
        native_session_id=native_session_id or state_data.get("native_session_id"),
        resume_hint=resume_hint or f"Continue {state_data.get('phase', '')} without re-planning.",
    )


def save_handoff(bundle: HandoffBundle, state_dir: Path) -> Path:
    path = state_dir / f"{bundle.task_id}-handoff.json"
    path.write_text(json.dumps(bundle.to_dict(), indent=2))
    return path


def load_handoff(state_dir: Path, task_id: str) -> HandoffBundle | None:
    path = state_dir / f"{task_id}-handoff.json"
    if not path.exists():
        return None
    return HandoffBundle.from_dict(json.loads(path.read_text()))


def render_handoff_prompt(bundle: HandoffBundle) -> str:
    """Render continuation prompt from bundle (no full transcripts)."""
    template = load_prompt("handoff_continuation")
    open_items = "\n".join(f"- {item}" for item in bundle.open_items) or "- (none)"
    return (
        template.replace("{{goal}}", bundle.goal)
        .replace("{{phase}}", bundle.phase)
        .replace("{{tier}}", bundle.tier)
        .replace("{{resume_hint}}", bundle.resume_hint)
        .replace("{{plan_digest}}", bundle.plan_digest)
        .replace("{{diff_stat}}", bundle.diff_stat)
        .replace("{{diff_excerpt}}", bundle.diff_excerpt)
        .replace("{{last_test_tail}}", bundle.last_test_tail)
        .replace("{{open_items}}", open_items)
        .replace("{{source_model}}", bundle.source_model)
    )
