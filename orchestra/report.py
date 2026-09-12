"""Assemble a markdown final report from orchestra task state."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .state import TaskState

_CLIP = 1200


def _clip(text: object, limit: int = _CLIP) -> str:
    raw = text.strip() if isinstance(text, str) else str(text or "").strip()
    if not raw:
        return "_none_"
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n... [truncated]"


def _agent_block(title: str, items: list[Any], empty: str) -> list[str]:
    if not items:
        return [f"## {title}", empty, ""]
    lines = [f"## {title}", ""]
    for i, item in enumerate(items, 1):
        model = item.get("model", "?")
        lines.append(f"### {i}. {model} ({len(item.get('output') or '')} chars)")
        lines.append("")
        lines.append(_clip(item.get("output")))
        lines.append("")
    return lines


def build_final_report(state: TaskState) -> str:
    """Render steps 1-9 plus verification into one markdown document."""
    d = state.data
    created = time.ctime(d.get("created_at") or 0)
    updated = time.ctime(d.get("updated_at") or 0)
    verify = d.get("verification_result") or {}
    tests = verify.get("tests") or {}
    lint = verify.get("lint") or {}
    types = verify.get("typecheck") or {}
    errors = d.get("errors") or []
    lines = [
        f"# Orchestra report: {state.task_id}",
        "",
        f"- **Task:** {state.task_description}",
        f"- **Phase:** {d.get('phase')}",
        f"- **Complexity:** {d.get('complexity') or 'unknown'}",
        f"- **Created:** {created}",
        f"- **Updated:** {updated}",
        f"- **Review verdict:** {d.get('review_verdict') or 'n/a'}",
        f"- **Fix iterations:** {d.get('fix_loop_count', 0)}/{d.get('fix_loop_max', 0)}",
        f"- **Backup:** {', '.join(b.get('slot', '?') for b in (d.get('backups') or [])) or 'n/a'}",
        "",
        "## 1. Triage",
        "",
        f"Model: {d.get('triage_model') or 'n/a'}",
        "",
        _clip(d.get("triage_output")),
        "",
        *_agent_block("2. Plans", d.get("planner_outputs") or [], "_no plans_"),
        *_agent_block("3. Critiques", d.get("critic_outputs") or [], "_no critiques_"),
        "## 4. Synthesis",
        "",
        f"Model: {d.get('synthesis_model') or 'n/a'}",
        "",
        _clip(d.get("final_plan") or d.get("synthesis_output")),
        "",
        "## 5. Implementation",
        "",
        f"Model: {d.get('implementer_model') or 'n/a'}",
        "",
        _clip(d.get("implementer_output")),
        "",
        *_agent_block("6. Reviews", d.get("review_outputs") or [], "_no reviews_"),
        "## 7. Review synthesis",
        "",
        f"Model: {d.get('review_synthesis_model') or 'n/a'}",
        "",
        _clip(d.get("review_synthesis_output")),
        "",
        *_agent_block("8. Fixes", d.get("fix_outputs") or [], "_no fix iterations_"),
        "## 9. Verification",
        "",
        f"- Tests: {'PASS' if tests.get('success') else 'FAIL'}",
        f"- Lint: {'PASS' if lint.get('success') else 'WARN/FAIL'}",
        f"- Types: {'PASS' if types.get('success') else 'WARN/FAIL/SKIP'}",
        "",
        _clip(tests.get("output"), 2000),
        "",
        "## Errors",
        "",
    ]
    if not errors:
        lines.append("_none_")
    else:
        for err in errors:
            lines.append(f"- [{err.get('phase')}] {_clip(err.get('error'), 400)}")
    lines.append("")
    return "\n".join(lines)


def write_final_report(state: TaskState) -> Path:
    """Write ``orchestra/state/reports/<task-id>.md`` and record the path."""
    dest = state.state_dir / "reports" / f"{state.task_id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_final_report(state), encoding="utf-8")
    state.data["final_report"] = str(dest)
    state.save()
    return dest
