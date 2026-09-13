"""Prompt template loading for orchestra roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROMPT_DIR = Path(__file__).parent

_ROLE_FILES = {
    "triage": "triage.md",
    "planner": "planner.md",
    "critic": "critic.md",
    "synthesizer": "synthesizer.md",
    "implementer": "implementer.md",
    "reviewer": "reviewer.md",
    "review_synthesizer": "review_synthesizer.md",
    "handoff_continuation": "handoff_continuation.md",
}


def load_prompt(role: str) -> str:
    """Load the prompt template for a role."""
    filename = _ROLE_FILES.get(role)
    if not filename:
        raise ValueError(f"Unknown role: {role}")
    path = PROMPT_DIR / filename
    return path.read_text()


def render_ci_verification(workspace: str) -> str:
    """Shared CI gate block for agent prompts (matches orchestra/ci_gate.py)."""
    from ..ci_gate import ci_command_lines

    template = (PROMPT_DIR / "verification.md").read_text()
    return template.replace("{{workspace}}", workspace).replace(
        "{{ci_commands}}", ci_command_lines()
    )


def render_prompt(role: str, **kwargs: Any) -> str:
    """Load and render a prompt template with variables."""
    template = load_prompt(role)
    workspace = str(kwargs.get("workspace", ""))
    if "{{ci_verification}}" in template and workspace:
        template = template.replace("{{ci_verification}}", render_ci_verification(workspace))
    route_notes = str(kwargs.pop("route_notes", "") or "")
    if "{{route_notes}}" in template:
        block = route_notes.strip()
        template = template.replace(
            "{{route_notes}}",
            f"\n{block}\n" if block else "",
        )
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
