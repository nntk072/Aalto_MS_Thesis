"""Prompt template loading for orchestra roles."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).parent

_ROLE_FILES = {
    "triage": "triage.md",
    "planner": "planner.md",
    "critic": "critic.md",
    "synthesizer": "synthesizer.md",
    "implementer": "implementer.md",
    "reviewer": "reviewer.md",
    "review_synthesizer": "review_synthesizer.md",
}


def load_prompt(role: str) -> str:
    """Load the prompt template for a role."""
    filename = _ROLE_FILES.get(role)
    if not filename:
        raise ValueError(f"Unknown role: {role}")
    path = PROMPT_DIR / filename
    return path.read_text()


def render_prompt(role: str, **kwargs) -> str:
    """Load and render a prompt template with variables."""
    template = load_prompt(role)
    for key, value in kwargs.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
