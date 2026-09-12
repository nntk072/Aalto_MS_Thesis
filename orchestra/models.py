"""Model and provider definitions for the orchestra."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Model:
    """A model that can be assigned to a role."""

    name: str
    provider: str
    cli: str
    roles: list[str]
    priority: int  # lower = higher priority
    quota_daily: int  # daily token budget; 0 = unlimited
    escalation_only: bool = False
    model_flag: str = ""  # CLI flag for model selection
    extra_args: list[str] = field(default_factory=list)
    env_var: str = ""  # environment variable for API key
    alias: str = ""  # alias used by the CLI tool

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.name}"

    def is_available(self) -> bool:
        """Check if the model's CLI and credentials are available."""
        if self.env_var:
            return bool(os.environ.get(self.env_var))
        # Check CLI binary exists
        from shutil import which

        return which(self.cli) is not None


# Default model roster — override in config.yaml
DEFAULT_MODELS: list[Model] = [
    Model(
        name="gemini-3.1-pro-preview",
        provider="gemini",
        cli="gemini",
        roles=["triage", "synthesizer", "reviewer", "critic", "review_synthesizer"],
        priority=1,
        quota_daily=500_000,
        escalation_only=True,
        model_flag="-m",
        env_var="GEMINI_API_KEY",
        extra_args=["--approval-mode", "yolo"],
    ),
    Model(
        name="mistral-medium-3.5",
        provider="mistral",
        cli="vibe",
        roles=["triage", "planner", "critic", "reviewer", "synthesizer", "review_synthesizer"],
        priority=2,
        quota_daily=1_000_000,
        model_flag="--agent",
        env_var="MISTRAL_API_KEY",
        alias="mistral-medium-3.5",
        extra_args=["--auto-approve"],
    ),
    Model(
        name="devstral-local",
        provider="ollama",
        cli="vibe",
        roles=["implementer", "planner", "reviewer"],
        priority=3,
        quota_daily=0,  # unlimited — local
        model_flag="--agent",
        alias="",
        extra_args=["--auto-approve"],
    ),
    Model(
        name="default",
        provider="opencode",
        cli="opencode",
        roles=[
            "triage",
            "planner",
            "reviewer",
            "implementer",
            "critic",
            "synthesizer",
            "review_synthesizer",
        ],
        priority=2,
        quota_daily=2_000_000,
        model_flag="--model",
        extra_args=["--auto"],
    ),
    Model(
        name="default",
        provider="kilo",
        cli="kilo",
        roles=["planner", "reviewer", "implementer", "critic", "synthesizer"],
        priority=3,
        quota_daily=2_000_000,
        model_flag="--model",
        extra_args=["--auto"],
    ),
]
