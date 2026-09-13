"""Model and provider definitions for the orchestra."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TIER_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
_COST_ORDER = {"free": 0, "low": 1, "medium": 2, "high": 3}
_CONFIG_PATH = Path(__file__).parent / "models.yaml"


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
    alias: str = ""  # legacy; Vibe uses model_id via VIBE_ACTIVE_MODEL
    registry_id: str = ""
    min_tier: str = "T0"
    cost_tier: str = "low"
    probe_forbidden: bool = False
    probe_policy: str = "tier0"
    rpd: int | None = None
    rpm: int | None = None
    effort_flag: str = ""
    effort_support: list[str] = field(default_factory=list)
    context_tokens: int | None = None
    native_resume: bool = False
    native_model_switch: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.registry_id:
            self.registry_id = f"{self.provider}-{self.name}".replace("/", "-")

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.name}"

    @property
    def model_id(self) -> str:
        """CLI model identifier (alias for name when loaded from YAML)."""
        return self.name

    def tier_rank(self) -> int:
        return _TIER_ORDER.get(self.min_tier.upper(), 0)

    def cost_rank(self) -> int:
        return _COST_ORDER.get(self.cost_tier, 1)

    def is_available(self) -> bool:
        """Check if the model's CLI and credentials are available."""
        from shutil import which

        if not which(self.cli):
            return False
        if self.cli == "vibe" and self.provider == "mistral":
            from .health import vibe_mistral_auth_ok

            return vibe_mistral_auth_ok()
        if self.env_var:
            return bool(os.environ.get(self.env_var))
        return True

    def vibe_model_env(self) -> str:
        """Model alias/id for VIBE_ACTIVE_MODEL."""
        if self.alias:
            return self.alias
        return self.name


def tier_rank(tier: str) -> int:
    return _TIER_ORDER.get(tier.upper(), 2)


def complexity_to_tier(complexity: str) -> str:
    mapping = {"trivial": "T0", "medium": "T2", "complex": "T3"}
    return mapping.get(complexity, "T2")


def _model_from_yaml(entry: dict[str, Any]) -> Model:
    model_id = entry.get("model_id", entry.get("name", "default"))
    return Model(
        name=model_id,
        provider=entry.get("provider", entry.get("cli", "")),
        cli=entry["cli"],
        roles=list(entry.get("roles", [])),
        priority=int(entry.get("priority", 5)),
        quota_daily=int(entry.get("quota_daily", 0)),
        escalation_only=bool(entry.get("escalation_only", False)),
        model_flag=entry.get("model_flag", ""),
        extra_args=list(entry.get("extra_args", [])),
        env_var=entry.get("env_var", ""),
        alias=entry.get("alias", ""),
        registry_id=entry.get("id", ""),
        min_tier=entry.get("min_tier", "T0"),
        cost_tier=entry.get("cost_tier", "low"),
        probe_forbidden=bool(entry.get("probe_forbidden", False)),
        probe_policy=entry.get("probe_policy", "tier0"),
        rpd=entry.get("rpd"),
        rpm=entry.get("rpm"),
        effort_flag=entry.get("effort_flag", ""),
        effort_support=list(entry.get("effort_support", [])),
        context_tokens=entry.get("context_tokens"),
        native_resume=bool(entry.get("native_resume", False)),
        native_model_switch=bool(entry.get("native_model_switch", False)),
        enabled=bool(entry.get("enabled", True)),
    )


# Default model roster — superseded by models.yaml when present
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
        registry_id="gemini-3.1-pro-preview",
        min_tier="T3",
        probe_forbidden=True,
        cost_tier="high",
        rpd=20,
    ),
    Model(
        name="mistral-medium-3.5",
        provider="mistral",
        cli="vibe",
        roles=["triage", "planner", "critic", "reviewer", "synthesizer", "review_synthesizer"],
        priority=2,
        quota_daily=1_000_000,
        env_var="MISTRAL_API_KEY",
        alias="mistral-medium-3.5",
        extra_args=["--auto-approve"],
        registry_id="vibe-mistral-medium-3.5",
    ),
    Model(
        name="devstral-local",
        provider="ollama",
        cli="vibe",
        roles=["implementer", "planner", "reviewer"],
        priority=3,
        quota_daily=0,
        extra_args=["--auto-approve"],
        registry_id="vibe-devstral-local",
        cost_tier="free",
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
        registry_id="opencode-default",
        effort_flag="oc_variant",
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
        registry_id="kilo-default",
        effort_flag="oc_variant",
    ),
]


_config_cache: dict[str, Any] | None = None


def load_orchestra_config(path: Path | None = None) -> dict[str, Any]:
    """Load models.yaml; returns effort_map, error_signatures, and models list."""
    global _config_cache
    path = path or _CONFIG_PATH
    if not path.exists():
        return {
            "version": 3,
            "effort_map": {},
            "error_signatures": {},
            "models": [],
        }
    data = yaml.safe_load(path.read_text()) or {}
    _config_cache = data
    return data


def load_models(path: Path | None = None) -> list[Model]:
    """Load roster from YAML, falling back to DEFAULT_MODELS."""
    cfg = load_orchestra_config(path)
    entries = cfg.get("models") or []
    if not entries:
        return list(DEFAULT_MODELS)
    return [_model_from_yaml(e) for e in entries if e.get("enabled", True)]


def get_effort_map() -> dict[str, dict[str, str]]:
    return load_orchestra_config().get("effort_map") or {}


def get_error_signatures() -> dict[str, Any]:
    return load_orchestra_config().get("error_signatures") or {}
