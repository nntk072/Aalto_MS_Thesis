"""Model router — quota-aware, fallback-aware model selection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .models import Model, DEFAULT_MODELS


class QuotaTracker:
    """Tracks daily token usage per model."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.usage_file = state_dir / "usage.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.usage_file.exists():
            self._data = json.loads(self.usage_file.read_text())

    def _save(self) -> None:
        self.usage_file.write_text(json.dumps(self._data, indent=2))

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def get_usage(self, model_name: str) -> dict:
        today = self._today()
        return self._data.get(today, {}).get(model_name, {"tokens": 0, "calls": 0})

    def record_usage(self, model_name: str, tokens: int) -> None:
        today = self._today()
        if today not in self._data:
            self._data[today] = {}
        if model_name not in self._data[today]:
            self._data[today][model_name] = {"tokens": 0, "calls": 0}
        self._data[today][model_name]["tokens"] += tokens
        self._data[today][model_name]["calls"] += 1
        self._save()

    def quota_exceeded(self, model: Model) -> bool:
        """Check if a model has exceeded its daily quota."""
        if model.quota_daily == 0:  # unlimited
            return False
        usage = self.get_usage(model.display_name)
        return usage["tokens"] >= model.quota_daily

    def remaining(self, model: Model) -> int:
        """Remaining token budget for today. Returns -1 for unlimited."""
        if model.quota_daily == 0:
            return -1
        usage = self.get_usage(model.display_name)
        return max(0, model.quota_daily - usage["tokens"])


class ModelRouter:
    """Selects models for roles with fallback and quota awareness."""

    def __init__(
        self,
        models: Optional[list[Model]] = None,
        state_dir: Optional[Path] = None,
    ):
        self.models = models or DEFAULT_MODELS
        state_dir = state_dir or Path(__file__).parent / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.quota = QuotaTracker(state_dir)

    def get_candidates(self, role: str) -> list[Model]:
        """Get all models that can fulfill a role, sorted by priority."""
        candidates = [m for m in self.models if role in m.roles]
        candidates.sort(key=lambda m: m.priority)
        return candidates

    def select(
        self,
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        escalate: bool = False,
    ) -> list[Model]:
        """Select models for a role.

        Filters out:
        - Models not available (missing CLI or credentials)
        - Models that exceeded daily quota
        - Escalation-only models for trivial tasks (unless escalate=True)

        Returns up to `count` models, ordered by priority.
        """
        candidates = self.get_candidates(role)

        # Filter: availability
        available = [m for m in candidates if m.is_available()]

        # Filter: quota
        within_quota = [m for m in available if not self.quota.quota_exceeded(m)]

        # Filter: escalation policy
        if task_complexity in ("trivial", "medium") and not escalate:
            within_quota = [m for m in within_quota if not m.escalation_only]

        # If nothing left and we filtered escalation models, re-add them
        if not within_quota and task_complexity == "medium" and not escalate:
            within_quota = [
                m for m in available if not self.quota.quota_exceeded(m)
            ]

        # Last resort: return whatever is available even over quota
        if not within_quota:
            within_quota = available

        selected = within_quota[:count]
        return selected

    def select_with_fallback(
        self,
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
    ) -> list[tuple[Model, list[Model]]]:
        """Select models with their fallback chains.

        Returns list of (primary_model, [fallback_models]) tuples.
        """
        selected = self.select(role, count, task_complexity)
        all_candidates = self.get_candidates(role)
        result = []
        for model in selected:
            # Fallback chain: other models of same role, different provider, by priority
            fallbacks = [
                m for m in all_candidates
                if m.name != model.name
                and m.provider != model.provider
                and m.is_available()
                and not self.quota.quota_exceeded(m)
            ]
            result.append((model, fallbacks))
        return result

    def stats(self) -> dict:
        """Return quota usage stats for all models."""
        stats = {}
        for model in self.models:
            usage = self.quota.get_usage(model.display_name)
            remaining = self.quota.remaining(model)
            stats[model.display_name] = {
                "tokens_used": usage["tokens"],
                "calls": usage["calls"],
                "quota": model.quota_daily,
                "remaining": remaining,
                "escalation_only": model.escalation_only,
                "available": model.is_available(),
            }
        return stats
