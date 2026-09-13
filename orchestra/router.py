"""Model router — quota-aware, fallback-aware model selection."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .health import HealthRegistry
from .models import Model, load_models, tier_rank

_PACIFIC = ZoneInfo("America/Los_Angeles")
_RPD_SAFETY = 1


class QuotaTracker:
    """Tracks daily token and request usage per model."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.usage_file = state_dir / "usage.json"
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.usage_file.exists():
            self._data = json.loads(self.usage_file.read_text())

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.usage_file.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump(self._data, fh, indent=2)
        os.replace(tmp, self.usage_file)

    def _today_local(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _today_pacific(self) -> str:
        from datetime import datetime

        return datetime.now(_PACIFIC).strftime("%Y-%m-%d")

    def _day_key(self, model: Model) -> str:
        if model.rpd is not None:
            return self._today_pacific()
        return self._today_local()

    def get_usage(self, model_name: str, model: Model | None = None) -> dict[str, int]:
        day = self._day_key(model) if model else self._today_local()
        entry = self._data.get(day, {}).get(model_name, {})
        calls = int(entry.get("calls", 0))
        return {
            "tokens": int(entry.get("tokens", 0)),
            "calls": calls,
            "requests": int(entry.get("requests", calls)),
        }

    def record_usage(self, model_name: str, tokens: int, model: Model | None = None) -> None:
        day = self._day_key(model) if model else self._today_local()
        lock_path = self.state_dir / "usage.lock"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            self._load()
            if day not in self._data:
                self._data[day] = {}
            if model_name not in self._data[day]:
                self._data[day][model_name] = {"tokens": 0, "calls": 0, "requests": 0}
            self._data[day][model_name]["tokens"] += tokens
            self._data[day][model_name]["calls"] += 1
            self._data[day][model_name]["requests"] = (
                self._data[day][model_name].get("requests", 0) + 1
            )
            self._save()
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    def quota_exceeded(self, model: Model) -> bool:
        """Check if a model has exceeded its daily quota."""
        usage = self.get_usage(model.display_name, model)
        if model.rpd is not None:
            return usage["requests"] >= max(0, model.rpd - _RPD_SAFETY)
        if model.quota_daily == 0:
            return False
        return usage["tokens"] >= model.quota_daily

    def remaining(self, model: Model) -> int:
        """Remaining budget for today. Returns -1 for unlimited tokens."""
        usage = self.get_usage(model.display_name, model)
        if model.rpd is not None:
            return max(0, model.rpd - _RPD_SAFETY - usage["requests"])
        if model.quota_daily == 0:
            return -1
        return max(0, model.quota_daily - usage["tokens"])


class ModelRouter:
    """Selects models for roles with fallback and quota awareness."""

    def __init__(
        self,
        models: list[Model] | None = None,
        state_dir: Path | None = None,
    ):
        self.models = models or load_models()
        state_dir = state_dir or Path(__file__).parent / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        self.quota = QuotaTracker(state_dir)
        self.health = HealthRegistry(state_dir)
        self.health.probe_tier0(self.models, check_version=False)

    def get_candidates(self, role: str) -> list[Model]:
        """Get all models that can fulfill a role, sorted by priority."""
        candidates = [m for m in self.models if role in m.roles and m.enabled]
        candidates.sort(key=lambda m: (m.priority, m.cost_rank()))
        return candidates

    def _filter_candidates(
        self,
        candidates: list[Model],
        task_tier: str,
        escalate: bool = False,
        exclude: tuple[str, str] | None = None,
    ) -> list[Model]:
        tier_n = tier_rank(task_tier)
        filtered: list[Model] = []
        for m in candidates:
            if exclude and (m.name, m.provider) == exclude:
                continue
            if not m.is_available():
                continue
            if not self.health.is_selectable(m):
                continue
            if self.quota.quota_exceeded(m):
                continue
            if m.escalation_only and not escalate and tier_n <= tier_rank("T2"):
                continue
            if m.tier_rank() > tier_n:
                continue
            filtered.append(m)
        filtered.sort(key=lambda m: (m.priority, m.cost_rank(), -self.quota.remaining(m)))
        return filtered

    def select(
        self,
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        """Select models for a role. Returns empty list when no candidate (PARK)."""
        from .models import complexity_to_tier

        tier = task_tier or complexity_to_tier(task_complexity)
        candidates = self.get_candidates(role)
        within = self._filter_candidates(candidates, tier, escalate=escalate)
        return within[:count]

    def fallback_chain(
        self,
        primary: Model,
        role: str,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        """Other available models for role if primary fails."""
        from .models import complexity_to_tier

        tier = task_tier or complexity_to_tier(task_complexity)
        candidates = self.get_candidates(role)
        return self._filter_candidates(
            candidates,
            tier,
            escalate=escalate,
            exclude=(primary.name, primary.provider),
        )

    def select_with_fallback(
        self,
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[tuple[Model, list[Model]]]:
        """Select models with their fallback chains."""
        selected = self.select(role, count, task_complexity, task_tier, escalate)
        return [
            (
                model,
                self.fallback_chain(model, role, task_complexity, task_tier, escalate),
            )
            for model in selected
        ]

    def stats(self) -> dict[str, Any]:
        """Return quota usage stats for all models."""
        stats: dict[str, Any] = {}
        for model in self.models:
            usage = self.quota.get_usage(model.display_name, model)
            remaining = self.quota.remaining(model)
            stats[model.display_name] = {
                "tokens_used": usage["tokens"],
                "calls": usage["calls"],
                "requests": usage["requests"],
                "quota": model.quota_daily,
                "rpd": model.rpd,
                "remaining": remaining,
                "escalation_only": model.escalation_only,
                "available": model.is_available(),
                "health": self.health.status(model).value,
                "registry_id": model.registry_id,
            }
        return stats
