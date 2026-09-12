"""Cost and token tracking for orchestra models."""

from __future__ import annotations

import json
import time
from pathlib import Path

# Approximate cost per 1M tokens (USD) — update as pricing changes
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gemini/gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "mistral/mistral-medium-3.5": {"input": 0.40, "output": 2.00},
    "ollama/devstral-local": {"input": 0.0, "output": 0.0},
    "ollama/devstral-small": {"input": 0.0, "output": 0.0},
    "opencode/default": {"input": 0.10, "output": 0.30},
    "kilo/default": {"input": 0.10, "output": 0.30},
}


class CostTracker:
    """Tracks token usage and cost per model and per task."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.cost_file = state_dir / "costs.json"
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        if self.cost_file.exists():
            self._data = json.loads(self.cost_file.read_text())

    def _save(self) -> None:
        self.cost_file.write_text(json.dumps(self._data, indent=2, default=str))

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        task_id: str | None = None,
    ) -> dict:
        """Record token usage for a model call. Returns cost breakdown."""
        today = self._today()

        model_cost = MODEL_COSTS.get(model, {"input": 0.10, "output": 0.30})
        input_cost = (prompt_tokens / 1_000_000) * model_cost["input"]
        output_cost = (completion_tokens / 1_000_000) * model_cost["output"]
        total_cost = round(input_cost + output_cost, 6)

        record = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": total_cost,
            "task_id": task_id,
            "timestamp": time.time(),
        }

        if today not in self._data:
            self._data[today] = {"calls": [], "totals": {}}
        self._data[today]["calls"].append(record)

        totals = self._data[today]["totals"]
        totals["prompt_tokens"] = totals.get("prompt_tokens", 0) + prompt_tokens
        totals["completion_tokens"] = totals.get("completion_tokens", 0) + completion_tokens
        totals["total_cost"] = round(totals.get("total_cost", 0) + total_cost, 6)

        model_key = model.replace("/", "_")
        if model_key not in totals:
            totals[model_key] = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        totals[model_key]["prompt_tokens"] += prompt_tokens
        totals[model_key]["completion_tokens"] += completion_tokens
        totals[model_key]["cost"] = round(totals[model_key]["cost"] + total_cost, 6)

        self._save()
        return record

    def get_daily_summary(self, date: str | None = None) -> dict:
        """Get cost summary for a specific date (defaults to today)."""
        date = date or self._today()
        day_data = self._data.get(date, {"calls": [], "totals": {}})
        totals = day_data.get("totals", {})
        return {
            "date": date,
            "total_calls": len(day_data.get("calls", [])),
            "prompt_tokens": totals.get("prompt_tokens", 0),
            "completion_tokens": totals.get("completion_tokens", 0),
            "total_cost": totals.get("total_cost", 0.0),
            "models": {
                k: v
                for k, v in totals.items()
                if k not in ("prompt_tokens", "completion_tokens", "total_cost")
            },
        }

    def get_monthly_summary(self, month: str | None = None) -> dict:
        """Get aggregated cost summary for a month."""
        month = month or time.strftime("%Y-%m")
        monthly = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0, "calls": 0}

        for date, day_data in self._data.items():
            if date.startswith(month):
                totals = day_data.get("totals", {})
                monthly["prompt_tokens"] += totals.get("prompt_tokens", 0)
                monthly["completion_tokens"] += totals.get("completion_tokens", 0)
                monthly["total_cost"] = round(
                    monthly["total_cost"] + totals.get("total_cost", 0.0), 6
                )
                monthly["calls"] += len(day_data.get("calls", []))

        monthly["month"] = month
        return monthly

    def get_task_cost(self, task_id: str) -> dict | None:
        """Get total cost for a specific task across all days."""
        task_calls = []
        total_cost = 0.0
        total_prompt = 0
        total_completion = 0

        for date, day_data in self._data.items():
            for call in day_data.get("calls", []):
                if call.get("task_id") == task_id:
                    task_calls.append(call)
                    total_cost += call.get("total_cost", 0)
                    total_prompt += call.get("prompt_tokens", 0)
                    total_completion += call.get("completion_tokens", 0)

        if not task_calls:
            return None

        return {
            "task_id": task_id,
            "calls": len(task_calls),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_cost": round(total_cost, 6),
        }
