"""Durable task state — track and resume orchestra pipeline execution."""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class Phase(str, Enum):
    """Pipeline phases in order."""
    PENDING = "pending"
    TRIAGE = "triage"
    PLANNING = "planning"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    REVIEW_SYNTHESIS = "review_synthesis"
    FIXING = "fixing"
    VERIFICATION = "verification"
    REPORTING = "reporting"
    DONE = "done"
    FAILED = "failed"

    def next(self) -> Phase:
        """Get the next phase in the pipeline."""
        members = list(Phase)
        idx = members.index(self)
        if idx < len(members) - 1:
            return members[idx + 1]
        return self

    @classmethod
    def runnable(cls) -> tuple[Phase, ...]:
        return tuple(p for p in cls if p not in (cls.PENDING, cls.DONE, cls.FAILED))

    @classmethod
    def parse(cls, raw: str) -> Phase:
        """Parse a phase name, alias, or step number (``2`` / ``planning``)."""
        key = raw.strip().lower()
        key = key.replace("[", "").replace("]", "").replace("/10", "").replace("/9", "")
        key = key.replace(" ", "_").replace("-", "_")
        if key.startswith("step_"):
            key = key[5:]
        aliases = {
            "1": cls.TRIAGE,
            "triage": cls.TRIAGE,
            "2": cls.PLANNING,
            "plan": cls.PLANNING,
            "planning": cls.PLANNING,
            "planners": cls.PLANNING,
            "3": cls.CRITIQUE,
            "critique": cls.CRITIQUE,
            "critic": cls.CRITIQUE,
            "4": cls.SYNTHESIS,
            "synthesis": cls.SYNTHESIS,
            "5": cls.IMPLEMENTATION,
            "implement": cls.IMPLEMENTATION,
            "implementation": cls.IMPLEMENTATION,
            "6": cls.REVIEW,
            "review": cls.REVIEW,
            "7": cls.REVIEW_SYNTHESIS,
            "6b": cls.REVIEW_SYNTHESIS,
            "review_synthesis": cls.REVIEW_SYNTHESIS,
            "8": cls.FIXING,
            "6c": cls.FIXING,
            "fix": cls.FIXING,
            "fixing": cls.FIXING,
            "9": cls.VERIFICATION,
            "verify": cls.VERIFICATION,
            "verification": cls.VERIFICATION,
            "10": cls.REPORTING,
            "report": cls.REPORTING,
            "reporting": cls.REPORTING,
            "final_report": cls.REPORTING,
        }
        if key in aliases:
            return aliases[key]
        try:
            phase = cls(key)
        except ValueError as exc:
            names = ", ".join(p.value for p in cls.runnable())
            raise ValueError(
                f"Unknown phase {raw!r}. Use a name or step 1-10 ({names})"
            ) from exc
        if phase not in cls.runnable():
            raise ValueError(f"Cannot start from {phase.value}")
        return phase


class TaskState:
    """Durable state for a single orchestra task."""

    def __init__(
        self,
        task_id: str,
        task_description: str,
        state_dir: Path,
        data: Optional[dict] = None,
    ):
        self.task_id = task_id
        self.task_description = task_description
        self.state_dir = state_dir
        self.state_file = state_dir / f"{task_id}.json"
        self.data: dict[str, Any] = data or self._default_data()
        if not self.data.get("task_description"):
            self.data["task_description"] = task_description

    def _default_data(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "phase": Phase.PENDING.value,
            "created_at": time.time(),
            "updated_at": time.time(),
            "complexity": None,
            "triage_output": None,
            "triage_model": None,
            "planner_outputs": [],
            "planner_session_map": {},
            "critic_outputs": [],
            "synthesis_output": None,
            "synthesis_model": None,
            "final_plan": None,
            "implementer_output": None,
            "implementer_model": None,
            "review_outputs": [],
            "review_synthesis_output": None,
            "review_synthesis_model": None,
            "review_verdict": None,
            "fix_loop_count": 0,
            "fix_loop_max": 3,
            "fix_outputs": [],
            "performance": {},
            "verification_result": None,
            "errors": [],
            "metadata": {},
        }

    def save(self) -> None:
        """Persist state to disk."""
        self.data["updated_at"] = time.time()
        self.state_file.write_text(json.dumps(self.data, indent=2, default=str))

    @classmethod
    def load(cls, task_id: str, state_dir: Path) -> "TaskState":
        """Load state from disk."""
        state_file = state_dir / f"{task_id}.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No state for task {task_id}")
        data = json.loads(state_file.read_text())
        return cls(
            task_id=data["task_id"],
            task_description=data["task_description"],
            state_dir=state_dir,
            data=data,
        )

    @property
    def phase(self) -> Phase:
        return Phase(self.data["phase"])

    def set_phase(self, phase: Phase) -> None:
        """Transition to a new phase."""
        current = Phase(self.data["phase"])
        if phase is Phase.FAILED and current in Phase.runnable():
            self.data["failed_at"] = current.value
        self.data["phase"] = phase.value
        self.save()

    def add_planner_output(self, model: str, output: str, session: str) -> None:
        """Record a planner's output."""
        self.data["planner_outputs"].append({
            "model": model,
            "output": output,
            "session": session,
            "timestamp": time.time(),
        })
        self.data["planner_session_map"][session] = model
        self.save()

    def add_critic_output(self, model: str, output: str, session: str) -> None:
        """Record a critic's output."""
        self.data["critic_outputs"].append({
            "model": model,
            "output": output,
            "session": session,
            "timestamp": time.time(),
        })
        self.save()

    def add_reviewer_output(self, model: str, output: str, session: str) -> None:
        """Record a reviewer's output."""
        self.data["review_outputs"].append({
            "model": model,
            "output": output,
            "session": session,
            "timestamp": time.time(),
        })
        self.save()

    def set_triage(self, output: str, model: str, complexity: str) -> None:
        """Store triage results."""
        self.data["triage_output"] = output
        self.data["triage_model"] = model
        self.data["complexity"] = complexity
        self.save()

    def set_synthesis(self, output: str, model: str) -> None:
        """Store synthesis results."""
        self.data["synthesis_output"] = output
        self.data["synthesis_model"] = model
        self.data["final_plan"] = output
        self.save()

    def set_implementation(self, output: str, model: str) -> None:
        """Store implementation results."""
        self.data["implementer_output"] = output
        self.data["implementer_model"] = model
        self.save()

    def set_review_synthesis(self, output: str, model: str, verdict: str) -> None:
        """Store review synthesis results."""
        self.data["review_synthesis_output"] = output
        self.data["review_synthesis_model"] = model
        self.data["review_verdict"] = verdict
        self.save()

    def add_fix_output(self, output: str, model: str) -> None:
        """Record a fix iteration output."""
        self.data["fix_outputs"].append({
            "model": model,
            "output": output,
            "loop": self.data["fix_loop_count"],
            "timestamp": time.time(),
        })
        self.data["fix_loop_count"] += 1
        self.save()

    def record_performance(self, model: str, role: str, success: bool, tokens_used: int = 0) -> None:
        """Record model performance for a role."""
        perf = self.data.setdefault("performance", {})
        if model not in perf:
            perf[model] = {"roles": {}, "total_calls": 0, "successes": 0}
        perf[model]["total_calls"] += 1
        if success:
            perf[model]["successes"] += 1
        if role not in perf[model]["roles"]:
            perf[model]["roles"][role] = {"calls": 0, "successes": 0}
        perf[model]["roles"][role]["calls"] += 1
        if success:
            perf[model]["roles"][role]["successes"] += 1
        perf[model]["tokens_used"] = perf[model].get("tokens_used", 0) + tokens_used
        self.save()

    def set_verification(self, result: dict) -> None:
        """Store verification results."""
        self.data["verification_result"] = result
        self.save()

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.data["errors"].append({
            "error": error,
            "phase": self.data["phase"],
            "timestamp": time.time(),
        })
        self.save()

    def get_planner_outputs_text(self) -> str:
        """Get all planner outputs as a single formatted text block."""
        if not self.data["planner_outputs"]:
            return "No plans generated."
        parts = []
        for i, plan in enumerate(self.data["planner_outputs"], 1):
            parts.append(f"### Plan {i} (from {plan['model']})\n{plan['output']}")
        return "\n\n".join(parts)

    def get_review_outputs_text(self) -> str:
        """Get all review outputs as a single formatted text block."""
        if not self.data["review_outputs"]:
            return "No reviews generated."
        parts = []
        for i, review in enumerate(self.data["review_outputs"], 1):
            parts.append(f"### Review {i} (from {review['model']})\n{review['output']}")
        return "\n\n".join(parts)

    def get_critic_outputs_text(self) -> str:
        """Get all critic outputs as a single formatted text block."""
        if not self.data["critic_outputs"]:
            return "No critiques generated."
        parts = []
        for i, critique in enumerate(self.data["critic_outputs"], 1):
            parts.append(f"### Critique {i} (from {critique['model']})\n{critique['output']}")
        return "\n\n".join(parts)


def list_tasks(state_dir: Path) -> list[dict]:
    """List all tasks in the state directory."""
    if not state_dir.exists():
        return []
    tasks = []
    for f in sorted(state_dir.glob("task-*.json")):
        try:
            data = json.loads(f.read_text())
            tasks.append({
                "task_id": data.get("task_id"),
                "phase": data.get("phase"),
                "description": data.get("task_description", "")[:60],
                "updated_at": data.get("updated_at"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return tasks


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"task-{int(time.time())}"

