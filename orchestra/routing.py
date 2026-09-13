"""Phase routing — agent-decided hops within semantic allowlists."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .parse import extract_json_objects
from .state import Phase

logger = logging.getLogger(__name__)

MAX_REVISITS = 3
VALID_VERIFY_MODES = frozenset({"full", "scoped", "skip"})

_GOTO_ALIASES: dict[str, Phase] = {
    "1": Phase.TRIAGE,
    "2": Phase.PLANNING,
    "3": Phase.CRITIQUE,
    "4": Phase.SYNTHESIS,
    "5": Phase.IMPLEMENTATION,
    "6": Phase.REVIEW,
    "7": Phase.REVIEW_SYNTHESIS,
    "8": Phase.FIXING,
    "9": Phase.VERIFICATION,
    "10": Phase.REPORTING,
}

ALLOWED_NEXT: dict[Phase, frozenset[Phase]] = {
    Phase.TRIAGE: frozenset({Phase.PLANNING, Phase.IMPLEMENTATION, Phase.REPORTING}),
    Phase.PLANNING: frozenset(
        {Phase.TRIAGE, Phase.CRITIQUE, Phase.SYNTHESIS, Phase.IMPLEMENTATION, Phase.REPORTING}
    ),
    Phase.CRITIQUE: frozenset(
        {Phase.TRIAGE, Phase.PLANNING, Phase.SYNTHESIS, Phase.IMPLEMENTATION, Phase.REPORTING}
    ),
    Phase.SYNTHESIS: frozenset(
        {Phase.TRIAGE, Phase.PLANNING, Phase.CRITIQUE, Phase.IMPLEMENTATION, Phase.REPORTING}
    ),
    Phase.IMPLEMENTATION: frozenset(
        {Phase.TRIAGE, Phase.PLANNING, Phase.REVIEW, Phase.VERIFICATION, Phase.REPORTING}
    ),
    Phase.REVIEW: frozenset(
        {
            Phase.TRIAGE,
            Phase.PLANNING,
            Phase.IMPLEMENTATION,
            Phase.REVIEW_SYNTHESIS,
            Phase.FIXING,
            Phase.VERIFICATION,
            Phase.REPORTING,
        }
    ),
    Phase.REVIEW_SYNTHESIS: frozenset(
        {
            Phase.TRIAGE,
            Phase.PLANNING,
            Phase.IMPLEMENTATION,
            Phase.REVIEW,
            Phase.FIXING,
            Phase.VERIFICATION,
            Phase.REPORTING,
        }
    ),
    Phase.FIXING: frozenset(
        {
            Phase.TRIAGE,
            Phase.PLANNING,
            Phase.IMPLEMENTATION,
            Phase.REVIEW,
            Phase.VERIFICATION,
            Phase.REPORTING,
        }
    ),
    Phase.VERIFICATION: frozenset(
        {
            Phase.TRIAGE,
            Phase.PLANNING,
            Phase.IMPLEMENTATION,
            Phase.REVIEW,
            Phase.FIXING,
            Phase.REPORTING,
        }
    ),
    Phase.REPORTING: frozenset(),
}


@dataclass
class RouteUpdate:
    """Parsed routing directive from an agent."""

    goto: Phase | None = None
    skip: frozenset[Phase] = field(default_factory=frozenset)
    planner_count: int | None = None
    critic_count: int | None = None
    reviewer_count: int | None = None
    prompt_notes: dict[str, str] = field(default_factory=dict)
    verify_mode: str | None = None


def parse_phase_ref(value: Any) -> Phase | None:
    """Parse a phase reference from int, step number, or phase name."""
    if value is None:
        return None
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _GOTO_ALIASES:
        return _GOTO_ALIASES[key]
    try:
        phase = Phase(key)
    except ValueError:
        return None
    if phase in Phase.runnable():
        return phase
    return None


def _parse_skip_list(raw: Any) -> frozenset[Phase]:
    if not isinstance(raw, list):
        return frozenset()
    phases: set[Phase] = set()
    for item in raw:
        phase = parse_phase_ref(item)
        if phase is not None:
            phases.add(phase)
    return frozenset(phases)


def _apply_route_dict(update: RouteUpdate, route: dict[str, Any]) -> None:
    if "goto" in route:
        update.goto = parse_phase_ref(route["goto"])
    if "skip" in route:
        update.skip = _parse_skip_list(route["skip"])
    for key in ("planner_count", "critic_count", "reviewer_count"):
        if key in route and route[key] is not None:
            setattr(update, key, int(route[key]))
    if isinstance(route.get("prompt_notes"), dict):
        for role, note in route["prompt_notes"].items():
            if isinstance(note, str) and note.strip():
                update.prompt_notes[str(role)] = note.strip()
    mode = route.get("verify_mode")
    if isinstance(mode, str) and mode in VALID_VERIFY_MODES:
        update.verify_mode = mode


def parse_route_from_text(text: str) -> RouteUpdate | None:
    """Extract the last ``route`` block from agent output."""
    update: RouteUpdate | None = None
    for obj in extract_json_objects(text):
        route = obj.get("route")
        if isinstance(route, dict):
            update = RouteUpdate()
            _apply_route_dict(update, route)
        elif "goto" in obj and any(k in obj for k in ("complexity", "tier", "needs_planning")):
            update = RouteUpdate()
            update.goto = parse_phase_ref(obj["goto"])
            _apply_route_dict(update, obj)
    return update


def merge_route(data: dict[str, Any], update: RouteUpdate | None) -> None:
    """Merge a parsed route update into task state."""
    if update is None:
        return
    route = data.setdefault("route", {})
    if update.goto is not None:
        route["last_goto"] = update.goto.value
    if update.skip:
        existing = _parse_skip_list(route.get("skip"))
        route["skip"] = sorted({p.value for p in existing | update.skip})
    for key in ("planner_count", "critic_count", "reviewer_count"):
        value = getattr(update, key)
        if value is not None:
            route[key] = max(0, int(value))
    if update.prompt_notes:
        notes = route.setdefault("prompt_notes", {})
        notes.update(update.prompt_notes)
    if update.verify_mode is not None:
        route["verify_mode"] = update.verify_mode
    data["route"] = route


def effective_planner_count(data: dict[str, Any], default: int) -> int:
    route = data.get("route") or {}
    if "planner_count" in route:
        return max(0, int(route["planner_count"]))
    return default


def effective_critic_count(data: dict[str, Any], default: int) -> int:
    route = data.get("route") or {}
    if "critic_count" in route:
        return max(0, int(route["critic_count"]))
    return default


def effective_reviewer_count(data: dict[str, Any], default: int) -> int:
    route = data.get("route") or {}
    if "reviewer_count" in route:
        return max(0, int(route["reviewer_count"]))
    return default


def _explicit_skip(data: dict[str, Any]) -> frozenset[Phase]:
    route = data.get("route") or {}
    return _parse_skip_list(route.get("skip"))


def _multi_plan(data: dict[str, Any], planner_count: int) -> bool:
    outputs = data.get("planner_outputs") or []
    return planner_count > 1 or len(outputs) > 1


def _multi_review(data: dict[str, Any], reviewer_count: int) -> bool:
    outputs = data.get("review_outputs") or []
    return reviewer_count > 1 or len(outputs) > 1


def should_auto_skip(
    phase: Phase,
    data: dict[str, Any],
    *,
    planner_count: int,
    reviewer_count: int,
) -> bool:
    """Return True when a merge-only phase has nothing to merge."""
    if phase in _explicit_skip(data):
        return True
    if phase is Phase.SYNTHESIS and planner_count <= 1:
        return True
    if phase is Phase.CRITIQUE and effective_critic_count(data, 1) <= 0:
        return True
    if phase is Phase.REVIEW_SYNTHESIS and reviewer_count <= 1:
        return True
    return False


def substitute_auto_skips(
    target: Phase,
    data: dict[str, Any],
    *,
    planner_count: int,
    reviewer_count: int,
) -> Phase:
    """Replace merge-only phases with their sensible successor."""
    if should_auto_skip(target, data, planner_count=planner_count, reviewer_count=reviewer_count):
        if target is Phase.SYNTHESIS:
            return Phase.IMPLEMENTATION
        if target is Phase.CRITIQUE:
            return Phase.SYNTHESIS if _multi_plan(data, planner_count) else Phase.IMPLEMENTATION
        if target is Phase.REVIEW_SYNTHESIS:
            verdict = data.get("review_verdict")
            if verdict == "fail":
                return Phase.FIXING
            return Phase.VERIFICATION
    return target


def is_goto_allowed(current: Phase, target: Phase) -> bool:
    return target in ALLOWED_NEXT.get(current, frozenset())


def is_coding_task(data: dict[str, Any]) -> bool:
    """Heuristic: tasks that require implementation before reporting."""
    for obj in extract_json_objects(str(data.get("triage_output") or "")):
        domain = obj.get("domain")
        if isinstance(domain, str) and domain in ("docs", "config", "research"):
            return False
    complexity = data.get("complexity")
    if complexity == "trivial":
        for obj in extract_json_objects(str(data.get("triage_output") or "")):
            if obj.get("needs_planning") is False:
                return False
    return True


def apply_soft_guards(
    current: Phase,
    target: Phase,
    data: dict[str, Any],
    *,
    tier: str,
) -> Phase:
    """Force implementation before early exit on coding tasks."""
    if target is not Phase.REPORTING:
        return target
    if data.get("implementation_succeeded"):
        return target
    if not is_coding_task(data):
        return target
    route = data.get("route") or {}
    verify_mode = route.get("verify_mode", "full")
    if tier == "T0" and verify_mode == "skip":
        return target
    if current is Phase.IMPLEMENTATION:
        return target
    logger.warning("soft guard: coding task → reporting without implementation; forcing phase 5")
    return Phase.IMPLEMENTATION


def apply_triage_bootstrap(data: dict[str, Any], *, tier: str, had_explicit_goto: bool) -> None:
    """T0 silent default: jump to implementation (then 9→10)."""
    if had_explicit_goto:
        return
    if tier != "T0":
        return
    merge_route(data, RouteUpdate(goto=Phase.IMPLEMENTATION))


def silent_default_next(
    current: Phase,
    data: dict[str, Any],
    *,
    planner_count: int,
    critic_count: int,
    reviewer_count: int,
    verification_passed: bool | None = None,
    tier: str = "T2",
) -> Phase:
    """Phase default when the agent omits ``route.goto``."""
    if current is Phase.TRIAGE:
        if tier == "T0":
            return Phase.IMPLEMENTATION
        return Phase.PLANNING
    if current is Phase.PLANNING:
        if critic_count >= 1:
            return Phase.CRITIQUE
        if _multi_plan(data, planner_count):
            return Phase.SYNTHESIS
        return Phase.IMPLEMENTATION
    if current is Phase.CRITIQUE:
        if _multi_plan(data, planner_count):
            return Phase.SYNTHESIS
        return Phase.IMPLEMENTATION
    if current is Phase.SYNTHESIS:
        return Phase.IMPLEMENTATION
    if current is Phase.IMPLEMENTATION:
        if reviewer_count >= 1:
            return Phase.REVIEW
        return Phase.VERIFICATION
    if current is Phase.REVIEW:
        if _multi_review(data, reviewer_count):
            return Phase.REVIEW_SYNTHESIS
        return Phase.VERIFICATION
    if current is Phase.REVIEW_SYNTHESIS:
        if data.get("review_verdict") == "fail":
            return Phase.FIXING
        return Phase.VERIFICATION
    if current is Phase.FIXING:
        return Phase.VERIFICATION
    if current is Phase.VERIFICATION:
        if verification_passed is True:
            return Phase.REPORTING
        if verification_passed is False:
            return Phase.FIXING
        return Phase.REPORTING
    if current is Phase.REPORTING:
        return Phase.DONE
    return current.next()


def check_revisit_cap(data: dict[str, Any], target: Phase) -> Phase:
    """Limit rewinds among phases 2–9."""
    if target in (Phase.TRIAGE, Phase.REPORTING, Phase.DONE, Phase.PENDING):
        return target
    visits = data.setdefault("phase_visit_counts", {})
    if visits.get(target.value, 0) >= MAX_REVISITS:
        logger.warning("revisit cap hit for %s; routing to reporting", target.value)
        return Phase.REPORTING
    return target


def record_phase_visit(data: dict[str, Any], phase: Phase) -> None:
    if phase not in Phase.runnable():
        return
    visits = data.setdefault("phase_visit_counts", {})
    visits[phase.value] = visits.get(phase.value, 0) + 1
    data["hop_count"] = int(data.get("hop_count") or 0) + 1


def resolve_next_phase(
    current: Phase,
    data: dict[str, Any],
    update: RouteUpdate | None,
    *,
    planner_count: int,
    critic_count: int,
    reviewer_count: int,
    verification_passed: bool | None = None,
    tier: str = "T2",
) -> tuple[Phase, str]:
    """Choose the next phase after ``current`` completes."""
    eff_planners = effective_planner_count(data, planner_count)
    eff_critics = effective_critic_count(data, critic_count)
    eff_reviewers = effective_reviewer_count(data, reviewer_count)

    route = data.get("route") or {}
    proposed = update.goto if update and update.goto is not None else None
    if proposed is None:
        proposed = parse_phase_ref(route.get("last_goto"))
    reason = "silent default"

    if proposed is not None:
        if is_goto_allowed(current, proposed):
            target = proposed
            reason = f"agent goto → {proposed.value}"
        else:
            logger.warning(
                "rejected goto %s from %s (not in allowlist)",
                proposed.value,
                current.value,
            )
            target = silent_default_next(
                current,
                data,
                planner_count=eff_planners,
                critic_count=eff_critics,
                reviewer_count=eff_reviewers,
                verification_passed=verification_passed,
                tier=tier,
            )
            reason = f"rejected illegal goto; default → {target.value}"
    else:
        target = silent_default_next(
            current,
            data,
            planner_count=eff_planners,
            critic_count=eff_critics,
            reviewer_count=eff_reviewers,
            verification_passed=verification_passed,
            tier=tier,
        )
        reason = f"default → {target.value}"

    target = substitute_auto_skips(
        target,
        data,
        planner_count=eff_planners,
        reviewer_count=eff_reviewers,
    )
    if target != proposed and reason.startswith("agent goto"):
        reason += f" (auto-skip → {target.value})"

    guarded = apply_soft_guards(current, target, data, tier=tier)
    if guarded is Phase.IMPLEMENTATION and target is Phase.REPORTING:
        reason = "soft guard → implementation"
    target = guarded

    target = check_revisit_cap(data, target)
    route.pop("last_goto", None)
    return target, reason


def route_notes_for_role(data: dict[str, Any], role: str) -> str:
    route = data.get("route") or {}
    notes = route.get("prompt_notes") or {}
    return str(notes.get(role, "") or "")
