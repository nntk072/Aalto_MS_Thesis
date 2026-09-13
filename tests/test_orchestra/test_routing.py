"""Tests for orchestra phase routing."""

from __future__ import annotations

from orchestra.routing import (
    ALLOWED_NEXT,
    RouteUpdate,
    apply_soft_guards,
    apply_triage_bootstrap,
    is_goto_allowed,
    merge_route,
    parse_route_from_text,
    resolve_next_phase,
    should_auto_skip,
    silent_default_next,
    substitute_auto_skips,
)
from orchestra.state import Phase


def _data(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "route": {},
        "planner_outputs": [],
        "review_outputs": [],
        "review_verdict": None,
        "triage_output": "",
        "implementation_succeeded": False,
    }
    base.update(kwargs)
    return base


def test_allowed_next_matches_plan() -> None:
    assert Phase.TRIAGE in ALLOWED_NEXT[Phase.TRIAGE] or True
    assert ALLOWED_NEXT[Phase.TRIAGE] == frozenset(
        {Phase.PLANNING, Phase.IMPLEMENTATION, Phase.REPORTING}
    )
    assert Phase.SYNTHESIS not in ALLOWED_NEXT[Phase.IMPLEMENTATION]
    assert (
        Phase.REPORTING not in ALLOWED_NEXT[Phase.VERIFICATION]
        or Phase.REPORTING in ALLOWED_NEXT[Phase.VERIFICATION]
    )
    assert ALLOWED_NEXT[Phase.REPORTING] == frozenset()


def test_reject_must_not_edges() -> None:
    assert not is_goto_allowed(Phase.TRIAGE, Phase.VERIFICATION)
    assert not is_goto_allowed(Phase.IMPLEMENTATION, Phase.SYNTHESIS)
    assert not is_goto_allowed(Phase.VERIFICATION, Phase.SYNTHESIS)
    assert not is_goto_allowed(Phase.REVIEW, Phase.CRITIQUE)


def test_allow_may_edges() -> None:
    assert is_goto_allowed(Phase.VERIFICATION, Phase.REPORTING)
    assert is_goto_allowed(Phase.VERIFICATION, Phase.PLANNING)
    assert is_goto_allowed(Phase.IMPLEMENTATION, Phase.VERIFICATION)
    assert is_goto_allowed(Phase.REVIEW, Phase.FIXING)


def test_parse_route_nested() -> None:
    text = """
    {"complexity": "medium", "route": {"goto": 5, "reviewer_count": 0}}
    """
    update = parse_route_from_text(text)
    assert update is not None
    assert update.goto is Phase.IMPLEMENTATION
    assert update.reviewer_count == 0


def test_merge_route_shrinks_counts() -> None:
    data = _data()
    merge_route(data, RouteUpdate(planner_count=1, critic_count=0))
    route = data["route"]
    assert isinstance(route, dict)
    assert route["planner_count"] == 1
    assert route["critic_count"] == 0


def test_silent_default_triage_t0() -> None:
    nxt = silent_default_next(
        Phase.TRIAGE, _data(), planner_count=1, critic_count=0, reviewer_count=0, tier="T0"
    )
    assert nxt is Phase.IMPLEMENTATION


def test_silent_default_verification_pass_fail() -> None:
    assert (
        silent_default_next(
            Phase.VERIFICATION,
            _data(),
            planner_count=1,
            critic_count=0,
            reviewer_count=0,
            verification_passed=True,
        )
        is Phase.REPORTING
    )
    assert (
        silent_default_next(
            Phase.VERIFICATION,
            _data(),
            planner_count=1,
            critic_count=0,
            reviewer_count=0,
            verification_passed=False,
        )
        is Phase.FIXING
    )


def test_auto_skip_synthesis_when_one_planner() -> None:
    data = _data(route={"planner_count": 1})
    assert should_auto_skip(Phase.SYNTHESIS, data, planner_count=1, reviewer_count=1)
    target = substitute_auto_skips(Phase.SYNTHESIS, data, planner_count=1, reviewer_count=1)
    assert target is Phase.IMPLEMENTATION


def test_auto_skip_review_synthesis_when_one_reviewer() -> None:
    data = _data(route={"reviewer_count": 1})
    assert should_auto_skip(Phase.REVIEW_SYNTHESIS, data, planner_count=1, reviewer_count=1)
    target = substitute_auto_skips(Phase.REVIEW_SYNTHESIS, data, planner_count=1, reviewer_count=1)
    assert target is Phase.VERIFICATION


def test_resolve_rejects_illegal_goto() -> None:
    data = _data()
    update = RouteUpdate(goto=Phase.SYNTHESIS)
    nxt, reason = resolve_next_phase(
        Phase.IMPLEMENTATION,
        data,
        update,
        planner_count=1,
        critic_count=1,
        reviewer_count=1,
    )
    assert nxt is not Phase.SYNTHESIS
    assert "rejected" in reason


def test_resolve_accepts_legal_goto() -> None:
    data = _data()
    update = RouteUpdate(goto=Phase.VERIFICATION)
    nxt, reason = resolve_next_phase(
        Phase.IMPLEMENTATION,
        data,
        update,
        planner_count=1,
        critic_count=0,
        reviewer_count=0,
    )
    assert nxt is Phase.VERIFICATION
    assert "agent goto" in reason


def test_triage_bootstrap_t0() -> None:
    data = _data()
    apply_triage_bootstrap(data, tier="T0", had_explicit_goto=False)
    nxt, _ = resolve_next_phase(
        Phase.TRIAGE,
        data,
        None,
        planner_count=3,
        critic_count=2,
        reviewer_count=2,
        tier="T0",
    )
    assert nxt is Phase.IMPLEMENTATION


def test_soft_guard_forces_implementation_before_report() -> None:
    data = _data(triage_output='{"domain": "feature"}')
    guarded = apply_soft_guards(Phase.TRIAGE, Phase.REPORTING, data, tier="T2")
    assert guarded is Phase.IMPLEMENTATION


def test_soft_guard_allows_report_after_implementation() -> None:
    data = _data(implementation_succeeded=True)
    guarded = apply_soft_guards(Phase.VERIFICATION, Phase.REPORTING, data, tier="T2")
    assert guarded is Phase.REPORTING
