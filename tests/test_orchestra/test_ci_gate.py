"""Tests for orchestra CI gate definitions."""

from __future__ import annotations

from orchestra.ci_gate import CI_CHECKS, first_failure, gate_passed


def test_ci_checks_match_github_ci_jobs() -> None:
    names = [c.name for c in CI_CHECKS]
    assert names == ["format", "lint", "typecheck", "tests"]
    assert CI_CHECKS[-1].argv == ["uv", "run", "pytest", "tests/", "-v"]
    assert "-m" not in CI_CHECKS[-1].argv
    assert "slow" not in CI_CHECKS[-1].argv


def test_render_ci_verification_includes_all_commands() -> None:
    from orchestra.prompts import render_ci_verification

    block = render_ci_verification("/tmp/ws")
    assert "/tmp/ws" in block
    assert "uv run mypy ." in block
    assert "uv run pytest tests/ -v" in block
    assert '-m "not slow"' not in block.split("Do not")[0]


def test_gate_passed_requires_all_checks() -> None:
    ok = {c.name: {"success": True, "output": ""} for c in CI_CHECKS}
    assert gate_passed(ok) is True
    ok["lint"] = {"success": False, "output": "error"}
    assert gate_passed(ok) is False
    assert first_failure(ok) == "lint"
