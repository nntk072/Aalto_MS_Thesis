"""Tests for orchestra pipeline control flow and verification argv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from orchestra.models import Model
from orchestra.pipeline import Pipeline
from orchestra.state import Phase, TaskState, list_tasks


def _dummy() -> Model:
    return Model(
        name="default",
        provider="opencode",
        cli="opencode",
        roles=[
            "triage",
            "planner",
            "critic",
            "synthesizer",
            "implementer",
            "reviewer",
            "review_synthesizer",
        ],
        priority=1,
        quota_daily=0,
        env_var="ORCHESTRA_TEST_KEY",
    )


def test_dry_run_reaches_done(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    pipeline = Pipeline(
        workspace=tmp_path,
        dry_run=True,
        planner_count=1,
        critic_count=1,
        reviewer_count=1,
    )
    pipeline.router.select = (
        lambda role, count=1, task_complexity="medium", task_tier=None, escalate=False: (  # type: ignore[method-assign]
            [dummy] * count
        )
    )
    state = pipeline.run("add a comment")
    assert state.phase == Phase.DONE
    report = Path(state.data["final_report"])
    assert report.is_file()
    text = report.read_text()
    assert "add a comment" in text
    assert "## 1. Triage" in text
    assert "## 9. Verification" in text


def test_complexity_override_applied_before_triage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    pipeline = Pipeline(workspace=tmp_path, dry_run=True, planner_count=1)
    seen: list[str] = []

    def select(
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        seen.append(task_complexity)
        return [dummy] * count

    pipeline.router.select = select  # type: ignore[method-assign]
    state = pipeline.run("add a comment", complexity="trivial")
    assert state.data["complexity"] == "trivial"
    assert state.data["complexity_override"] is True
    assert "trivial" in seen


def test_verification_reenters_fixing(tmp_path) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=True, max_fixes=2)
    calls: list[str] = []

    def fake_fix() -> None:
        calls.append("fix")
        assert pipeline.state is not None
        pipeline.state.data["fix_loop_count"] = pipeline.state.data.get("fix_loop_count", 0) + 1
        pipeline.state.save()

    def fake_verify() -> None:
        calls.append("verify")
        assert pipeline.state is not None
        if pipeline.state.data.get("fix_loop_count", 0) < 2:
            pipeline.state.set_phase(Phase.FIXING)
        else:
            pipeline.state.set_phase(Phase.DONE)

    pipeline._phase_fixing = fake_fix  # type: ignore[method-assign]
    pipeline._phase_verification = fake_verify  # type: ignore[method-assign]
    task_id = "task-test-fix-loop"
    state = TaskState(task_id, "fix me", pipeline.state_dir)
    state.set_phase(Phase.FIXING)
    pipeline.run("fix me", resume_from=task_id)
    assert calls == ["fix", "verify", "fix", "verify"]
    assert pipeline.state is not None
    assert pipeline.state.phase == Phase.DONE


def test_reporting_next_is_done_with_parked_enum() -> None:
    assert Phase.REPORTING.next() == Phase.DONE
    assert Phase.PARKED not in Phase.runnable()


def test_phase_parse_step_aliases() -> None:
    assert Phase.parse("2") is Phase.PLANNING
    assert Phase.parse("planning") is Phase.PLANNING
    assert Phase.parse("step_2") is Phase.PLANNING
    assert Phase.parse("7") is Phase.REVIEW_SYNTHESIS
    assert Phase.parse("8") is Phase.FIXING
    assert Phase.parse("9") is Phase.VERIFICATION
    assert Phase.parse("10") is Phase.REPORTING
    assert Phase.parse("report") is Phase.REPORTING
    assert Phase.parse("verify") is Phase.VERIFICATION


def test_from_requires_resume(tmp_path) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    try:
        pipeline.run("t", start_phase="2")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "--from" in str(exc)


def test_resume_failed_restarts_failed_step(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    pipeline = Pipeline(
        workspace=tmp_path, dry_run=True, planner_count=1, critic_count=1, reviewer_count=1
    )
    pipeline.router.select = (
        lambda role, count=1, task_complexity="medium", task_tier=None, escalate=False: (  # type: ignore[method-assign]
            [dummy] * count
        )
    )
    state = TaskState("task-r", "t", pipeline.state_dir)
    state.set_phase(Phase.VERIFICATION)
    state.set_phase(Phase.FAILED)
    assert state.data["failed_at"] == "verification"
    result = pipeline.run("t", resume_from="task-r")
    assert result.phase == Phase.DONE


def test_resume_from_step_2(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    pipeline = Pipeline(
        workspace=tmp_path, dry_run=True, planner_count=1, critic_count=1, reviewer_count=1
    )
    pipeline.router.select = (
        lambda role, count=1, task_complexity="medium", task_tier=None, escalate=False: (  # type: ignore[method-assign]
            [dummy] * count
        )
    )
    state = TaskState("task-s2", "t", pipeline.state_dir)
    state.set_phase(Phase.CRITIQUE)
    result = pipeline.run("t", resume_from="task-s2", start_phase="2")
    assert result.phase == Phase.DONE


def test_resume_done_without_from_stays_done(tmp_path) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    state = TaskState("task-done", "t", pipeline.state_dir)
    state.set_phase(Phase.DONE)
    result = pipeline.run("t", resume_from="task-done")
    assert result.phase == Phase.DONE


def test_run_verification_uses_interpreter_and_package_paths(tmp_path) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=False)
    recorded: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        capture: bool = True,
        timeout: float | None = None,
        heartbeat: float = 0,
    ) -> subprocess.CompletedProcess[str]:
        recorded.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    pipeline.sessions._run = fake_run  # type: ignore[method-assign]
    pipeline.state = TaskState("task-v", "t", pipeline.state_dir)
    pipeline._run_verification()
    pytest_cmd = recorded[0]
    assert pytest_cmd[0] == sys.executable
    assert pytest_cmd[1:4] == ["-m", "pytest", "tests/"]
    assert pytest_cmd[-2:] == ["-m", "not slow"]
    lint_cmd = recorded[1]
    assert lint_cmd[0] == "ruff"
    assert "quant_rl" in lint_cmd
    assert lint_cmd[-1] != "."


def test_verification_failure_enters_fix_loop_even_if_review_passed(tmp_path) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=False, max_fixes=2)
    pipeline.state = TaskState("task-vfail", "t", pipeline.state_dir)
    pipeline.state.data["review_verdict"] = "pass"
    pipeline._run_verification = lambda: {  # type: ignore[method-assign]
        "tests": {"success": False, "output": "FAILED tests/foo.py::test_x"}
    }
    pipeline._phase_verification()
    assert pipeline.state.phase == Phase.FIXING


def test_fixing_skips_when_review_passed_and_tests_not_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    dummy.roles = [*dummy.roles, "fixer"]
    selected: list[str] = []

    def select(
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        selected.append(role)
        return [dummy]

    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    pipeline.router.select = select  # type: ignore[method-assign]
    pipeline.state = TaskState("task-skipfix", "t", pipeline.state_dir)
    pipeline.state.data["review_verdict"] = "pass"
    pipeline.state.data["review_synthesis_output"] = "lgtm"
    pipeline._phase_fixing()
    assert selected == []


def test_fixing_runs_when_verification_already_failed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRA_TEST_KEY", "1")
    dummy = _dummy()
    dummy.roles = [*dummy.roles, "fixer"]
    selected: list[str] = []

    def select(
        role: str,
        count: int = 1,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        selected.append(role)
        return [dummy]

    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    pipeline.router.select = select  # type: ignore[method-assign]
    pipeline.state = TaskState("task-dofix", "t", pipeline.state_dir)
    pipeline.state.data["review_verdict"] = "pass"
    pipeline.state.data["review_synthesis_output"] = "lgtm"
    pipeline.state.data["verification_result"] = {
        "tests": {"success": False, "output": "FAILED tests/foo.py::test_x"}
    }
    pipeline._phase_fixing()
    assert selected == ["implementer"]


def test_list_tasks_ignores_costs_json(tmp_path) -> None:
    (tmp_path / "costs.json").write_text("{}")
    (tmp_path / "usage.json").write_text("{}")
    (tmp_path / "task-1.json").write_text(
        '{"task_id": "task-1", "phase": "done", "task_description": "hello"}'
    )
    tasks = list_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "task-1"


def test_invoke_required_falls_back_when_primary_fails(tmp_path, monkeypatch) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=False)
    pipeline.state = TaskState("task-fb1", "t", pipeline.state_dir)
    primary = _dummy()
    fallback = Model(
        name="backup",
        provider="other",
        cli="cli",
        roles=["triage"],
        priority=1,
        quota_daily=0,
        env_var="ORCHESTRA_TEST_KEY",
    )
    seen: dict[str, object] = {}

    def fake_chain(
        model: Model,
        role: str,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        seen["role"] = role
        return [fallback]

    pipeline.router.fallback_chain = fake_chain  # type: ignore[assignment]
    attempts: list[str] = []

    def fake_invoke(
        model: Model,
        role: str,
        prompt: str,
        timeout: int,
        lines: int = 0,
        native_session_id: str | None = None,
    ) -> tuple[str, str]:
        attempts.append(model.display_name)
        if model is primary:
            raise TimeoutError("primary timed out")
        return "session-backup", "output from backup"

    pipeline._invoke = fake_invoke  # type: ignore[method-assign]
    session, output = pipeline._invoke_required(primary, "triage", "prompt", 600)
    assert (session, output) == ("session-backup", "output from backup")
    assert attempts.count(primary.display_name) >= 1
    assert fallback.display_name in attempts
    assert seen["role"] == "triage"


def test_invoke_required_uses_select_role_for_fallback(tmp_path, monkeypatch) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=False)
    pipeline.state = TaskState("task-fb3", "t", pipeline.state_dir)
    primary = _dummy()
    fallback = Model(
        name="backup",
        provider="other",
        cli="cli",
        roles=["implementer"],
        priority=1,
        quota_daily=0,
        env_var="ORCHESTRA_TEST_KEY",
    )
    seen: dict[str, object] = {}

    def fake_chain(
        model: Model,
        role: str,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        seen["role"] = role
        return [fallback]

    pipeline.router.fallback_chain = fake_chain  # type: ignore[assignment]
    attempts: list[str] = []

    def fake_invoke(
        model: Model,
        role: str,
        prompt: str,
        timeout: int,
        lines: int = 0,
        native_session_id: str | None = None,
    ) -> tuple[str, str]:
        attempts.append(model.display_name)
        assert role == "fixer"
        if model is primary:
            raise TimeoutError("primary timed out")
        return "session-backup", "output from backup"

    pipeline._invoke = fake_invoke  # type: ignore[method-assign]
    session, output = pipeline._invoke_required(
        primary, "fixer", "prompt", 600, select_role="implementer"
    )
    assert (session, output) == ("session-backup", "output from backup")
    assert attempts.count(primary.display_name) >= 1
    assert fallback.display_name in attempts
    assert seen["role"] == "implementer"


def test_invoke_required_raises_when_all_models_fail(tmp_path, monkeypatch) -> None:
    pipeline = Pipeline(workspace=tmp_path, dry_run=False)
    pipeline.state = TaskState("task-fb2", "t", pipeline.state_dir)
    primary = _dummy()
    fallback = Model(
        name="backup",
        provider="other",
        cli="cli",
        roles=["triage"],
        priority=1,
        quota_daily=0,
        env_var="ORCHESTRA_TEST_KEY",
    )

    def fake_chain(
        model: Model,
        role: str,
        task_complexity: str = "medium",
        task_tier: str | None = None,
        escalate: bool = False,
    ) -> list[Model]:
        return [fallback]

    pipeline.router.fallback_chain = fake_chain  # type: ignore[assignment]

    def fake_invoke(
        model: Model,
        role: str,
        prompt: str,
        timeout: int,
        lines: int = 0,
        native_session_id: str | None = None,
    ) -> tuple[str, str]:
        raise RuntimeError("agent failed")

    pipeline._invoke = fake_invoke  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        pipeline._invoke_required(primary, "triage", "prompt", 600)
