"""Tests for orchestra pipeline control flow and verification argv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    pipeline.router.select = lambda role, count=1, **kw: [dummy] * count  # type: ignore[method-assign]
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

    def select(role, count=1, task_complexity="medium", **kw):
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
        pipeline.state.data["fix_loop_count"] = pipeline.state.data.get("fix_loop_count", 0) + 1
        pipeline.state.save()

    def fake_verify() -> None:
        calls.append("verify")
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
    assert pipeline.state.phase == Phase.DONE


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
    pipeline.router.select = lambda role, count=1, **kw: [dummy] * count  # type: ignore[method-assign]
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
    pipeline.router.select = lambda role, count=1, **kw: [dummy] * count  # type: ignore[method-assign]
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
    ) -> subprocess.CompletedProcess:
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
    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    pipeline.router.select = lambda role, count=1, **kw: selected.append(role) or [dummy]  # type: ignore[method-assign]
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
    pipeline = Pipeline(workspace=tmp_path, dry_run=True)
    pipeline.router.select = lambda role, count=1, **kw: selected.append(role) or [dummy]  # type: ignore[method-assign]
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
