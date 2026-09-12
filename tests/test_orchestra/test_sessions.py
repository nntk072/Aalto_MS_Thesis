"""Tests for tmux-free spawn quoting and collect() failure handling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from orchestra.agent_runner import run_job
from orchestra.models import Model
from orchestra.sessions import SessionManager


def _opencode() -> Model:
    return Model(
        name="default",
        provider="opencode",
        cli="opencode",
        roles=["planner"],
        priority=2,
        quota_daily=0,
    )


def test_spawn_writes_prompt_file_not_tmux_argv(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture: bool = True, timeout: float | None = None, heartbeat: float = 0
    ) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    sm._run = fake_run  # type: ignore[method-assign]
    prompt = '```json\n{"complexity": "trivial|medium|complex"}\n```\n'
    session = sm.spawn(_opencode(), "triage", prompt, task_id="task-1")
    joined = " ".join(" ".join(c) for c in calls)
    assert "trivial|medium|complex" not in joined
    assert "orchestra.agent_runner" in joined
    assert "--" in calls[0]
    job_path = tmp_path / "orchestra" / "state" / "runs" / session / "job.json"
    spec = json.loads(job_path.read_text())
    assert Path(spec["prompt_file"]).read_text() == prompt


def test_spawn_keep_alive_sets_remain_on_exit(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture: bool = True, timeout: float | None = None, heartbeat: float = 0
    ) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    sm._run = fake_run  # type: ignore[method-assign]
    sm.spawn(_opencode(), "planner", "plan this", keep_alive=True)
    flat = [" ".join(c) for c in calls]
    assert any("remain-on-exit" in s for s in flat)
    assert any("respawn-pane" in s for s in flat)


def test_collect_raises_on_nonzero_exit(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess(  # type: ignore[method-assign]
        cmd, 0, stdout="", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("bash: json: command not found\n")
    (run_dir / "exit_code").write_text("1")
    try:
        sm.collect(session, timeout=1, lines=50)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "exited 1" in str(exc)


def test_collect_raises_on_empty_output(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess(  # type: ignore[method-assign]
        cmd, 0, stdout="", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("   \n")
    (run_dir / "exit_code").write_text("0")
    try:
        sm.collect(session, timeout=1, lines=50)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "empty output" in str(exc)


def test_collect_raises_on_exec_header_only(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess(  # type: ignore[method-assign]
        cmd, 0, stdout="", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("[orchestra] exec opencode run --auto <prompt>\n")
    (run_dir / "exit_code").write_text("0")
    try:
        sm.collect(session, timeout=1, lines=0)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "empty output" in str(exc)


def test_build_command_vibe_does_not_pass_model_as_agent() -> None:
    model = Model(
        name="devstral-local",
        provider="ollama",
        cli="vibe",
        roles=["planner"],
        priority=3,
        quota_daily=0,
        alias="",
        extra_args=["--auto-approve"],
    )
    cmd = SessionManager.build_command(model, "hello")
    assert cmd == ["vibe", "--auto-approve", "-p", "hello"]
    assert "devstral-local" not in cmd


def test_build_command_opencode_puts_flags_before_prompt() -> None:
    model = _opencode()
    model.extra_args = ["--auto"]
    cmd = SessionManager.build_command(model, "the real prompt", prompt_file="/tmp/prompt.md")
    assert cmd[:3] == ["opencode", "run", "--auto"]
    assert cmd[-1] == "the real prompt"
    assert "--file" not in cmd
    assert "/tmp/prompt.md" not in cmd


def test_collect_timeout_includes_log_tail(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess(  # type: ignore[method-assign]
        cmd, 0, stdout="", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("still grepping plots.py\n")
    sm.is_running = lambda _name: True  # type: ignore[method-assign]
    try:
        sm.collect(session, timeout=0, idle_timeout=0, lines=20)
        raise AssertionError("expected TimeoutError")
    except TimeoutError as exc:
        assert "timed out" in str(exc)
        assert "still grepping plots.py" in str(exc)


def test_wait_for_idle_vanished_session_fails(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm.session_exists = lambda _n: False  # type: ignore[method-assign]
    sm.exit_code = lambda _n: None  # type: ignore[method-assign]
    sm.is_running = lambda _n: False  # type: ignore[method-assign]
    assert sm.wait_for_idle("gone", timeout=20, poll_interval=1) is False


def test_agent_runner_execs_without_shell(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("```json\n{}\n```")
    log_file = tmp_path / "out.log"
    exit_file = tmp_path / "exit"
    job_file = tmp_path / "job.json"
    model = _opencode()
    job_file.write_text(
        json.dumps(
            {
                "model": asdict(model),
                "prompt_file": str(prompt_file),
                "log_file": str(log_file),
                "exit_file": str(exit_file),
                "cwd": str(tmp_path),
            }
        )
    )
    monkeypatch.setattr(
        "orchestra.sessions.SessionManager.build_command",
        staticmethod(lambda _m, _p, prompt_file=None: [sys.executable, "-c", "print('ran-ok')"]),
    )
    before = os.getcwd()
    assert run_job(job_file) == 0
    assert os.getcwd() == before
    assert "ran-ok" in log_file.read_text()
    assert exit_file.read_text().strip() == "0"


def test_run_heartbeat_times_out(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    result = sm._run(["sleep", "30"], heartbeat=0.2, timeout=0.6)
    assert result.returncode == 124
    assert result.stderr == "timeout"
