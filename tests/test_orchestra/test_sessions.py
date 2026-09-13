"""Tests for tmux-free spawn quoting and collect() failure handling."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from orchestra.agent_runner import run_job
from orchestra.models import CODEX_EXEC_EXTRA_ARGS, Model, load_models
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


def _codex() -> Model:
    return Model(
        name="gpt-5.6-luna",
        provider="openai",
        cli="codex",
        roles=["planner"],
        priority=1,
        quota_daily=0,
        model_flag="-m",
        effort_flag="codex_reasoning",
        effort_support=["none", "low", "medium", "high"],
        extra_args=list(CODEX_EXEC_EXTRA_ARGS),
    )


def test_build_command_codex_uses_model_and_capped_reasoning() -> None:
    cmd = SessionManager.build_command(_codex(), "plan this", effort="high")
    assert cmd[:2] == ["codex", "exec"]
    assert cmd[2 : 2 + len(CODEX_EXEC_EXTRA_ARGS)] == CODEX_EXEC_EXTRA_ARGS
    model_at = cmd.index("-m")
    assert cmd[model_at : model_at + 2] == ["-m", "gpt-5.6-luna"]
    cfg_at = cmd.index("-c")
    assert cmd[cfg_at : cfg_at + 2] == ["-c", "model_reasoning_effort=high"]
    assert cmd[-1] == "plan this"
    assert "xhigh" not in cmd
    assert "--ask-for-approval" not in cmd
    assert "--no-alt-screen" not in cmd
    assert "--sandbox" not in cmd


def test_codex_yaml_extra_args_match_codex_exec_cli() -> None:
    yaml_codex = next(m for m in load_models() if m.cli == "codex")
    extra = yaml_codex.extra_args
    assert extra == list(CODEX_EXEC_EXTRA_ARGS)
    assert "--ask-for-approval" not in extra
    assert "--no-alt-screen" not in extra
    # `--approve-for-me` already implies workspace-write and conflicts with `--sandbox`.
    assert not ("--sandbox" in extra and "--approve-for-me" in extra)
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        pytest.skip("codex CLI not installed")
    # `--help` skips clap conflict checks; empty stdin only exercises argv parsing.
    probe = subprocess.run(
        [codex_bin, "exec", *extra],
        capture_output=True,
        text=True,
        timeout=5,
        stdin=subprocess.DEVNULL,
    )
    err = probe.stderr or probe.stdout
    assert "cannot be used with" not in err
    assert "unexpected argument" not in err
    assert probe.returncode != 2, err


def test_build_command_codex_rejects_uncapped_reasoning() -> None:
    with pytest.raises(ValueError, match="Unsupported Codex reasoning effort"):
        SessionManager.build_command(_codex(), "plan this", effort="xhigh")


def test_codex_effort_resolves_by_role() -> None:
    assert SessionManager.resolve_effort(_codex(), "T2", role="planner") == "medium"
    assert SessionManager.resolve_effort(_codex(), "T3", role="implementer") == "high"


def test_spawn_writes_prompt_file_not_tmux_argv(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture: bool = True, timeout: float | None = None, heartbeat: float = 0
    ) -> subprocess.CompletedProcess[str]:
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
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    sm._run = fake_run  # type: ignore[method-assign]
    sm.spawn(_opencode(), "planner", "plan this", keep_alive=True)
    flat = [" ".join(c) for c in calls]
    assert any("remain-on-exit" in s for s in flat)
    assert any("respawn-pane" in s for s in flat)


def test_collect_raises_on_nonzero_exit(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
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
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
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
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
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


def test_build_command_vibe_uses_env_not_agent() -> None:
    model = Model(
        name="mistral-medium-3.5",
        provider="mistral",
        cli="vibe",
        roles=["planner"],
        priority=2,
        quota_daily=0,
        alias="mistral-medium-3.5",
        extra_args=["--auto-approve"],
    )
    env = SessionManager.build_command_env(model)
    assert env["VIBE_ACTIVE_MODEL"] == "mistral-medium-3.5"
    cmd = SessionManager.build_command(model, "hello")
    assert "--agent" not in cmd
    assert "mistral-medium-3.5" not in cmd


def test_build_command_cline_includes_thinking() -> None:
    model = Model(
        name="z-ai/glm-5.3-flash",
        provider="cline",
        cli="cline",
        roles=["planner"],
        priority=2,
        quota_daily=0,
        model_flag="-m",
        effort_flag="cline_thinking",
        extra_args=["--json"],
    )
    cmd = SessionManager.build_command(model, "plan this", effort="high")
    assert "cline" in cmd
    assert "--thinking" in cmd
    assert "high" in cmd
    assert "-m" in cmd


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


def test_build_command_kilo_skips_variant_none() -> None:
    model = Model(
        name="default",
        provider="kilo",
        cli="kilo",
        roles=["review_synthesizer"],
        priority=3,
        quota_daily=0,
        extra_args=["--auto"],
        effort_flag="oc_variant",
    )
    cmd = SessionManager.build_command(model, "hi", effort="none")
    assert cmd[:3] == ["kilo", "run", "--auto"]
    assert "--variant" not in cmd
    cmd_high = SessionManager.build_command(model, "hi", effort="high")
    assert cmd_high[cmd_high.index("--variant") + 1] == "high"


def test_collect_timeout_includes_log_tail(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
        cmd, 0, stdout="", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("still grepping plots.py\n")
    sm.is_running = lambda _name: True  # type: ignore[assignment]
    try:
        sm.collect(session, timeout=0, idle_timeout=0, lines=20)
        raise AssertionError("expected TimeoutError")
    except TimeoutError as exc:
        assert "timed out" in str(exc)
        assert "still grepping plots.py" in str(exc)


def test_collect_success_kills_session(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture: bool = True, timeout: float | None = None, heartbeat: float = 0
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess[str](cmd, 0, stdout="", stderr="")

    sm._run = fake_run  # type: ignore[method-assign]
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("review synthesis done\n")
    (run_dir / "exit_code").write_text("0")
    sm.collect(session, timeout=1, idle_timeout=0)
    assert any(c[:3] == ["tmux", "kill-session", "-t"] and c[3] == session for c in calls)


def test_collect_keep_alive_leaves_successful_session(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], capture: bool = True, timeout: float | None = None, heartbeat: float = 0
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess[str](cmd, 0, stdout="", stderr="")

    sm._run = fake_run  # type: ignore[method-assign]
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text("ok\n")
    (run_dir / "exit_code").write_text("0")
    sm.collect(session, timeout=1, idle_timeout=0, keep_alive=True)
    assert not any(c[:2] == ["tmux", "kill-session"] for c in calls)


def test_cleanup_filters_task_and_role(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=True)
    names = [
        "orchestra-review_synthesizer-kilo-default-kilo-task-1-11",
        "orchestra-review_synthesizer-opencode-default-opencode-task-1-12",
        "orchestra-planner-kilo-default-kilo-task-1-13",
        "orchestra-review_synthesizer-kilo-default-kilo-task-2-14",
    ]
    sm.list_sessions = lambda: names  # type: ignore[method-assign]
    killed: list[str] = []

    def fake_kill(session_name: str) -> None:
        killed.append(session_name)

    sm.kill = fake_kill  # type: ignore[method-assign]
    assert sm.cleanup(task_id="task-1", role="review_synthesizer") == 2
    assert killed == names[:2]


def test_wait_for_idle_vanished_session_fails(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm.session_exists = lambda _n: False  # type: ignore[assignment]
    sm.exit_code = lambda _n: None  # type: ignore[assignment]
    sm.is_running = lambda _n: False  # type: ignore[assignment]
    assert sm.wait_for_idle("gone", timeout=20, poll_interval=1) is False


def test_wait_for_idle_aborts_on_transport_error(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
        cmd, 0, stdout="0\n", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text(
        "Cannot connect to API: The socket connection was closed unexpectedly (ECONNRESET)\n"
    )
    sm.is_running = lambda _name: True  # type: ignore[assignment]
    sm.session_exists = lambda _n: True  # type: ignore[assignment]
    sm.exit_code = lambda _n: None  # type: ignore[assignment]
    assert sm.wait_for_idle(session, timeout=20, poll_interval=0, idle_timeout=120) is False


def test_inspect_live_log_ignores_quoted_transport_errors() -> None:
    from orchestra.failures import inspect_live_log

    nested = (
        '{"ts":"2026-09-13T18:28:15.325Z","type":"agent_event","event":'
        '{"type":"content_end","contentType":"tool","output":'
        '"failed to connect to websocket: HTTP error: 307 Temporary Redirect"}}'
    )
    assert inspect_live_log(nested) is None
    assert inspect_live_log('{"type":"error","message":"ECONNRESET"}') == "abort"


def test_wait_for_idle_completes_on_cline_done_event(tmp_path) -> None:
    from orchestra.failures import agent_completed

    text = (
        '{"ts":"2026-09-13T18:08:55.116Z","type":"agent_event",'
        '"event":{"type":"done","reason":"completed","text":"audit report"}}'
    )
    assert agent_completed(text.lower()) is True
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
        cmd, 0, stdout="0\n", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text(text + "\n")
    sm.is_running = lambda _name: True  # type: ignore[assignment]
    sm.session_exists = lambda _n: True  # type: ignore[assignment]
    sm.exit_code = lambda _n: None  # type: ignore[assignment]
    assert sm.wait_for_idle(session, timeout=20, poll_interval=0, idle_timeout=120) is True


def test_wait_for_idle_aborts_on_codex_usage_limit(tmp_path) -> None:
    sm = SessionManager(workspace=tmp_path, dry_run=False)
    sm._run = lambda cmd, capture=True, timeout=None, heartbeat=0: subprocess.CompletedProcess[str](  # type: ignore[method-assign]
        cmd, 0, stdout="0\n", stderr=""
    )
    session = sm.spawn(_opencode(), "planner", "hi", task_id="t")
    run_dir = tmp_path / "orchestra" / "state" / "runs" / session
    (run_dir / "output.log").write_text(
        "ERROR: You've hit your usage limit. try again at Oct 13th, 2026 3:06 PM.\n"
    )
    sm.is_running = lambda _name: True  # type: ignore[assignment]
    sm.session_exists = lambda _n: True  # type: ignore[assignment]
    sm.exit_code = lambda _n: None  # type: ignore[assignment]
    assert sm.wait_for_idle(session, timeout=20, poll_interval=0, idle_timeout=120) is False


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
        staticmethod(
            lambda _m, _p, prompt_file=None, effort=None, native_session_id=None: [
                sys.executable,
                "-c",
                "print('ran-ok')",
            ]
        ),
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
