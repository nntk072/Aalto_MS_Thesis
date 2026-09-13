"""Execute a CLI coding agent from a job spec file (no shell)."""

from __future__ import annotations

import errno
import json
import os
import pty
import sys
from pathlib import Path

from .models import Model
from .sessions import SessionManager


def _preview_cmd(cmd: list[str], prompt: str) -> str:
    shown = ["<prompt>" if a == prompt else a for a in cmd]
    return "[orchestra] exec " + " ".join(shown)


def _exit_from_waitstatus(status: int) -> int:
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _run_with_pty(cmd: list[str], log_file: Path, cwd: str | None) -> int:
    """Run ``cmd`` in a PTY so TUI CLIs still write output, tee'd to the log."""
    prev = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        with log_file.open("ab") as log:

            def reader(fd: int) -> bytes:
                data = os.read(fd, 4096)
                log.write(data)
                log.flush()
                return data

            status = pty.spawn(cmd, reader)
    finally:
        os.chdir(prev)
    return _exit_from_waitstatus(status)


def run_job(job_path: Path) -> int:
    """Run one agent job. Writes log + exit-code files. Returns the exit code."""
    job = json.loads(job_path.read_text())
    model = Model(**job["model"])
    prompt = Path(job["prompt_file"]).read_text()
    log_file = Path(job["log_file"])
    exit_file = Path(job["exit_file"])
    cwd = job.get("cwd") or None
    log_file.parent.mkdir(parents=True, exist_ok=True)
    for key, value in (job.get("env") or {}).items():
        os.environ[str(key)] = str(value)
    cmd = SessionManager.build_command(
        model,
        prompt,
        prompt_file=job.get("prompt_file"),
        effort=job.get("effort"),
        native_session_id=job.get("native_session_id"),
    )
    log_file.write_text(_preview_cmd(cmd, prompt) + "\n", encoding="utf-8")
    code = 1
    try:
        try:
            code = _run_with_pty(cmd, log_file, cwd)
        except OSError as exc:
            if exc.errno == errno.E2BIG:
                short = (
                    "Read the full prompt from this file and follow it exactly:\n"
                    f"{job['prompt_file']}\n"
                )
                cmd = SessionManager.build_command(
                    model,
                    short,
                    prompt_file=job.get("prompt_file"),
                    effort=job.get("effort"),
                    native_session_id=job.get("native_session_id"),
                )
                code = _run_with_pty(cmd, log_file, cwd)
            else:
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(f"[orchestra] exec failed: {exc}\n")
                code = 127
    except OSError as exc:
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"[orchestra] exec failed: {exc}\n")
        code = 127
    exit_file.write_text(str(code), encoding="utf-8")
    return code


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python -m orchestra.agent_runner JOB.json\n")
        sys.exit(2)
    sys.exit(run_job(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
