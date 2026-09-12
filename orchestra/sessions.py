"""tmux session management for CLI agents."""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .models import Model

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


class SessionManager:
    """Manages tmux sessions for agent execution."""

    def __init__(self, workspace: Path, dry_run: bool = False):
        self.workspace = workspace
        self.dry_run = dry_run
        self.active_sessions: dict[str, dict] = {}
        self.runs_dir = workspace / "orchestra" / "state" / "runs"

    @staticmethod
    def display_session_name(role: str, model: Model, task_id: str = "") -> str:
        """Build a human-readable tmux session name.

        Format: ``orchestra-{role}-{provider}-{model}-{cli}[-{task_id}]-{ts}``
        so ``tmux ls`` and the Oasis GUI show role + model + CLI at a glance.
        """
        safe_model = model.name.replace("/", "-").replace(":", "-").replace(".", "-")
        parts = ["orchestra", role, model.provider, safe_model, model.cli]
        if task_id:
            parts.append(task_id)
        parts.append(str(int(time.time())))
        return "-".join(parts)

    def _run(
        self,
        cmd: list[str],
        capture: bool = True,
        timeout: float | None = None,
        heartbeat: float = 0,
    ) -> subprocess.CompletedProcess:
        """Run a command as argv (never through a shell)."""
        if self.dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if heartbeat > 0:
            return self._run_heartbeat(cmd, timeout=timeout, heartbeat=heartbeat)
        try:
            return subprocess.run(
                cmd,
                capture_output=capture,
                text=True,
                cwd=str(self.workspace),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="timeout")

    def _run_heartbeat(
        self,
        cmd: list[str],
        timeout: float | None,
        heartbeat: float,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd``, streaming stdout and printing a beat when it is silent."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.workspace),
            start_new_session=True,
        )
        assert proc.stdout is not None
        chunks: list[str] = []
        start = time.time()
        last_beat = start
        last_out = start
        while True:
            now = time.time()
            if timeout is not None and now - start > timeout:
                self._kill_pg(proc)
                proc.wait()
                return subprocess.CompletedProcess(cmd, 124, "".join(chunks), "timeout")
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    chunks.append(line)
                    last_out = now
                    print(line, end="", flush=True)
                elif proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
                    print(rest, end="", flush=True)
                break
            if now - last_out >= heartbeat and now - last_beat >= heartbeat:
                print(f"    still running ({int(now - start)}s, no new output)", flush=True)
                last_beat = now
        return subprocess.CompletedProcess(cmd, proc.wait(), "".join(chunks), "")

    @staticmethod
    def _kill_pg(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()

    def session_exists(self, name: str) -> bool:
        """Check if a tmux session exists."""
        result = self._run(["tmux", "has-session", "-t", name], timeout=10)
        return result.returncode == 0

    def list_sessions(self) -> list[str]:
        """List all tmux session names."""
        result = self._run(["tmux", "list-sessions", "-F", "#{session_name}"], timeout=10)
        if result.returncode != 0:
            return []
        return result.stdout.strip().splitlines()

    @staticmethod
    def build_command(
        model: Model,
        prompt: str,
        prompt_file: str | None = None,
    ) -> list[str]:
        """Build the CLI argv for a model (prompt is one argv token, no shell).

        When ``prompt_file`` is set it is reserved for CLIs that take a path
        flag; OpenCode/Kilo get flags first, then the prompt as the message
        (their ``--file`` option is a greedy array and must not precede text).
        """
        if model.cli == "gemini":
            cmd = ["gemini", "-p", prompt]
            if model.name and model.model_flag:
                cmd.extend([model.model_flag, model.name])
            cmd.extend(model.extra_args)
        elif model.cli == "vibe":
            # `--agent` is a vibe permission profile (ask/plan/auto-approve),
            # not an Ollama model name. Flags before `-p` so `-p` always
            # consumes the prompt text.
            cmd = ["vibe", *model.extra_args]
            if model.alias:
                cmd.extend(["--agent", model.alias])
            cmd.extend(["-p", prompt])
        elif model.cli == "opencode":
            # Flags before the message: `--file` is a greedy yargs array and
            # would swallow the prompt text as extra filenames.
            cmd = ["opencode", "run", *model.extra_args]
            if model.name and model.name != "default":
                cmd.extend([model.model_flag or "--model", model.name])
            cmd.append(prompt)
        elif model.cli == "kilo":
            cmd = ["kilo", "run", *model.extra_args]
            if model.name and model.name != "default":
                cmd.extend([model.model_flag or "--model", model.name])
            cmd.append(prompt)
        else:
            raise ValueError(f"Unknown CLI: {model.cli}")
        return cmd

    def _job_dir(self, session_name: str) -> Path:
        path = self.runs_dir / session_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def spawn(
        self,
        model: Model,
        role: str,
        prompt: str,
        session_name: str | None = None,
        task_id: str = "",
        keep_alive: bool = False,
    ) -> str:
        """Spawn a tmux session that runs ``orchestra.agent_runner`` on a job file.

        The prompt is written to disk and never interpolated into a shell string,
        so JSON, backticks, and newlines cannot be executed as bash.

        When ``keep_alive`` is True, ``remain-on-exit`` is set before the runner
        starts so the pane stays visible after the agent exits. Completion is
        detected via the exit-code file / ``#{pane_dead}``, not shell liveness.
        """
        if session_name is None:
            session_name = self.display_session_name(role, model, task_id)

        job_dir = self._job_dir(session_name)
        prompt_file = job_dir / "prompt.md"
        log_file = job_dir / "output.log"
        exit_file = job_dir / "exit_code"
        job_file = job_dir / "job.json"
        prompt_file.write_text(prompt, encoding="utf-8")
        job_file.write_text(
            json.dumps(
                {
                    "model": asdict(model),
                    "prompt_file": str(prompt_file),
                    "log_file": str(log_file),
                    "exit_file": str(exit_file),
                    "cwd": str(self.workspace),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        runner_cmd = [
            sys.executable,
            "-m",
            "orchestra.agent_runner",
            str(job_file),
        ]

        if self.dry_run:
            print(f"[DRY RUN] tmux new-session ... {' '.join(runner_cmd)}")
        elif keep_alive:
            self._run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_name,
                    "-c",
                    str(self.workspace),
                ]
            )
            self._run(["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"])
            result = self._run(
                ["tmux", "respawn-pane", "-k", "-t", session_name, "--", *runner_cmd]
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to spawn session {session_name}: {result.stderr}")
        else:
            result = self._run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_name,
                    "-c",
                    str(self.workspace),
                    "--",
                    *runner_cmd,
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to spawn session {session_name}: {result.stderr}")

        self.active_sessions[session_name] = {
            "model": model.display_name,
            "role": role,
            "started": time.time(),
            "log_file": str(log_file),
            "exit_file": str(exit_file),
            "job_file": str(job_file),
        }
        return session_name

    def exit_code(self, session_name: str) -> int | None:
        """Return the agent exit code if the runner has finished, else None."""
        info = self.active_sessions.get(session_name, {})
        path = Path(info["exit_file"]) if info.get("exit_file") else None
        if path is None:
            fallback = self.runs_dir / session_name / "exit_code"
            path = fallback if fallback.exists() else None
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        try:
            return int(text)
        except ValueError:
            return 1

    def read_output(self, session_name: str, lines: int = 0) -> str:
        """Read agent output from the log and/or tmux pane; keep the longer copy."""
        chunks: list[str] = []
        info = self.active_sessions.get(session_name, {})
        log_path = Path(info["log_file"]) if info.get("log_file") else None
        if log_path is None:
            candidate = self.runs_dir / session_name / "output.log"
            log_path = candidate if candidate.exists() else None
        if log_path is not None and log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            chunks.append(_ANSI_RE.sub("", text))
        start = f"-{lines}" if lines > 0 else "-10000"
        result = self._run(
            ["tmux", "capture-pane", "-t", session_name, "-p", "-S", start],
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            chunks.append(_ANSI_RE.sub("", result.stdout))
        text = max(chunks, key=len, default="")
        if lines > 0:
            text = "\n".join(text.splitlines()[-lines:])
        return text

    def is_running(self, session_name: str) -> bool:
        """True while the agent runner is still the pane process."""
        if self.exit_code(session_name) is not None:
            return False
        if not self.session_exists(session_name):
            return False
        result = self._run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_dead}"],
            timeout=10,
        )
        if result.returncode != 0:
            return False
        flags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if flags and all(flag == "1" for flag in flags):
            return False
        return True

    def _log_path(self, session_name: str) -> Path | None:
        info = self.active_sessions.get(session_name, {})
        if info.get("log_file"):
            return Path(info["log_file"])
        candidate = self.runs_dir / session_name / "output.log"
        return candidate if candidate.exists() else None

    def wait_for_idle(
        self,
        session_name: str,
        timeout: int = 300,
        poll_interval: int = 5,
        idle_timeout: int = 120,
    ) -> bool:
        """Wait for the agent to finish. Returns True if idle, False if timed out.

        ``timeout`` is a wall-clock cap. ``idle_timeout`` fires earlier if the
        log file stops growing while the process is still alive (stuck TUI /
        permission prompt). Log growth resets the idle timer so a working
        agent is not killed mid-read. If the tmux session disappears without an
        exit file, this returns False (failure), not success.
        """
        start = time.time()
        last_change = start
        last_beat = start
        last_size = -1
        log_path = self._log_path(session_name)
        while time.time() - start < timeout:
            if self.exit_code(session_name) is not None:
                return True
            exists = self.session_exists(session_name)
            if not exists:
                # Runner died without writing exit_code (tmux crash / SIGHUP).
                return False
            if not self.is_running(session_name):
                return self.exit_code(session_name) is not None
            path = log_path or self._log_path(session_name)
            log_path = path
            size = path.stat().st_size if path is not None and path.exists() else 0
            now = time.time()
            if size != last_size:
                last_size = size
                last_change = now
            elif idle_timeout > 0 and (now - last_change) >= idle_timeout:
                return False
            if now - last_beat >= 15:
                print(
                    f"    still waiting ({session_name}: {int(now - start)}s, log={size}B)",
                    flush=True,
                )
                last_beat = now
            time.sleep(poll_interval)
        return False

    def collect(
        self,
        session_name: str,
        timeout: int,
        lines: int = 0,
        idle_timeout: int = 120,
    ) -> str:
        """Wait for an agent, then return its log. Raises on timeout/nonzero/empty."""
        ok = self.wait_for_idle(session_name, timeout=timeout, idle_timeout=idle_timeout)
        output = self.read_output(session_name, lines=lines)
        if not ok:
            self.kill(session_name)
            tail = (
                output[-1500:]
                if output.strip()
                else "(no output.log — runner may not have started)"
            )
            raise TimeoutError(
                f"Session {session_name} timed out after {timeout}s "
                f"(exit_code={'missing' if self.exit_code(session_name) is None else self.exit_code(session_name)})\n{tail}"
            )
        code = self.exit_code(session_name)
        if code is not None and code != 0:
            tail = output[-800:] if output else ""
            raise RuntimeError(f"Session {session_name} exited {code}\n{tail}")
        usable = "\n".join(
            ln for ln in output.splitlines() if not ln.startswith("[orchestra] exec")
        ).strip()
        if not usable:
            raise RuntimeError(f"Session {session_name} produced empty output")
        return output

    def kill(self, session_name: str) -> None:
        """Kill a tmux session."""
        self._run(["tmux", "kill-session", "-t", session_name], timeout=10)
        self.active_sessions.pop(session_name, None)

    def cleanup(self, prefix: str = "orchestra-") -> int:
        """Kill all orchestra-managed sessions. Returns count killed."""
        killed = 0
        for name in self.list_sessions():
            if name.startswith(prefix):
                self.kill(name)
                killed += 1
        return killed
