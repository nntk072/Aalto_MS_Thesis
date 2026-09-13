"""Pipeline orchestrator — executes the full orchestra workflow with data flow."""

from __future__ import annotations

import time
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import click

from .backup import create_snapshot
from .ci_gate import CI_CHECKS, ci_command_lines, gate_passed
from .cost import CostTracker
from .failures import FailureKind, classify_failure
from .handoff import build_handoff_bundle, render_handoff_prompt, save_handoff
from .models import Model, complexity_to_tier, get_error_signatures, load_models
from .parse import parse_complexity, parse_tier, parse_verdict, static_tier_fast_path
from .prompts import render_prompt
from .report import write_final_report
from .router import ModelRouter
from .routing import (
    apply_triage_bootstrap,
    effective_critic_count,
    effective_planner_count,
    effective_reviewer_count,
    merge_route,
    parse_route_from_text,
    record_phase_visit,
    resolve_next_phase,
    route_notes_for_role,
)
from .sessions import SessionManager
from .state import Phase, TaskState, generate_task_id

WORKSPACE = Path(__file__).resolve().parent.parent
_DIFF_CHARS = 50_000
_MAX_PHASE_STEPS = 40


def _cmd_tail(result: CompletedProcess[str], limit: int = 1000) -> str:
    text = (result.stdout or "") + (result.stderr or "")
    return text[-limit:]


class Pipeline:
    """Orchestrates the multi-model coding orchestra pipeline."""

    def __init__(
        self,
        workspace: Path = WORKSPACE,
        planner_count: int = 3,
        critic_count: int = 2,
        reviewer_count: int = 2,
        dry_run: bool = False,
        max_fixes: int = 3,
        keep_alive: bool = False,
    ):
        self.workspace = workspace
        self.planner_count = planner_count
        self.critic_count = critic_count
        self.reviewer_count = reviewer_count
        self.dry_run = dry_run
        self.max_fixes = max_fixes
        self.keep_alive = keep_alive
        self.escalate_mode = False
        self.state_dir = workspace / "orchestra" / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cost_tracker = CostTracker(state_dir=self.state_dir)
        self.router = ModelRouter(
            models=load_models(),
            state_dir=self.state_dir,
        )
        self.router.health.probe_tier0(self.router.models, check_version=False)
        self.sessions = SessionManager(workspace=workspace, dry_run=dry_run)
        self._state: TaskState | None = None
        self._tier_override: str | None = None

    @property
    def state(self) -> TaskState:
        """The active task state; set once the pipeline begins running."""
        if self._state is None:
            raise RuntimeError("Pipeline has not loaded a task state")
        return self._state

    @state.setter
    def state(self, value: TaskState) -> None:
        self._state = value

    def _record_model_call(
        self,
        model: Model,
        prompt: str,
        output: str,
        task_id: str | None = None,
        success: bool = True,
        role: str = "general",
    ) -> None:
        """Estimate tokens and record cost + quota for a model call."""
        model_name = model.display_name
        prompt_tokens = self.cost_tracker.estimate_tokens(prompt)
        completion_tokens = self.cost_tracker.estimate_tokens(output)
        self.cost_tracker.record_usage(model_name, prompt_tokens, completion_tokens, task_id)
        self.router.quota.record_usage(model_name, prompt_tokens + completion_tokens, model=model)
        if self._state is not None:
            self._state.record_performance(
                model_name, role, success, prompt_tokens + completion_tokens
            )

    def _task_tier(self) -> str:
        if self._tier_override:
            return self._tier_override
        tier = self.state.data.get("tier")
        if tier:
            return str(tier)
        complexity = self.state.data.get("complexity") or "medium"
        return complexity_to_tier(str(complexity))

    def _parse_complexity(self, triage_output: str) -> str:
        """Extract complexity from triage JSON output."""
        return parse_complexity(triage_output)

    def _parse_verdict(self, synthesis_output: str) -> str:
        """Extract verdict from review synthesis output."""
        return parse_verdict(synthesis_output)

    def run(
        self,
        task_description: str,
        resume_from: str | None = None,
        complexity: str | None = None,
        start_phase: str | None = None,
        tier: str | None = None,
    ) -> TaskState:
        """Run the full pipeline for a task.

        After each phase, advances to ``Phase.next()`` unless the phase set
        DONE, FAILED, or jumped (verification → FIXING). FIXING ↔ VERIFICATION
        therefore iterates until tests pass or the fix budget is exhausted.

        ``start_phase`` (name or step number) jumps to that phase on resume,
        including from DONE/FAILED. Stuck at planning → ``--from 2``.
        """
        if start_phase and not resume_from:
            raise ValueError("--from requires --resume <task-id>")
        self._tier_override = tier
        try:
            return self._run_phases(
                task_description,
                resume_from=resume_from,
                complexity=complexity,
                start_phase=start_phase,
                tier=tier,
            )
        finally:
            self._close_task_sessions()

    def _close_task_sessions(self) -> None:
        """Close leftover tmux panes for this task after fallback or finish."""
        if self.dry_run or self._state is None:
            return
        n = self.sessions.cleanup(task_id=self._state.task_id)
        if n:
            click.echo(f"  Closed {n} tmux session(s)")

    def _run_phases(
        self,
        task_description: str,
        resume_from: str | None,
        complexity: str | None,
        start_phase: str | None,
        tier: str | None,
    ) -> TaskState:
        """Execute phase loop; ``run`` always reaps tmux sessions after this."""
        if resume_from:
            self.state = TaskState.load(resume_from, self.state_dir)
            self.state.data["fix_loop_max"] = self.max_fixes
            if self.state.phase == Phase.PARKED:
                parked_at = self.state.data.get("parked_at", Phase.TRIAGE.value)
                self.state.set_phase(Phase(parked_at))
            target = self._resume_phase(start_phase)
            if target is None:
                click.echo(f"Task already in terminal phase: {self.state.phase.value}")
                return self.state
            self.state.set_phase(target)
            click.echo(f"Resuming task {self.state.task_id} from phase: {target.value}")
        else:
            task_id = generate_task_id()
            self.state = TaskState(
                task_id=task_id,
                task_description=task_description,
                state_dir=self.state_dir,
            )
            self.state.data["fix_loop_max"] = self.max_fixes
            self.state.data.setdefault("route", {})
            route = self.state.data["route"]
            route.setdefault("planner_count", self.planner_count)
            route.setdefault("critic_count", self.critic_count)
            route.setdefault("reviewer_count", self.reviewer_count)
            self.state.save()
            click.echo(f"Task ID: {task_id}")
            click.echo(f"Task: {task_description}")
            click.echo("=" * 60)
            if not self.dry_run:
                self._snapshot("00-start")
            if self.keep_alive:
                click.echo("  [KEEP-ALIVE] successful panes stay until the task finishes.")
                click.echo(f"  Watch with: tmux ls | grep {task_id}")
                click.echo("  Attach with: tmux attach -t <session>  (Ctrl-b d to detach)")

        if complexity:
            self.state.data["complexity"] = complexity
            self.state.data["complexity_override"] = True
            if not tier:
                self.state.data["tier"] = complexity_to_tier(complexity)
            self.state.save()
        if tier:
            self.state.data["tier"] = tier.upper()
            self.state.save()
        elif not resume_from and not self.state.data.get("tier"):
            fast = static_tier_fast_path(task_description)
            if fast:
                self.state.data["tier"] = fast
                self.state.data["complexity"] = "trivial"
                self.state.save()

        if self.state.phase in (Phase.DONE, Phase.FAILED, Phase.PARKED):
            click.echo(f"Task already in terminal phase: {self.state.phase.value}")
            return self.state
        if self.state.phase == Phase.PENDING:
            self.state.set_phase(Phase.TRIAGE)

        steps = 0
        while self.state.phase not in (Phase.DONE, Phase.FAILED, Phase.PARKED):
            steps += 1
            if steps > _MAX_PHASE_STEPS:
                raise RuntimeError(
                    f"Pipeline exceeded {_MAX_PHASE_STEPS} phase steps "
                    f"(stuck at {self.state.phase.value})"
                )
            phase = self.state.phase
            try:
                verification_passed = self._execute_phase(phase)
            except Exception as e:
                self.state.add_error(str(e))
                self.state.set_phase(Phase.FAILED)
                click.echo(f"\nERROR in {phase.value}: {e}", err=True)
                raise
            if self.state.phase in (Phase.DONE, Phase.FAILED, Phase.PARKED):
                break
            if self.state.phase != phase:
                continue
            if phase is Phase.REPORTING:
                self.state.set_phase(Phase.DONE)
                break
            self._advance_after_phase(phase, verification_passed=verification_passed)
            if self.state.phase in (Phase.DONE, Phase.FAILED, Phase.PARKED):
                break

        if self.state.is_parked():
            click.echo(
                f"\nTask PARKED: {self.state.data.get('park_reason', 'no models available')}"
            )
        return self.state

    def _resume_phase(self, start_phase: str | None) -> Phase | None:
        """Phase to execute next on resume, or None to leave a DONE task alone."""
        if start_phase:
            return Phase.parse(start_phase)
        current = self.state.phase
        if current is Phase.PENDING:
            return Phase.TRIAGE
        if current in Phase.runnable():
            return current
        if current is Phase.FAILED:
            failed_at = self.state.data.get("failed_at")
            if failed_at:
                return Phase.parse(str(failed_at))
            return Phase.VERIFICATION
        return None

    def _backup_slot(self, phase: Phase) -> str:
        numbers = {
            Phase.TRIAGE: 1,
            Phase.PLANNING: 2,
            Phase.CRITIQUE: 3,
            Phase.SYNTHESIS: 4,
            Phase.IMPLEMENTATION: 5,
            Phase.REVIEW: 6,
            Phase.REVIEW_SYNTHESIS: 7,
            Phase.FIXING: 8,
            Phase.VERIFICATION: 9,
            Phase.REPORTING: 10,
        }
        n = numbers.get(phase, 0)
        if phase is Phase.FIXING:
            loop = int(self.state.data.get("fix_loop_count") or 0)
            return f"{n:02d}-{phase.value}-{loop}"
        return f"{n:02d}-{phase.value}"

    def _snapshot(self, slot: str) -> None:
        if self.dry_run:
            return
        dest = self.state_dir / "backups" / self.state.task_id / slot
        meta = create_snapshot(self.workspace, dest)
        entry = {
            "slot": slot,
            "archive": meta["archive"],
            "files": meta["files"],
            "created_at": meta["created_at"],
        }
        self.state.data.setdefault("backups", []).append(entry)
        self.state.save()
        click.echo(f"  Backup [{slot}]: {meta['files']} files")

    def _effective_planner_count(self) -> int:
        return effective_planner_count(self.state.data, self.planner_count)

    def _effective_critic_count(self) -> int:
        return effective_critic_count(self.state.data, self.critic_count)

    def _effective_reviewer_count(self) -> int:
        return effective_reviewer_count(self.state.data, self.reviewer_count)

    def _phase_agent_output(self, phase: Phase) -> str:
        data = self.state.data
        if phase is Phase.TRIAGE:
            return str(data.get("triage_output") or "")
        if phase is Phase.PLANNING:
            outputs = data.get("planner_outputs") or []
            return str(outputs[-1]["output"]) if outputs else ""
        if phase is Phase.CRITIQUE:
            outputs = data.get("critic_outputs") or []
            return str(outputs[-1]["output"]) if outputs else ""
        if phase is Phase.SYNTHESIS:
            return str(data.get("synthesis_output") or "")
        if phase is Phase.IMPLEMENTATION:
            return str(data.get("implementer_output") or "")
        if phase is Phase.REVIEW:
            outputs = data.get("review_outputs") or []
            return str(outputs[-1]["output"]) if outputs else ""
        if phase is Phase.REVIEW_SYNTHESIS:
            return str(data.get("review_synthesis_output") or "")
        if phase is Phase.FIXING:
            outputs = data.get("fix_outputs") or []
            return str(outputs[-1]["output"]) if outputs else ""
        return ""

    def _advance_after_phase(
        self,
        completed: Phase,
        *,
        verification_passed: bool | None = None,
    ) -> None:
        output = self._phase_agent_output(completed)
        update = parse_route_from_text(output) if output else None
        merge_route(self.state.data, update)
        next_phase, reason = resolve_next_phase(
            completed,
            self.state.data,
            update,
            planner_count=self.planner_count,
            critic_count=self.critic_count,
            reviewer_count=self.reviewer_count,
            verification_passed=verification_passed,
            tier=self._task_tier(),
        )
        record_phase_visit(self.state.data, next_phase)
        self.state.save()
        click.echo(f"  Route: {completed.value} → {next_phase.value} ({reason})")
        self.state.set_phase(next_phase)

    def _execute_phase(self, phase: Phase) -> bool | None:
        """Execute a single pipeline phase. Returns verification pass/fail when applicable."""
        self.state.set_phase(phase)
        if phase is not Phase.REPORTING:
            self._snapshot(self._backup_slot(phase))
        if phase == Phase.TRIAGE:
            self._phase_triage()
            return None
        if phase == Phase.PLANNING:
            self._phase_planning()
            return None
        if phase == Phase.CRITIQUE:
            self._phase_critique()
            return None
        if phase == Phase.SYNTHESIS:
            self._phase_synthesis()
            return None
        if phase == Phase.IMPLEMENTATION:
            self._phase_implementation()
            return None
        if phase == Phase.REVIEW:
            self._phase_review()
            return None
        if phase == Phase.REVIEW_SYNTHESIS:
            self._phase_review_synthesis()
            return None
        if phase == Phase.FIXING:
            self._phase_fixing()
            return None
        if phase == Phase.VERIFICATION:
            return self._phase_verification()
        if phase == Phase.REPORTING:
            self._phase_report()
            return None
        return None

    def _echo_cmd(self, model: Model, role: str | None = None) -> None:
        effort = self.sessions.resolve_effort(model, self._task_tier(), role=role)
        cmd = self.sessions.build_command(model, "<prompt>", effort=effort)
        click.echo(f"    Command: {' '.join(cmd)}")

    def _invoke(
        self,
        model: Model,
        role: str,
        prompt: str,
        timeout: int,
        lines: int = 0,
        native_session_id: str | None = None,
    ) -> tuple[str, str]:
        """Spawn an agent, wait, and return ``(session, output)``. Raises on failure."""
        session = self.sessions.spawn(
            model,
            role,
            prompt,
            task_id=self.state.task_id,
            keep_alive=self.keep_alive,
            tier=self._task_tier(),
            native_session_id=native_session_id,
        )
        click.echo(f"    Session: {session}")
        try:
            output = self.sessions.collect(
                session, timeout=timeout, lines=lines, keep_alive=self.keep_alive
            )
        except (RuntimeError, TimeoutError):
            self.sessions.cleanup(task_id=self.state.task_id, role=role)
            raise
        return session, output

    def _invoke_required(
        self,
        model: Model,
        role: str,
        prompt: str,
        timeout: int,
        lines: int = 0,
        select_role: str | None = None,
    ) -> tuple[str, str]:
        """Run a role with ``model``, retrying with alternate models on failure.

        Records cost/performance under ``role``. ``select_role`` overrides the
        roster role used to build the fallback chain (e.g. a fixer picked from
        the implementer roster) and defaults to ``role``.
        """
        complexity = self.state.data.get("complexity") or "medium"
        task_tier = self._task_tier()
        candidates = [
            model,
            *self.router.fallback_chain(
                model,
                select_role or role,
                task_complexity=complexity,
                task_tier=task_tier,
                escalate=self.escalate_mode,
            ),
        ]
        if not candidates:
            self.state.park("No models available for role")
            raise RuntimeError("Task parked: no models available")

        last_error: BaseException | None = None
        use_handoff = False
        active_prompt = prompt
        signatures = get_error_signatures()

        prev_model = model
        for idx, candidate in enumerate(candidates):
            if use_handoff and idx > 0:
                bundle = build_handoff_bundle(
                    self.state.data,
                    self.workspace,
                    source_model=prev_model.display_name,
                    resume_hint=f"Continue {role} after fallback from previous model.",
                )
                save_handoff(bundle, self.state_dir)
                active_prompt = render_handoff_prompt(bundle)

            native_id: str | None = None
            if candidate.native_resume and candidate is model:
                raw = self.state.data.get("native_session_id")
                native_id = str(raw) if raw else None

            retries = 0
            max_retries = 2
            while retries <= max_retries:
                try:
                    session, output = self._invoke(
                        candidate,
                        role,
                        active_prompt,
                        timeout,
                        lines,
                        native_session_id=native_id,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    text = str(exc)
                    classified = classify_failure(
                        text,
                        exit_code=124 if isinstance(exc, TimeoutError) else None,
                        cli=candidate.cli,
                        signatures=signatures,
                    )
                    self._record_model_call(
                        candidate,
                        active_prompt,
                        text,
                        self.state.task_id,
                        success=False,
                        role=role,
                    )
                    self.router.health.record_failure(
                        candidate,
                        classified.kind.value,
                        classified.message,
                        throttled_until=classified.retry_after,
                        exhausted_until=classified.exhausted_until,
                    )
                    if (
                        classified.kind
                        in (
                            FailureKind.PROVIDER_ERROR,
                            FailureKind.TRANSIENT_RATE_LIMIT,
                            FailureKind.UNKNOWN,
                        )
                        and classified.retry_after is None
                        and retries < max_retries
                    ):
                        retries += 1
                        time.sleep(min(2**retries, 8))
                        continue
                    click.echo(
                        f"  WARNING: {candidate.display_name} failed "
                        f"({classified.kind.value}); trying fallback",
                        err=True,
                    )
                    self.sessions.cleanup(task_id=self.state.task_id, role=role)
                    last_error = exc
                    prev_model = candidate
                    use_handoff = True
                    break
                else:
                    self._record_model_call(
                        candidate,
                        active_prompt,
                        output,
                        self.state.task_id,
                        success=True,
                        role=role,
                    )
                    self.router.health.record_success(candidate)
                    if candidate.native_resume:
                        sid = self.sessions.extract_session_id(candidate.cli, output)
                        if sid:
                            self.state.data["native_session_id"] = sid
                            self.state.save()
                    return session, output
            else:
                continue

        self.state.park("All models failed or unavailable")
        if last_error is not None:
            raise last_error
        raise RuntimeError("Task parked: all models failed")

    def _select_models(self, role: str, count: int = 1) -> list[Model]:
        """Select models or park the task when none are available."""
        complexity = self.state.data.get("complexity") or "medium"
        models = self.router.select(
            role,
            count=count,
            task_complexity=complexity,
            task_tier=self._task_tier(),
            escalate=self.escalate_mode,
        )
        if not models:
            self.state.park(f"No {role} models available")
        return models

    def _git_diff(self) -> str:
        """Return unstaged+staged diff, truncated for prompts."""
        result = self.sessions._run(["git", "diff", "HEAD"], capture=True)
        if result.returncode != 0 or not (result.stdout or "").strip():
            result = self.sessions._run(["git", "diff"], capture=True)
        text = result.stdout if result.returncode == 0 else "Could not get diff"
        if len(text) > _DIFF_CHARS:
            return text[:_DIFF_CHARS] + "\n... [diff truncated]"
        return text

    def _phase_triage(self) -> None:
        """Phase 1: Triage the task."""
        click.echo("\n[1/10] TRIAGE")
        models = self._select_models("triage", count=1)
        if not models:
            return
        model = models[0]
        click.echo(f"  Model: {model.display_name}")

        prompt = render_prompt(
            "triage",
            task=self.state.task_description,
            route_notes=route_notes_for_role(self.state.data, "triage"),
        )
        if self.dry_run:
            self._echo_cmd(model, "triage")
            if not self.state.data.get("complexity_override") and not self.state.data.get(
                "complexity"
            ):
                self.state.data["complexity"] = "medium"
                self.state.save()
            tier = self._task_tier()
            apply_triage_bootstrap(self.state.data, tier=tier, had_explicit_goto=False)
            return

        _session, output = self._invoke_required(model, "triage", prompt, timeout=600, lines=0)
        if self.state.data.get("complexity_override"):
            detected_complexity = self.state.data["complexity"]
        else:
            detected_complexity = self._parse_complexity(output)
        detected_tier = parse_tier(output)
        self.state.set_triage(output, model.display_name, detected_complexity, tier=detected_tier)
        route_update = parse_route_from_text(output)
        apply_triage_bootstrap(
            self.state.data,
            tier=detected_tier,
            had_explicit_goto=route_update is not None and route_update.goto is not None,
        )
        click.echo(f"  Complexity: {detected_complexity}  Tier: {detected_tier}")

    def _phase_planning(self) -> None:
        """Phase 2: Parallel planning with multiple models."""
        planner_n = self._effective_planner_count()
        click.echo(f"\n[2/10] PARALLEL PLANNING ({planner_n} planners)")

        if planner_n <= 0:
            click.echo("  Skipping planning (planner_count=0).")
            return

        models = self._select_models("planner", count=planner_n)
        if not models:
            return
        if len(models) < planner_n:
            click.echo(f"  WARNING: requested {planner_n} planners, got {len(models)}")

        self.state.data["planner_outputs"] = []
        self.state.data["planner_session_map"] = {}
        self.state.save()

        spawned: list[tuple[str, Model, str]] = []
        for i, model in enumerate(models):
            click.echo(f"  Planner {i + 1}: {model.display_name}")
            prompt = render_prompt(
                "planner",
                task=self.state.task_description,
                triage=self.state.data.get("triage_output", ""),
                workspace=str(self.workspace),
                route_notes=route_notes_for_role(self.state.data, "planner"),
            )
            if self.dry_run:
                self._echo_cmd(model, "planner")
                continue
            session = self.sessions.spawn(
                model,
                "planner",
                prompt,
                task_id=self.state.task_id,
                keep_alive=self.keep_alive,
                tier=self._task_tier(),
            )
            spawned.append((session, model, prompt))
            click.echo(f"    Session: {session}")

        if self.dry_run:
            return

        click.echo("  Waiting for planners...")
        successes = 0
        for session_name, model, prompt in spawned:
            click.echo(f"  … {model.display_name}")
            try:
                output = self.sessions.collect(
                    session_name, timeout=600, lines=0, keep_alive=self.keep_alive
                )
            except (RuntimeError, TimeoutError) as exc:
                click.echo(f"  WARNING: {model.display_name} failed: {exc}", err=True)
                self._record_model_call(
                    model, prompt, str(exc), self.state.task_id, False, "planner"
                )
                continue
            self.state.add_planner_output(model.display_name, output, session_name)
            self._record_model_call(model, prompt, output, self.state.task_id, True, "planner")
            click.echo(f"  {model.display_name}: {len(output)} chars received")
            successes += 1
        if successes == 0:
            raise RuntimeError("All planners failed")

    def _phase_critique(self) -> None:
        """Phase 3: Critique each plan independently."""
        critic_n = self._effective_critic_count()
        click.echo(f"\n[3/10] CRITIQUE ({critic_n} critics)")

        if critic_n <= 0:
            click.echo("  Skipping critique (critic_count=0).")
            return

        models = self._select_models("critic", count=critic_n)
        if not models:
            return
        plans_text = self.state.get_planner_outputs_text()
        self.state.data["critic_outputs"] = []
        self.state.save()

        spawned: list[tuple[str, Model, str]] = []
        for i, model in enumerate(models):
            click.echo(f"  Critic {i + 1}: {model.display_name}")
            prompt = render_prompt(
                "critic",
                plan=plans_text,
                other_plans=plans_text,
                triage=self.state.data.get("triage_output", ""),
                route_notes=route_notes_for_role(self.state.data, "critic"),
            )
            if self.dry_run:
                self._echo_cmd(model, "critic")
                continue
            session = self.sessions.spawn(
                model,
                "critic",
                prompt,
                task_id=self.state.task_id,
                keep_alive=self.keep_alive,
                tier=self._task_tier(),
            )
            spawned.append((session, model, prompt))
            click.echo(f"    Session: {session}")

        if self.dry_run:
            return

        successes = 0
        for session_name, model, prompt in spawned:
            try:
                output = self.sessions.collect(
                    session_name, timeout=300, lines=0, keep_alive=self.keep_alive
                )
            except (RuntimeError, TimeoutError) as exc:
                click.echo(f"  WARNING: {model.display_name} failed: {exc}", err=True)
                self._record_model_call(
                    model, prompt, str(exc), self.state.task_id, False, "critic"
                )
                continue
            self.state.add_critic_output(model.display_name, output, session_name)
            self._record_model_call(model, prompt, output, self.state.task_id, True, "critic")
            successes += 1
        if successes == 0:
            raise RuntimeError("All critics failed")

    def _phase_synthesis(self) -> None:
        """Phase 4: Synthesize plans and critiques into final plan."""
        click.echo("\n[4/10] SYNTHESIS")

        models = self._select_models("synthesizer", count=1)
        if not models:
            return
        model = models[0]
        click.echo(f"  Model: {model.display_name}")

        prompt = render_prompt(
            "synthesizer",
            plans=self.state.get_planner_outputs_text(),
            critiques=self.state.get_critic_outputs_text(),
            triage=self.state.data.get("triage_output", ""),
            workspace=str(self.workspace),
            route_notes=route_notes_for_role(self.state.data, "synthesizer"),
        )
        if self.dry_run:
            self._echo_cmd(model, "synthesizer")
            return

        _session, output = self._invoke_required(model, "synthesizer", prompt, timeout=300, lines=0)
        self.state.set_synthesis(output, model.display_name)
        click.echo(f"  Final plan: {len(output)} chars")

    def _phase_implementation(self) -> None:
        """Phase 5: Implement the final plan."""
        click.echo("\n[5/10] IMPLEMENTATION")

        models = self._select_models("implementer", count=1)
        if not models:
            return
        model = models[0]
        click.echo(f"  Model: {model.display_name}")

        prompt = render_prompt(
            "implementer",
            plan=self.state.data.get("final_plan", ""),
            triage=self.state.data.get("triage_output", ""),
            workspace=str(self.workspace),
            route_notes=route_notes_for_role(self.state.data, "implementer"),
        )
        if self.dry_run:
            self._echo_cmd(model, "implementer")
            self.state.data["implementation_succeeded"] = True
            self.state.save()
            return

        _session, output = self._invoke_required(model, "implementer", prompt, timeout=900, lines=0)
        self.state.set_implementation(output, model.display_name)

    def _phase_review(self) -> None:
        """Phase 6: Independent review of implementation."""
        reviewer_n = self._effective_reviewer_count()
        click.echo(f"\n[6/10] REVIEW ({reviewer_n} reviewers)")

        if reviewer_n <= 0:
            click.echo("  Skipping review (reviewer_count=0).")
            return

        models = self._select_models("reviewer", count=reviewer_n)
        if not models:
            return
        if len(models) < reviewer_n:
            click.echo(f"  WARNING: requested {reviewer_n} reviewers, got {len(models)}")

        diff_text = self._git_diff()
        test_results = "Not yet run (verification runs full CI gate: format, lint, mypy, pytest)."
        if not self.dry_run:
            click.echo("  Skipping pre-review pytest; verification is the test gate.")

        self.state.data["review_outputs"] = []
        self.state.save()

        spawned: list[tuple[str, Model, str]] = []
        for i, model in enumerate(models):
            click.echo(f"  Reviewer {i + 1}: {model.display_name}")
            prompt = render_prompt(
                "reviewer",
                plan=self.state.data.get("final_plan", ""),
                diff=diff_text,
                test_results=test_results,
                route_notes=route_notes_for_role(self.state.data, "reviewer"),
            )
            if self.dry_run:
                self._echo_cmd(model, "reviewer")
                continue
            session = self.sessions.spawn(
                model,
                "reviewer",
                prompt,
                task_id=self.state.task_id,
                keep_alive=self.keep_alive,
                tier=self._task_tier(),
            )
            spawned.append((session, model, prompt))
            click.echo(f"    Session: {session}")

        if self.dry_run:
            return

        for session_name, model, prompt in spawned:
            try:
                output = self.sessions.collect(
                    session_name, timeout=300, lines=0, keep_alive=self.keep_alive
                )
            except (RuntimeError, TimeoutError) as exc:
                click.echo(f"  WARNING: {model.display_name} failed: {exc}", err=True)
                self._record_model_call(
                    model, prompt, str(exc), self.state.task_id, False, "reviewer"
                )
                continue
            self.state.add_reviewer_output(model.display_name, output, session_name)
            self._record_model_call(model, prompt, output, self.state.task_id, True, "reviewer")

    def _phase_review_synthesis(self) -> None:
        """Phase 7: Synthesize multiple reviews into actionable report."""
        click.echo("\n[7/10] REVIEW SYNTHESIS")

        models = self._select_models("review_synthesizer", count=1)
        if not models:
            return
        model = models[0]
        click.echo(f"  Model: {model.display_name}")

        prompt = render_prompt(
            "review_synthesizer",
            task=self.state.task_description,
            plan=self.state.data.get("final_plan", ""),
            diff=self._git_diff(),
            test_results=str(self.state.data.get("verification_result", "Not yet run")),
            reviews=self.state.get_review_outputs_text(),
            route_notes=route_notes_for_role(self.state.data, "review_synthesizer"),
        )
        if self.dry_run:
            self._echo_cmd(model, "review_synthesizer")
            return

        _session, output = self._invoke_required(
            model, "review_synthesizer", prompt, timeout=1800, lines=0
        )
        verdict = self._parse_verdict(output)
        self.state.set_review_synthesis(output, model.display_name, verdict)
        click.echo(f"  Verdict: {verdict}")

    def _verification_gate_failed(self) -> bool:
        result = self.state.data.get("verification_result")
        if not result:
            return False
        return not gate_passed(result)

    def _phase_fixing(self) -> None:
        """Phase 8: Apply targeted fixes based on review synthesis."""
        fix_count = self.state.data.get("fix_loop_count", 0)
        max_fixes = self.state.data.get("fix_loop_max", 3)
        verdict = self.state.data.get("review_verdict")
        synthesis = self.state.data.get("review_synthesis_output")

        click.echo(f"\n[8/10] FIX LOOP (iteration {fix_count + 1}/{max_fixes})")

        tests_failed = self._verification_gate_failed()
        if not tests_failed and (not synthesis or not verdict):
            click.echo("  No review synthesis available. Skipping fixes.")
            return

        if not tests_failed and verdict == "pass":
            click.echo("  Review passed; skipping fixes until verification.")
            return

        if fix_count >= max_fixes:
            click.echo(f"  Max fix iterations ({max_fixes}) reached. Giving up.")
            return

        models = self._select_models("implementer", count=1)
        if not models:
            return
        model = models[0]
        click.echo(f"  Model: {model.display_name}")

        synthesis = self.state.data.get("review_synthesis_output", "") or ""
        verify = self.state.data.get("verification_result") or {}
        ci_failures = self._format_ci_failure_output(verify)
        prompt = f"""You are fixing specific issues found during code review or CI verification.

## Original Task
{self.state.task_description}

## Review Synthesis
{synthesis or "(none)"}

## CI verification failures (if already ran)
{ci_failures or "(not yet run)"}

## Required CI gate (same as GitHub CI — not nightly)
{ci_command_lines()}

## Instructions
1. Fix ONLY the failing checks / critical issues listed above
2. Do NOT refactor unrelated code
3. Do NOT add new features
4. Make minimal, targeted changes
5. Re-run only the failing check(s); phase 9 will re-run the full CI gate
6. Report what you fixed and command results

Focus on correctness and safety. Do not change code style unless it's part of the fix.

## Routing (phase 8 — fixing)

| May goto | 1, 2, 5, 6, 9, 10 |
| Must-not goto | 3, 4, 7 |

Silent default: → **9**. Optional JSON tail:

```json
{{"route": {{"goto": 9}}}}
```"""

        if self.dry_run:
            self._echo_cmd(model, "fixer")
            return

        _session, output = self._invoke_required(
            model, "fixer", prompt, timeout=600, lines=0, select_role="implementer"
        )
        self.state.add_fix_output(output, model.display_name)
        click.echo(f"  Fix applied: {len(output)} chars")

    def _phase_verification(self) -> bool:
        """Phase 9: Deterministic verification with fix loop.

        Returns True when the CI gate passes, False when fixes are needed.
        """
        click.echo("\n[9/10] VERIFICATION")

        route = self.state.data.get("route") or {}
        verify_mode = route.get("verify_mode", "full")

        if self.dry_run:
            click.echo("  Would run CI gate (.github/workflows/ci.yml, not nightly):")
            for check in CI_CHECKS:
                timeout = f" timeout={check.timeout}s" if check.timeout else ""
                click.echo(f"    {' '.join(check.argv)}{timeout}")
            if verify_mode == "skip":
                click.echo("  verify_mode=skip (dry-run treats as pass)")
            return True

        if verify_mode == "skip":
            click.echo("  verify_mode=skip — skipping CI gate")
            skipped = {
                check.name: {"success": True, "output": "skipped (verify_mode=skip)"}
                for check in CI_CHECKS
            }
            self.state.set_verification(skipped)
            return True

        results = self._run_verification()
        self.state.set_verification(results)

        fix_count = self.state.data.get("fix_loop_count", 0)
        max_fixes = self.state.data.get("fix_loop_max", 3)

        if gate_passed(results):
            click.echo("  CI gate: PASS")
            return True

        click.echo(self._format_ci_failure_output(results)[-1500:])
        if fix_count >= max_fixes:
            click.echo(
                f"\n[FAIL] Max fix iterations ({max_fixes}) reached. Manual intervention needed."
            )
            self.state.add_error(f"CI gate still failing after {max_fixes} fix attempts")
            self._phase_report()
            self.state.set_phase(Phase.FAILED)
            return True

        click.echo(f"\n[FAIL] CI gate failed. Routing to fix loop ({fix_count + 1}/{max_fixes})...")
        return False

    def _format_ci_failure_output(self, results: dict[str, Any]) -> str:
        parts: list[str] = []
        for check in CI_CHECKS:
            entry = results.get(check.name) or {}
            if entry.get("success"):
                continue
            label = check.name.upper()
            output = str(entry.get("output") or "(no output)")
            parts.append(f"### {label}\n{output[-800:]}")
        return "\n\n".join(parts) if parts else ""

    def _run_verification(self) -> dict[str, Any]:
        """Run the same checks as GitHub CI (format, lint, mypy, full pytest)."""
        results: dict[str, Any] = {}
        for check in CI_CHECKS:
            argv = check.argv
            timeout = check.timeout
            timeout_note = f" (timeout={timeout}s)" if timeout else ""
            click.echo(f"  Running: {' '.join(argv)}{timeout_note}")
            heartbeat = 15.0 if check.name == "tests" else 0.0
            result = self.sessions._run(
                argv,
                capture=True,
                timeout=timeout,
                heartbeat=heartbeat,
            )
            ok = result.returncode == 0
            results[check.name] = {
                "success": ok,
                "output": _cmd_tail(result, limit=2000 if check.name == "tests" else 500),
            }
            click.echo(f"  {check.name}: {'PASS' if ok else 'FAIL'}")
        return results

    def _phase_report(self) -> None:
        """Phase 10: Write a markdown report covering steps 1-9."""
        click.echo("\n[10/10] FINAL REPORT")
        path = write_final_report(self.state)
        click.echo(f"  Wrote {path}")
        click.echo(f"\n[PASS] Pipeline complete. Report: {path}")
