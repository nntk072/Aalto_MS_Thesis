"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .models import Model
from .pipeline_base import PipelineMixinBase
from .prompts import render_prompt
from .routing import (
    route_notes_for_role,
)


class PhasePlanningMixin(PipelineMixinBase):
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
