"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .models import Model
from .pipeline_base import PipelineMixinBase
from .prompts import render_prompt
from .routing import (
    route_notes_for_role,
)


class PhaseCritiqueMixin(PipelineMixinBase):
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
