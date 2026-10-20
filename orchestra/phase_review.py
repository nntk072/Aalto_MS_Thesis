"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .models import Model
from .pipeline_base import PipelineMixinBase
from .prompts import render_prompt
from .routing import (
    route_notes_for_role,
)


class PhaseReviewMixin(PipelineMixinBase):
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
