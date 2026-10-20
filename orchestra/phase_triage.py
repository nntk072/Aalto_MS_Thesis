"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .parse import parse_tier
from .pipeline_base import PipelineMixinBase
from .prompts import render_prompt
from .routing import (
    apply_triage_bootstrap,
    parse_route_from_text,
    route_notes_for_role,
)


class PhaseTriageMixin(PipelineMixinBase):
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
