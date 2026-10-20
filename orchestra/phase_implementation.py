"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .pipeline_base import PipelineMixinBase
from .prompts import render_prompt
from .routing import (
    route_notes_for_role,
)


class PhaseImplementationMixin(PipelineMixinBase):
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
