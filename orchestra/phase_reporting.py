"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .pipeline_base import PipelineMixinBase
from .report import write_final_report


class PhaseReportingMixin(PipelineMixinBase):
    def _phase_report(self) -> None:
        """Phase 10: Write a markdown report covering steps 1-9."""
        click.echo("\n[10/10] FINAL REPORT")
        path = write_final_report(self.state)
        click.echo(f"  Wrote {path}")
        click.echo(f"\n[PASS] Pipeline complete. Report: {path}")
