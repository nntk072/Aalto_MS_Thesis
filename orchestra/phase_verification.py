"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .ci_gate import CI_CHECKS, gate_passed
from .pipeline_base import PipelineMixinBase
from .state import Phase


class PhaseVerificationMixin(PipelineMixinBase):
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
