"""Focused orchestra phase implementation."""

from __future__ import annotations

import click

from .ci_gate import ci_command_lines
from .pipeline_base import PipelineMixinBase


class PhaseFixingMixin(PipelineMixinBase):
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
