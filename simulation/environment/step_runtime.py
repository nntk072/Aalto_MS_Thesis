"""Composition root for the environment step transition services."""

from __future__ import annotations

from .action_intent_runtime import ActionIntentRuntimeMixin
from .entry_funnel_runtime import EntryFunnelRuntimeMixin
from .step_accounting_runtime import StepAccountingRuntimeMixin
from .step_action_execution import StepActionExecutionMixin
from .step_constraints_runtime import StepConstraintsRuntimeMixin
from .step_end_runtime import StepEndRuntimeMixin
from .step_info_runtime import StepInfoRuntimeMixin


class StepRuntimeMixin(
    ActionIntentRuntimeMixin,
    StepConstraintsRuntimeMixin,
    StepActionExecutionMixin,
    StepAccountingRuntimeMixin,
    EntryFunnelRuntimeMixin,
    StepInfoRuntimeMixin,
    StepEndRuntimeMixin,
):
    """Compose action, constraint, execution, accounting and episode stages."""

    pass


__all__ = ["StepRuntimeMixin"]
