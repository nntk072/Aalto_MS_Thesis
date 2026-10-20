"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .market_view import FeatureRow
from .mixin_base import RuntimeMixinBase


class EntryFunnelRuntimeMixin(RuntimeMixinBase):
    def _update_entry_funnel(
        self,
        feat_row: FeatureRow | pd.Series,
        entry_machine: Any,
        entry_diag_before: dict[str, int],
    ) -> dict[str, int]:
        entry_funnel: dict[str, int] = {"bars_seen": 1}
        if entry_machine is not None:
            prior_state = entry_machine.state
            prior_candidates = entry_machine.candidate_created
            evidence = self.entry_evidence_context(feat_row)
            entry_machine.observe(evidence)
            if entry_machine.state == "IN_POSITION" and self.position is None:
                entry_machine.mark_closed()
            entry_funnel.update(
                {
                    "strategy_context_available": int(evidence.direction in (-1, 1)),
                    "entry_evidence": int(evidence.evidence_detected),
                    "candidate_created": int(entry_machine.candidate_created > prior_candidates),
                    "candidate_armed": int(
                        entry_machine.state == "ARMED" and prior_state != "ARMED"
                    ),
                    "trigger_requested": int(self._trigger_requested_this_step),
                    "hold_low_intensity": int(
                        self._entry_diag["hold_low_intensity"]
                        > entry_diag_before.get("hold_low_intensity", 0)
                    ),
                    "hold_no_context": int(
                        self._entry_diag["hold_no_context"]
                        > entry_diag_before.get("hold_no_context", 0)
                    ),
                    "hold_risk_floor": int(
                        self._entry_diag["hold_risk_floor"]
                        > entry_diag_before.get("hold_risk_floor", 0)
                    ),
                    "entry_rejected": int(bool(self._entry_rejection_reason)),
                    "position_opened": int(self._opened_this_step),
                }
            )
        return entry_funnel


__all__ = ["EntryFunnelRuntimeMixin"]
