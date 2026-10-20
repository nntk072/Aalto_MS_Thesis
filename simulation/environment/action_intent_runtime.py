"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .entry_state import EntryCandidate
from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase


class ActionIntentRuntimeMixin(RuntimeMixinBase):
    _decision_action: dict[str, float]

    def _prepare_action_intent(
        self,
        action: int | float | np.ndarray[Any, Any],
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
        pending_order: dict[str, Any] | None,
    ) -> tuple[Any, str | None, int, float, float, str, EntryCandidate | None]:
        # Decode action (discrete or continuous)
        self._note_session_open(session_id, feat_row)
        self._arm_po3_confirm(session_id)
        entry_machine = self.entry_state_machine
        entry_state_for_action = entry_machine.state if entry_machine is not None else None
        discrete_action, risk_frac, rr_ratio, tp_mode = self._decode_action(action, feat_row)
        self._selected_tp_mode = tp_mode
        entry_candidate: EntryCandidate | None = None
        if entry_machine is not None:
            action_requested = self._entry_action_requested(discrete_action)
            if entry_state_for_action in ("FLAT", "CANDIDATE"):
                if action_requested:
                    self._reject("entry_state_not_armed")
                discrete_action = 0
                risk_frac = 0.0
            elif entry_state_for_action == "IN_POSITION":
                discrete_action = 0
                risk_frac = 0.0
            elif action_requested:
                entry_candidate = entry_machine.request_trigger()
                if entry_candidate is not None:
                    self._trigger_requested_this_step = True
                    discrete_action = entry_candidate.direction
                    self._decision_action["action_direction"] = float(entry_candidate.direction)
                else:
                    discrete_action = 0
                    risk_frac = 0.0
            else:
                discrete_action = 0
                risk_frac = 0.0

        if pending_order is not None:
            if discrete_action != 0:
                self._reject("entry_pending")
            discrete_action = 0
            risk_frac = 0.0
            entry_candidate = None

        # Check entry gate for new positions
        # Suppress new entries on the last bar of a session (overnight block
        # just closed a position or would have if there was one).
        entering_new_session = self.block_overnight and self._is_ny_session_boundary()
        if entering_new_session and discrete_action != 0:
            # Counting only when an entry was actually requested keeps this a
            # true funnel: a plain hold is not a rejection.
            discrete_action = 0
            if self.strategy_actions:
                self._reject("overnight_boundary")
        # Session entry cap + post-close cooldown (trader / strategy actions).
        if self.strategy_actions and discrete_action != 0:
            if self._in_open_blackout(session_id):
                discrete_action = 0
                risk_frac = 0.0
                self._reject("open_blackout")
            elif (
                self.max_entries_per_session > 0
                and self._session_entry_counts.get(session_id, 0) >= self.max_entries_per_session
            ):
                discrete_action = 0
                risk_frac = 0.0
                self._reject("session_cap")
            elif self.entry_cooldown_bars > 0 and self.step_idx < self._cooldown_until_bar:
                discrete_action = 0
                risk_frac = 0.0
                self._reject("entry_cooldown")

        if self.strategy_actions:
            discrete_action, risk_frac = self._lock_unconfirmed_side(
                discrete_action, feat_row, risk_frac
            )

        if self.block_overnight and discrete_action != 0 and fill_idx < self._n_bars:
            fill_ny_pos = int(
                np.searchsorted(np.asarray(self._ny_indices, dtype=np.int64), int(fill_idx))
            )
            has_ny_fill = (
                fill_ny_pos < len(self._ny_indices)
                and int(self._ny_indices[fill_ny_pos]) == fill_idx
            )
            has_same_session_followup = False
            if has_ny_fill and fill_ny_pos + 1 < len(self._ny_indices):
                next_ny_idx = int(self._ny_indices[fill_ny_pos + 1])
                fill_session = (
                    int(self._session_ids_arr[fill_idx])
                    if self._session_ids_arr is not None
                    else session_id
                )
                next_session = (
                    int(self._session_ids_arr[next_ny_idx])
                    if self._session_ids_arr is not None
                    else fill_session
                )
                has_same_session_followup = fill_session == next_session
            if not has_ny_fill or not has_same_session_followup:
                discrete_action = 0
                risk_frac = 0.0
                if self.strategy_actions:
                    self._reject("overnight_boundary")

        # Check entry gate for new positions
        if discrete_action != 0:  # Only check for long/short entries
            if entry_machine is not None and entry_state_for_action == "ARMED":
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
                if bool(getattr(self.strategy, "previous_session_pd_gate", False)):
                    pd_eligible = getattr(self.strategy, "previous_session_pd_eligible")
                    gate_ok = gate_ok and bool(pd_eligible(feat_row, discrete_action))
            elif self.strategy_actions:
                gate_ok = self.strategy.validate_entry(
                    direction=discrete_action,
                    row=feat_row,
                )
            else:
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
            if not gate_ok:
                # Gate not satisfied, force to hold
                rejected_direction = discrete_action
                discrete_action = 0
                risk_frac = 0.0
                if self.strategy_actions:
                    reason_getter = getattr(self.strategy, "entry_rejection_reason", None)
                    reject_reason = (
                        str(reason_getter(direction=rejected_direction, row=feat_row))
                        if callable(reason_getter)
                        else "entry_gate"
                    )
                    self._reject(reject_reason)

        return (
            entry_machine,
            entry_state_for_action,
            discrete_action,
            risk_frac,
            rr_ratio,
            tp_mode,
            entry_candidate,
        )


__all__ = ["ActionIntentRuntimeMixin"]
