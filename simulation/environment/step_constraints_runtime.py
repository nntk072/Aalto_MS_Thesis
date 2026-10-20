"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase


class StepConstraintsRuntimeMixin(RuntimeMixinBase):
    _decision_action: dict[str, float]
    _pending_entry_order: dict[str, Any] | None

    def _enforce_step_constraints(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any,
        pending_due: bool,
        entry_fill_bid: float,
        entry_fill_ask: float,
        entry_fill_time: Any,
        market_fill_idx: int,
        entry_machine: Any,
    ) -> tuple[bool, bool, bool, bool]:
        # Check guardrails
        reason, session_blocked, done, truncated = self._check_guardrails(
            bar_time, session_id, fill_bid, fill_ask, fill_time=fill_time
        )

        position_has_completed_bar = True
        if self.position is not None and self.position.entry_timestamp is not None:
            bar_end = bar_time + pd.Timedelta(minutes=self._minutes_per_step)
            position_has_completed_bar = (
                pd.Timestamp(cast(Any, self.position.entry_timestamp)) < bar_end
            )
        if not reason and not session_blocked and position_has_completed_bar:
            # A bar-close entry is effective at this bar's end. Its earlier
            # high/low cannot trigger a stop/target or position guard.
            self._check_sl_tp(bar, fill_bid, fill_ask)
            self._apply_position_guards(
                bar,
                bar_time,
                fill_bid,
                fill_ask,
                eod=False,
                feat_row=feat_row,
                bar_idx=self.step_idx,
                fill_time=fill_time,
            )

        self._peak_dd_fired = False
        peak_dd_block = self._apply_peak_trailing_dd(
            fill_bid, fill_ask, bar_time, fill_time=fill_time
        )

        # A delayed entry becomes active only when its scheduled quote is the
        # market bar being processed. It cannot be marked or stopped earlier.
        if pending_due:
            pending = self._pending_entry_order
            self._pending_entry_order = None
            pending_soft_block = self.guardrails.check_soft_daily(
                self.account
            ) or self.guardrails.check_soft_trailing_dd(self.account)
            if (
                pending is not None
                and not done
                and not session_blocked
                and not peak_dd_block
                and not pending_soft_block
            ):
                current_decision = self._decision_action
                selected_names = (
                    "_selected_sl_anchor",
                    "_selected_tp_fraction",
                    "_selected_exit_mode",
                    "_selected_sl_mode",
                    "_sl_mode_explicit",
                    "_selected_tp_mode",
                    "_tp_mode_explicit",
                    "_selected_tp_selections",
                    "_selected_tp_z",
                )
                current_selected = {name: getattr(self, name) for name in selected_names}
                self._decision_action = dict(pending["decision_action"])
                for name, value in pending["selected"].items():
                    setattr(self, name, value)
                self._try_enter_position(
                    int(pending["direction"]),
                    float(pending["risk_frac"]),
                    float(pending["rr_ratio"]),
                    bar,
                    pending["feat_row"],
                    pending["decision_time"],
                    session_id,
                    entry_fill_bid,
                    entry_fill_ask,
                    entry_fill_time,
                    entry_candidate=pending["entry_candidate"],
                    fill_bar_idx=market_fill_idx,
                    order_submission_time=pending["order_submission_time"],
                    decision_bar_idx=int(pending["decision_bar_idx"]),
                    signal_bar_idx=int(pending["signal_bar_idx"]),
                )
                self._decision_action = current_decision
                for name, value in current_selected.items():
                    setattr(self, name, value)
                if entry_machine is not None and self._opened_this_step:
                    entry_machine.mark_entered()
            elif pending is not None:
                self.trade_log.append(
                    {
                        "type": "order_cancel",
                        "reason": "execution_blocked_before_scheduled_fill",
                        "decision_time": pending["decision_time"],
                        "order_submission_time": pending["order_submission_time"],
                        "time": fill_time,
                        "fill_time": None,
                        "bar": self.step_idx,
                    }
                )

        return done, truncated, session_blocked, peak_dd_block


__all__ = ["StepConstraintsRuntimeMixin"]
