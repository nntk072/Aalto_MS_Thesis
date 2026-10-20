"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from decisioning.observation_state import ENTRY_STATE_DIM, entry_state_one_hot

from simulation.environment.diagnostics import note_debug_step
from simulation.environment.episode_equity import assess_train_equity

from .mixin_base import RuntimeMixinBase


class StepEndRuntimeMixin(RuntimeMixinBase):
    _pending_entry_order: dict[str, Any] | None
    _opened_this_step: bool
    episode_step_count: int
    _ep_reward_sum: float

    def _resolve_step_end(
        self,
        action: int | float | np.ndarray[Any, Any],
        discrete_action: int,
        risk_frac: float,
        entry_machine: Any,
        valuation_time: Any,
        close_reason: str,
        reward: float,
        reward_parts: dict[str, float],
        obs: dict[str, np.ndarray[Any, Any]],
        info: dict[str, Any],
        done: bool,
        truncated: bool,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._advance_ny_step()
        if entry_machine is not None and entry_machine.state == "IN_POSITION":
            if self.position is None:
                entry_machine.mark_closed()
                info["entry_state"] = entry_machine.state
                if self.entry_state_observation:
                    entry_start = self.observation_schema.width - ENTRY_STATE_DIM
                    obs["account"][entry_start:] = np.asarray(
                        entry_state_one_hot(entry_machine.state), dtype=np.float32
                    )
        self.episode_step_count += 1

        # Past last NY bar → year survived (truncated, not terminated).
        if self._ny_pos >= len(self._ny_indices) and not done:
            truncated = True

        # Apply max_episode_steps truncation last so it overrides
        # data-end signals consistently. Do not convert a year-fail
        # (terminated) into a pure truncation when the cap also fires.
        if self.max_episode_steps is not None and self.episode_step_count >= self.max_episode_steps:
            truncated = True

        if (done or truncated) and self._pending_entry_order is not None:
            pending = self._pending_entry_order
            self._pending_entry_order = None
            self.trade_log.append(
                {
                    "type": "order_cancel",
                    "reason": "episode_ended_before_scheduled_fill",
                    "decision_time": pending["decision_time"],
                    "order_submission_time": pending["order_submission_time"],
                    "time": valuation_time,
                    "fill_time": None,
                    "bar": self.step_idx,
                }
            )

        if self._debug_trace and self._debug_window is not None:
            note_debug_step(
                self._debug_window,
                reward=float(reward),
                reward_parts=reward_parts,
                opened=bool(self._opened_this_step),
                entered=discrete_action in (1, -1),
                risk_frac=float(risk_frac),
                close_reason=close_reason,
                action=action,
                position=self.position is not None,
                position_direction=0 if self.position is None else int(self.position.direction),
            )

        self._ep_reward_sum += float(reward)
        if done or truncated:
            n_trades = len([t for t in self.trade_log if t.get("type") == "open"])
            gate = assess_train_equity(
                pd.Series(self.equity_curve, dtype=float),
                initial=float(self._ep_start_equity),
                breached=bool(self.breach_log),
            )
            completed_data_split = self._ny_pos >= len(self._ny_indices)
            if done:
                if self.repeated_payout_policy and self.cycle_successes > 0 and not self.breach_log:
                    episode_end_reason = "payout_success"
                else:
                    episode_end_reason = "risk_breach" if self.breach_log else "terminated"
            elif completed_data_split:
                episode_end_reason = "data_end"
            elif self.max_episode_steps is not None and (
                self.episode_step_count >= self.max_episode_steps
            ):
                episode_end_reason = "max_episode_steps"
            else:
                episode_end_reason = "truncated"
            info["episode_equity"] = {
                "end_equity": float(self.account.equity),
                "start_equity": float(self._ep_start_equity),
                "reward_sum": float(self._ep_reward_sum),
                "n_trades": int(n_trades),
                "episode_steps": int(self.episode_step_count),
                "completed_data_split": bool(completed_data_split),
                "episode_end_reason": episode_end_reason,
                "breach_reason": str(self.breach_log[-1]) if self.breach_log else "",
                "slope": float(gate["slope"]),
                "max_peak_trailing_dd": float(gate["max_peak_trailing_dd"]),
            }

        return obs, float(reward), done, truncated, info


__all__ = ["StepEndRuntimeMixin"]
