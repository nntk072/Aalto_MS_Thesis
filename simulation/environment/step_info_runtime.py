"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .entry_diagnostics import _previous_session_audit
from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase


class StepInfoRuntimeMixin(RuntimeMixinBase):
    _decision_action: dict[str, float]
    _previous_constraint_ratios: dict[str, float]

    def _build_step_info(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        valuation_time: Any,
        reward_parts: dict[str, float],
        entry_funnel: dict[str, int],
        entry_machine: Any,
        entry_state_for_action: str | None,
        done: bool,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], str, dict[str, Any]]:
        obs = self._get_observation()
        close_reason = ""
        if self.trade_log:
            last_trade = self.trade_log[-1]
            if last_trade.get("bar") == self.step_idx and last_trade.get("type") not in (
                "open",
                "reject",
                "sl_update",
            ):
                close_reason = str(last_trade.get("reason") or "")
        if self._entry_rejection_reason and not self._opened_this_step:
            self.trade_log.append(
                {
                    "type": "reject",
                    "entry_rejection_reason": self._entry_rejection_reason,
                    "bar": self.step_idx,
                    "time": bar_time,
                    **(
                        _previous_session_audit(self.strategy, feat_row, float(bar.close))
                        if feat_row is not None
                        else {}
                    ),
                    **({"entry_state": entry_machine.state} if entry_machine is not None else {}),
                    **self._decision_action,
                }
            )
        info: dict[str, Any] = {
            "equity": self.account.equity,
            "valuation_time": valuation_time,
            "position": self.position is not None,
            "position_direction": 0 if self.position is None else int(self.position.direction),
            "close_reason": close_reason,
            "reward_parts": reward_parts,
            "entry_rejection_reason": self._entry_rejection_reason,
            **{f"entry_{k}": v for k, v in self._entry_diag.items()},
            # Derived, never stored, so it cannot drift from the buckets.
            "entry_rejected_total": sum(
                v for k, v in self._entry_diag.items() if k.startswith("rejected_")
            ),
            "entry_funnel": entry_funnel,
        }
        if self.repeated_payout_policy:
            info.update(
                {
                    "trading_policy_mode": self.trading_policy.mode,
                    "daily_loss": float(self.account.canonical_daily_loss()),
                    "daily_floor": float(
                        self.account.daily_floor(self.guardrails.daily_loss_limit)
                    ),
                    "remaining_daily_capacity": float(
                        self.account.remaining_daily_capacity(self.guardrails.daily_loss_limit)
                    ),
                    "remaining_account_capacity": float(
                        self.account.remaining_account_capacity(self.trading_policy.account_floor)
                    ),
                    "cumulative_payouts": float(self.cumulative_payouts),
                    "total_economic_value": float(self.account.equity + self.cumulative_payouts),
                    "cycle_count": int(self.cycle_count),
                    "cycle_successes": int(self.cycle_successes),
                    "cycle_failures": int(self.cycle_failures),
                    "requested_risk_usd": float(self._last_requested_risk_usd),
                    "approved_risk_usd": float(self._last_approved_risk_usd),
                    "execution_reserve_usd": float(self._last_execution_reserve_usd),
                }
            )
        constraint_ratios = {
            "daily": (
                self.account.canonical_daily_loss() / self.guardrails.daily_limit_for(self.account)
                if self.guardrails.daily_limit_for(self.account) > 0.0
                else 0.0
            ),
            "max_loss": (
                self.account.loss_from_initial() / self.guardrails.max_loss_limit_for(self.account)
                if self.guardrails.max_loss_limit_for(self.account) > 0.0
                else 0.0
            ),
            "trailing": (
                self.account.trailing_drawdown_pct() / self.guardrails.trailing_dd_limit
                if self.guardrails.trailing_dd_limit > 0.0
                else 0.0
            ),
        }
        breach_reason = str(self.breach_log[-1]) if done and self.breach_log else ""
        daily_breach = breach_reason == "daily_loss"
        max_loss_breach = breach_reason == "max_drawdown"
        trailing_breach = breach_reason == "trailing_dd"
        info["constraints"] = {
            "daily_ratio": float(constraint_ratios["daily"]),
            "max_loss_ratio": float(constraint_ratios["max_loss"]),
            "trailing_ratio": float(constraint_ratios["trailing"]),
            "daily_cost": max(
                0.0, constraint_ratios["daily"] - self._previous_constraint_ratios["daily"]
            )
            + float(daily_breach),
            "max_loss_cost": max(
                0.0,
                constraint_ratios["max_loss"] - self._previous_constraint_ratios["max_loss"],
            )
            + float(max_loss_breach),
            "trailing_cost": max(
                0.0,
                constraint_ratios["trailing"] - self._previous_constraint_ratios["trailing"],
            )
            + float(trailing_breach),
            "daily_breach": daily_breach,
            "max_loss_breach": max_loss_breach,
            "trailing_breach": trailing_breach,
            "breach_reason": breach_reason,
            "previous_daily_ratio": self._previous_constraint_ratios["daily"],
            "previous_max_loss_ratio": self._previous_constraint_ratios["max_loss"],
            "previous_trailing_ratio": self._previous_constraint_ratios["trailing"],
        }
        self._previous_constraint_ratios = constraint_ratios
        if entry_machine is not None:
            info["entry_state"] = entry_machine.state
            info["entry_state_action"] = entry_state_for_action
        prior_open = self._prior_open_payload()
        if prior_open is not None:
            info["prior_open"] = prior_open
        risk_u = self._decision_action.get("risk_u")
        if isinstance(risk_u, (int, float)) and np.isfinite(risk_u):
            info["risk_u"] = float(risk_u)
        return obs, close_reason, info


__all__ = ["StepInfoRuntimeMixin"]
