"""Focused simulation transition service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase
from .timebase import BAR_TIME_TZ


class StepAccountingRuntimeMixin(RuntimeMixinBase):
    def _record_step_accounting(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        step_trade_log_start: int,
        done: bool,
        truncated: bool,
    ) -> tuple[bool, Any, float, float, dict[str, float]]:
        payout_terminal = self._handle_repeated_payout(bar_time)
        if payout_terminal:
            done = True

        valuation_time = bar_time + pd.Timedelta(minutes=self._minutes_per_step)
        for event in self.trade_log[step_trade_log_start:]:
            event_time = event.get("fill_time")
            if event_time is None and event.get("time_precision") == "bar_interval":
                event_time = event.get("event_time_end")
            if event_time is not None:
                event_timestamp = pd.Timestamp(event_time)
                if event_timestamp.tzinfo is None:
                    event_timestamp = event_timestamp.tz_localize("UTC")
                event_timestamp = event_timestamp.tz_convert(BAR_TIME_TZ)
                if event_timestamp > valuation_time:
                    valuation_time = event_timestamp
        self.equity_curve.append(self.account.equity)
        self.equity_times.append(valuation_time)
        pnl_step = (
            self.account.equity + float(self._payout_withdrawn_this_step) - self.equity_curve[-2]
        )
        self.pnl_history.append(pnl_step)

        reward = self._calculate_reward(pnl_step, bar, feat_row, done, truncated)
        reward_parts = {
            key: float(value) for key, value in getattr(self.reward_fn, "last_parts", {}).items()
        }
        if self.peak_trailing_dd_limit > 0.0 and self._peak_dd_fired:
            # Penalty each bar the training peak-DD ceiling is still hit.
            reward -= 1.0
            reward_parts["peak_dd"] = float(reward_parts.get("peak_dd", 0.0)) - 1.0
        return done, valuation_time, pnl_step, float(reward), reward_parts


__all__ = ["StepAccountingRuntimeMixin"]
