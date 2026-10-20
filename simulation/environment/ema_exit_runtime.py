"""Focused simulation execution service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.execution.broker import Position

from .market_view import BarView
from .mixin_base import RuntimeMixinBase


class EmaExitRuntimeMixin(RuntimeMixinBase):
    position: Position | None

    def _ema21_at(self, idx: int) -> float:
        if idx < 0 or idx >= self._n_bars:
            return float("nan")
        if self._bundle_arrays is not None:
            value = self._bundle_arrays["ema_21"][idx]
            return float(value) if not np.isnan(value) else float("nan")
        if "ema_21" not in self._features_col_to_idx:
            return float("nan")
        value = self._features_arr[idx, self._features_col_to_idx["ema_21"]]
        if pd.isna(value):
            return float("nan")
        return float(value)

    def _maybe_ema_exit(self, bar: BarView, idx: int, log_time: Any) -> None:
        """Close an EMA-21 trade after 1R of favorable travel, if the stop did not.

        A long exits when the bar close is below ``ema_21``. A short exits when
        the close is above it. The stop check runs before this.
        """
        pos = self.position
        if pos is None or pos.exit_mode != "ema_21":
            return
        risk = float(pos.stop_distance)
        if risk <= 1e-12:
            return
        if pos.direction == 1:
            favorable = float(bar.high) - float(pos.entry_price)
            crossed = float(bar.close) < self._ema21_at(idx)
        else:
            favorable = float(pos.entry_price) - float(bar.low)
            crossed = float(bar.close) > self._ema21_at(idx)
        pos.best_favorable = max(float(pos.best_favorable), favorable)
        if pos.best_favorable + 1e-12 < risk or not crossed or not np.isfinite(self._ema21_at(idx)):
            return
        execution = getattr(self, "_active_step_execution", None)
        if execution is not None:
            quote = execution["quote"]
            fill_quote = (quote.bid, quote.ask)
            fill_time = quote.tick_time
            execution_meta = dict(execution["metadata"])
        elif getattr(self, "tick_execution_required", False):
            order_time = pd.Timestamp(log_time) + pd.Timedelta(minutes=self._minutes_per_step)
            quote, execution_meta = self._resolve_randomized_tick_quote(order_time)
            fill_quote = (quote.bid, quote.ask)
            fill_time = quote.tick_time
        else:
            fill_quote = self._bar_quote(bar)
            fill_time = log_time + pd.Timedelta(minutes=self._minutes_per_step)
            execution_meta = {}
        pnl, fill_price = self.broker.close_position(self.account, pos, fill_quote)
        record = {
            "type": "close",
            "pnl": pnl,
            "price": fill_price,
            "reason": "ema_21_exit",
            "fill_quote_source": "tick" if execution_meta else "bar_close",
            "bar": idx,
            "time": log_time,
            "decision_time": log_time,
            "fill_time": fill_time,
            "equity": self.account.equity,
            **execution_meta,
        }
        self._stamp_exit(record, fill_price)
        self.trade_log.append(record)
        self.position = None
        self.sessions_with_trades.add(self._current_session_id())
        self._note_close(pnl)
