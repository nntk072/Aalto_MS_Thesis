"""Focused simulation execution service."""

from __future__ import annotations

from typing import Any

import numpy as np

from .mixin_base import RuntimeMixinBase


class TradeAccountingRuntimeMixin(RuntimeMixinBase):
    _last_realized_close_pnl: float | None
    _last_realized_close_r: float | None
    _pending_realized_r: float | None

    def _note_close(self, pnl: float, realized_r: float | None = None) -> None:
        """Record realized close PnL/R and start the post-close entry cooldown."""
        # The exit fill has already been counted by _stamp_exit. Retain the
        # aggregate when several TP legs execute inside the same env.step().
        if self._last_realized_close_pnl is None:
            self._last_realized_close_pnl = float(pnl)
        if realized_r is None:
            realized_r = self._pending_realized_r
        if self._last_realized_close_r is None and realized_r is not None:
            self._last_realized_close_r = float(realized_r)
        self._pending_realized_r = None
        if self.strategy_actions and self.entry_cooldown_bars > 0:
            self._cooldown_until_bar = self.step_idx + self.entry_cooldown_bars

    def _stamp_exit(self, record: dict[str, Any], fill_price: float) -> None:
        """Stamp one unique fill and accumulate its net realised PnL/R once."""
        pos = self.position
        if pos is None or "fill_id" in record:
            return
        active_execution = getattr(self, "_active_step_execution", None)
        if active_execution is not None:
            for key, value in active_execution["metadata"].items():
                record.setdefault(key, value)
        # The environment uses observed ticks only for triggered stop/target
        # exits. Other fills use the bar open, bar close, or an OHLC-derived
        # level. Preserve that distinction in every raw fill record so reports
        # can audit execution-source parity by chronological split.
        record.setdefault(
            "fill_quote_source",
            "bar_range_level" if record.get("time_precision") == "bar_interval" else "bar_close",
        )
        record.setdefault(
            "order_submission_time",
            record.get("decision_time", pos.entry_timestamp),
        )
        if record.get("time_precision") == "bar_interval":
            record.setdefault("fill_time", None)
        else:
            record.setdefault("fill_time", record.get("time"))
        # close_position has already reduced pos.size. Derive each fill's size
        # from immutable entry quantity and previously stamped exit quantities.
        initial = float(pos.initial_size or pos.tp_initial_size or pos.size)
        close_size = max(0.0, initial - float(pos.filled_size) - float(pos.size))
        if close_size <= 1e-9 and "size" in record:
            close_size = float(record["size"])
        pos.filled_size += close_size
        pos.fill_count += 1
        record["position_id"] = pos.position_id
        record["fill_id"] = f"{pos.position_id}:f{pos.fill_count}"
        record["size"] = close_size
        record["position_complete"] = bool(pos.size <= 1e-9)
        # Broker pnl already includes exit commission. Entry commission was
        # debited on open; allocate it exactly once, pro rata across fills.
        entry_cost = float(self.broker.cost_model.total_cost(close_size))
        net_pnl = float(record.get("pnl", 0.0)) - entry_cost
        record["entry_commission_allocated"] = entry_cost
        record["net_pnl"] = net_pnl
        pos.position_net_pnl += net_pnl
        risk_usd = pos.planned_risk_usd
        if risk_usd is None or risk_usd <= 1e-12:
            dist = float(pos.stop_distance)
            if dist <= 1e-12 and pos.sl_initial_price is not None:
                dist = abs(float(pos.entry_price) - float(pos.sl_initial_price))
            risk_usd = dist * initial * float(getattr(self.broker, "contract_size", 1.0))
        realized = net_pnl / risk_usd if risk_usd > 1e-12 else float("nan")
        planned = pos.planned_rr
        record["sl_ref"] = pos.sl_ref
        record["tp_ref"] = pos.tp_ref
        record["planned_rr"] = float("nan") if planned is None else float(planned)
        record["realized_r"] = realized
        previous_pnl = self._last_realized_close_pnl
        self._last_realized_close_pnl = (0.0 if previous_pnl is None else previous_pnl) + net_pnl
        if np.isfinite(realized):
            previous_r = self._last_realized_close_r
            self._last_realized_close_r = (0.0 if previous_r is None else previous_r) + realized
            self._pending_realized_r = self._last_realized_close_r
        record["exit_mode"] = pos.exit_mode
        record["sl_buffer"] = pos.sl_buffer
        record["manipulation"] = pos.manipulation
        record["n_sl"] = pos.n_sl
        record["n_tp"] = pos.n_tp
        if record["position_complete"]:
            record["position_net_pnl"] = pos.position_net_pnl
            record["position_realized_r"] = (
                pos.position_net_pnl / risk_usd if risk_usd > 1e-12 else float("nan")
            )
            for row in reversed(self.trade_log):
                if row.get("type") == "open" and row.get("position_id") == pos.position_id:
                    row["realized_r"] = record["position_realized_r"]
                    row["position_net_pnl"] = pos.position_net_pnl
                    row["position_complete"] = True
                    break
