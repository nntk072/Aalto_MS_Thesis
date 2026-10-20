from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from trading.execution.broker import Position
from trading.positions.multi_tp import reorder_remaining_targets

from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase
from .position_constants import _PRICE_TOL, _SL_MODE_RANK, _TP_MODE_RANK


class PositionRuntimeMixin(RuntimeMixinBase):
    position: Position | None

    def _apply_breakeven(self, pos: Position, favorable: float) -> None:
        """Move the stop to breakeven once the trade has earned its risk.

        Only ever tightens toward entry: never widens a stop and never rewrites
        a stop that already sits at or beyond breakeven. Disabled when
        ``breakeven_trigger_r <= 0``.
        """
        if self._breakeven_trigger_r <= 0.0 or pos.breakeven_done:
            return
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return
        risk_pts = abs(pos.entry_price - float(sl0))
        if risk_pts <= 0.0:
            return
        if favorable < self._breakeven_trigger_r * risk_pts:
            return
        buffer = self._breakeven_buffer_pts
        if pos.direction == 1:
            new_sl = pos.entry_price + buffer
            if pos.sl_price is None or new_sl > float(pos.sl_price):
                pos.sl_price = new_sl
                pos.breakeven_done = True
        else:
            new_sl = pos.entry_price - buffer
            if pos.sl_price is None or new_sl < float(pos.sl_price):
                pos.sl_price = new_sl
                pos.breakeven_done = True

    def _apply_trailing_stop(
        self,
        pos: Position,
        feat_row: FeatureRow | pd.Series | BarView | None,
        *,
        bar_idx: int | None = None,
        event_time: Any = None,
    ) -> None:
        """Tighten only against a confirmed swing on the protective side.

        Swing feature values must already be available at the decision bar;
        the current tradeable bar close constrains stop geometry when known.
        """
        if pos.sl_mode != "trailing":
            return
        buf = self.sl_buffer_pts
        previous = pos.sl_price
        reference_name: str | None = None
        reference_price: float | None = None
        if pos.direction == 1:
            swing = self._finite(feat_row, "last_swing_low")
            if swing is not None:
                new_sl = float(swing) - buf
                current = self._finite(feat_row, "close")
                if (current is None or new_sl < current) and (
                    pos.sl_price is None or new_sl > float(pos.sl_price)
                ):
                    pos.sl_price = new_sl
                    reference_name = "last_swing_low"
                    reference_price = float(swing)
        elif pos.direction == -1:
            swing = self._finite(feat_row, "last_swing_high")
            if swing is not None:
                new_sl = float(swing) + buf
                current = self._finite(feat_row, "close")
                if (current is None or new_sl > current) and (
                    pos.sl_price is None or new_sl < float(pos.sl_price)
                ):
                    pos.sl_price = new_sl
                    reference_name = "last_swing_high"
                    reference_price = float(swing)
        if reference_name is not None and hasattr(self, "trade_log"):
            effective_time = None
            if bar_idx is not None and bar_idx + 1 < self._n_bars:
                effective_time = self._bar_time(bar_idx + 1)
            self.trade_log.append(
                {
                    "type": "sl_update",
                    "position_id": pos.position_id,
                    "time": event_time,
                    "effective_time": effective_time,
                    "direction": pos.direction,
                    "previous_price": previous,
                    "price": pos.sl_price,
                    "reference": reference_name,
                    "reference_price": reference_price,
                    "buffer_points": buf,
                    "reason": "confirmed_swing",
                }
            )

    def _reorder_tp_levels_safe(self, pos: Position) -> None:
        """Enforce remaining target ordering via the trading domain invariant."""
        cost_model = getattr(self, "cost_model", None)
        point_size = float(getattr(cost_model, "point_size", _PRICE_TOL))
        reorder_remaining_targets(pos, point_size=point_size, price_tolerance=_PRICE_TOL)

    def _log_tp_update(
        self,
        pos: Position,
        *,
        slot: int,
        previous: float,
        price: float,
        reference: str,
        reference_price: float | None,
        reason: str,
        event_time: Any,
        bar_idx: int | None,
    ) -> None:
        """Record the TP mutation and the first bar on which it can execute."""
        if not hasattr(self, "trade_log"):
            return
        effective_time = None
        if bar_idx is not None and bar_idx + 1 < self._n_bars:
            effective_time = self._bar_time(bar_idx + 1)
        setattr(pos, f"tp{slot}_ref", reference)
        self.trade_log.append(
            {
                "type": "tp_update",
                "position_id": pos.position_id,
                "time": event_time,
                "effective_time": effective_time,
                "direction": pos.direction,
                "tp_slot": slot,
                "previous_price": previous,
                "price": price,
                "mode": pos.tp_mode,
                "reference": reference,
                "reference_price": reference_price,
                "reason": reason,
            }
        )

    def _apply_tp_breakeven(
        self,
        pos: Position,
        favorable: float,
        *,
        current_price: float | None = None,
        event_time: Any = None,
        bar_idx: int | None = None,
    ) -> None:
        """Pull the nearest unhit TP to entry ± alpha*initial-R, at most once.

        This legacy ``breakeven`` TP mode does not modify the stop. A pullback
        level already crossed by the current market is left unchanged to avoid
        creating a retroactive limit fill.
        """
        if not self.allow_agent_tp_mode or pos.tp_breakeven_done:
            return
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return
        risk_pts = abs(pos.entry_price - float(sl0))
        if risk_pts <= 0:
            return
        alpha = self.tp_breakeven_alpha  # config field, default 0.5
        if pos.direction == 1:
            new_tp = pos.entry_price + alpha * risk_pts
        else:
            new_tp = pos.entry_price - alpha * risk_pts
        # Apply only to nearest active TP (tp1 if active, else tp2, else tp3)
        for slot, attr in enumerate(("tp1_price", "tp2_price", "tp3_price"), start=1):
            if pos.tp_hit_mask & (1 << (slot - 1)):
                continue
            current = getattr(pos, attr)
            if current is None:
                continue
            pos.tp_breakeven_done = True
            if pos.direction == 1:
                tightens = new_tp < float(current)
                still_ahead = current_price is None or new_tp > current_price
            else:
                tightens = new_tp > float(current)
                still_ahead = current_price is None or new_tp < current_price
            if tightens and still_ahead:
                setattr(pos, attr, new_tp)
                self._log_tp_update(
                    pos,
                    slot=slot,
                    previous=float(current),
                    price=float(new_tp),
                    reference="legacy_tp_pullback_initial_r_v1",
                    reference_price=float(sl0),
                    reason="favorable_excursion_reached_alpha_initial_r",
                    event_time=event_time,
                    bar_idx=bar_idx,
                )
            break  # only the nearest remaining, unhit slot can be pulled

    def _apply_tp_trailing(
        self,
        pos: Position,
        feat_row: Any,
        *,
        event_time: Any = None,
        bar_idx: int | None = None,
    ) -> None:
        """Extend only the nearest unhit TP to a newer, executable swing."""
        if not self.allow_agent_tp_mode or pos.tp_mode != "trailing":
            return
        reference: str
        if pos.direction == 1:
            swing = self._finite(feat_row, "last_swing_high")
            reference = "last_swing_high"
        elif pos.direction == -1:
            swing = self._finite(feat_row, "last_swing_low")
            reference = "last_swing_low"
        else:
            return
        if swing is None:
            return

        close = self._finite(feat_row, "close")
        cost_model = getattr(self, "cost_model", None)
        increment = float(getattr(cost_model, "point_size", _PRICE_TOL))
        if not np.isfinite(increment) or increment <= 0.0:
            increment = _PRICE_TOL
        scaled = float(swing) / increment
        candidate = (
            float(np.ceil(scaled - 1e-12) * increment)
            if pos.direction == 1
            else float(np.floor(scaled + 1e-12) * increment)
        )
        if close is not None and pos.direction * (candidate - close) <= 0:
            return

        # Target slot identities and allocated quantities are stable. One
        # confirmed swing updates only the nearest remaining slot; later legs
        # keep their original distinct references and ordering.
        for slot in (1, 2, 3):
            if pos.tp_hit_mask & (1 << (slot - 1)):
                continue
            attr = f"tp{slot}_price"
            old = getattr(pos, attr)
            if old is None:
                continue
            if pos.direction * (candidate - float(old)) > 0:
                # Do not let this slot overtake a later active slot. A target
                # update that would invalidate another slot would silently
                # discard its allocated quantity, so wait for a distinct
                # reference that preserves the original slot ordering.
                tol = max(_PRICE_TOL, increment / 2)
                for later_slot in range(slot + 1, 4):
                    if pos.tp_hit_mask & (1 << (later_slot - 1)):
                        continue
                    later = getattr(pos, f"tp{later_slot}_price")
                    if later is None:
                        continue
                    overtakes = (
                        candidate >= float(later) - tol
                        if pos.direction == 1
                        else candidate <= float(later) + tol
                    )
                    if overtakes:
                        return
                setattr(pos, attr, candidate)
                self._log_tp_update(
                    pos,
                    slot=slot,
                    previous=float(old),
                    price=candidate,
                    reference=reference,
                    reference_price=float(swing),
                    reason="new_confirmed_favorable_swing",
                    event_time=event_time,
                    bar_idx=bar_idx,
                )
            break
        self._reorder_tp_levels_safe(pos)

    def _apply_position_guards(
        self,
        bar: BarView,
        bar_time: Any,
        fill_bid: float,
        fill_ask: float,
        *,
        eod: bool,
        feat_row: FeatureRow | pd.Series | None = None,
        bar_idx: int | None = None,
        fill_time: Any | None = None,
    ) -> None:
        """Age, stale-progress, breakeven, and max-loss guards for a position."""
        pos = self.position
        if pos is None:
            return
        if pos.entry_timestamp is not None:
            age_h = (
                pd.Timestamp(cast(Any, bar_time)) - pd.Timestamp(cast(Any, pos.entry_timestamp))
            ).total_seconds() / 3600.0
            if age_h > self._eod_max_age_hours:
                self._force_close_guard(
                    fill_bid, fill_ask, "max_age", bar_time, fill_time=fill_time
                )
                return
        if pos.direction == 1:
            favorable = float(bar.high) - pos.entry_price
            adverse = pos.entry_price - float(bar.low)
        else:
            favorable = pos.entry_price - float(bar.low)
            adverse = float(bar.high) - pos.entry_price
        pos.mfe = max(pos.mfe, favorable)
        pos.mae = max(pos.mae, adverse)
        pos.best_favorable = max(pos.best_favorable, favorable)
        # Advance the SL mode state machine (monotonic: fixed → breakeven →
        # trailing).  An agent override (outside the defer band) locks the mode;
        # only the default state machine advances here, and only forward.
        if not pos.sl_mode_overridden:
            default_mode = self._default_sl_mode(pos)
            if _SL_MODE_RANK.get(default_mode, 0) > _SL_MODE_RANK.get(pos.sl_mode, 0):
                pos.sl_mode = default_mode
        if pos.sl_mode != "fixed":
            self._apply_breakeven(pos, pos.best_favorable)
        self._apply_trailing_stop(
            pos,
            feat_row if feat_row is not None else bar,
            bar_idx=bar_idx,
            event_time=bar_time,
        )
        # Advance the TP mode state machine (monotonic: fixed → breakeven → trailing).
        if self.allow_agent_tp_mode and not pos.tp_mode_overridden:
            default_tp = self._default_tp_mode(pos)
            if _TP_MODE_RANK.get(default_tp, 0) > _TP_MODE_RANK.get(pos.tp_mode, 0):
                pos.tp_mode = default_tp
        if self.allow_agent_tp_mode and pos.tp_mode != "fixed":
            self._apply_tp_breakeven(
                pos,
                pos.best_favorable,
                current_price=float(bar.close),
                event_time=bar_time,
                bar_idx=bar_idx,
            )
        if self.allow_agent_tp_mode:
            self._apply_tp_trailing(
                pos,
                feat_row if feat_row is not None else bar,
                event_time=bar_time,
                bar_idx=bar_idx,
            )
        thresh = self._eod_progress_atr * float(pos.entry_atr or 0.0)
        if thresh > 0.0 and (favorable - pos.last_progress_favorable) >= thresh:
            pos.last_progress_favorable = favorable
            pos.stale_counter = 0
        else:
            pos.stale_counter += 1
        notional_adverse = adverse * pos.size * self.contract_size
        if notional_adverse >= self._eod_adverse_cut_usd(pos):
            self._force_close_guard(
                fill_bid, fill_ask, "eod_max_loss", bar_time, fill_time=fill_time
            )
            return
        if eod and pos.stale_counter >= self._eod_stale_bars:
            self._force_close_guard(fill_bid, fill_ask, "stale", bar_time, fill_time=fill_time)

    def _force_close_guard(
        self,
        fill_bid: float,
        fill_ask: float,
        reason: str,
        bar_time: Any,
        *,
        fill_time: Any | None = None,
    ) -> None:
        if self.position is None:
            return
        execution_meta: dict[str, Any] = {}
        if getattr(self, "tick_execution_required", False):
            order_time = (
                pd.Timestamp(fill_time)
                if fill_time is not None
                else pd.Timestamp(bar_time) + pd.Timedelta(minutes=self._minutes_per_step)
            )
            quote, execution_meta = self._resolve_randomized_tick_quote(order_time)
            fill_bid, fill_ask = quote.bid, quote.ask
            fill_time = quote.tick_time
        pnl, fill_price = self.broker.close_position(
            self.account, self.position, (fill_bid, fill_ask)
        )
        record = {
            "type": "eod_close",
            "pnl": pnl,
            "price": fill_price,
            "reason": reason,
            "bar": self.step_idx,
            "time": bar_time,
            "decision_time": bar_time,
            "fill_time": fill_time if fill_time is not None else bar_time,
            "equity": self.account.equity,
            **execution_meta,
        }
        self._stamp_exit(record, fill_price)
        self.trade_log.append(record)
        self.position = None
        self.sessions_with_trades.add(self._current_session_id())
        self._note_close(pnl)
