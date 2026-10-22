"""Focused simulation execution service."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from market_data.loading.ticks import (
    TickExecutionUnavailable,
)
from trading.execution.broker import Position

from .market_view import BarView
from .mixin_base import RuntimeMixinBase


class ProtectiveLevelRuntimeMixin(RuntimeMixinBase):
    position: Position | None

    def _check_sl_tp(
        self,
        bar: BarView,
        fill_bid: float,
        fill_ask: float,
        bar_idx: int | None = None,
    ) -> None:
        """Check whether the open position hit its SL or TP on this bar.

        With a tick book, the first bid (long) or ask (short) beyond the
        level is the fill. ``fill_delay_ms`` is 0, so a later tick is not
        used: a stop cannot bounce back to a better price. That crossing
        tick is already through the level, so a loss can exceed $100 and a
        target can pay more than $100. With no crossing tick, the fill stays
        at the level. Slippage, when configured, moves that level fill only
        against the position. If both levels sit inside one bar and there is
        no tick, the stop wins.
        """
        if self.position is None:
            return
        record: dict[str, Any]
        idx = self.step_idx if bar_idx is None else bar_idx
        log_time = (
            self._bar_time(idx)
            if 0 <= idx < len(self._bar_times)
            else self._bar_time(self.step_idx)
        )
        # Touch detection uses the bar range. The next quote must not be the fill.
        _ = (fill_bid, fill_ask)

        # Multi-level positions require a chronological tick replay. A single
        # SL/TP lookup loses the slot identities and can liquidate the entire
        # position at TP1 (or ignore TP1 when only tp1_price is populated).
        has_tick_slots = any(
            getattr(self.position, f"tp{slot}_price") is not None for slot in (1, 2, 3)
        )
        if has_tick_slots and self._tickbook is not None:
            if self._check_multi_tp_ticks(log_time, idx):
                return

        tick_hit = None if has_tick_slots else self._tick_exit_quote(log_time)
        if tick_hit is not None:
            kind, quote, fill_time, execution_meta = tick_hit
            pnl, fill_price = self._close_at_quote(self.position, quote, tick_fill=True)
            record = {
                "type": "stop_close" if kind == "sl" else "tp_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_sl" if kind == "sl" else "structure_tp",
                "fill_quote_source": "tick",
                "bar": idx,
                "time": fill_time,
                "equity": self.account.equity,
                **execution_meta,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
            return

        if (
            getattr(self, "tick_execution_required", False)
            or getattr(self, "randomized_fill_latency_ms", 0.0) > 0
        ):
            position = self.position
            assert position is not None
            stop_touched = position.sl_price is not None and (
                (position.direction == 1 and float(bar.low) <= position.sl_price)
                or (position.direction == -1 and float(bar.high) >= position.sl_price)
            )
            target_levels = [getattr(position, f"tp{slot}_price") for slot in (1, 2, 3)]
            if not any(target is not None for target in target_levels):
                target_levels = [position.tp_price]
            target_touched = any(
                target is not None
                and (
                    (position.direction == 1 and float(bar.high) >= target)
                    or (position.direction == -1 and float(bar.low) <= target)
                )
                for target in target_levels
            )
            if stop_touched or target_touched:
                bar_end = log_time + pd.Timedelta(minutes=getattr(self, "_minutes_per_step", 1))
                if self._tickbook is None or not self._tickbook.has_ticks_between(
                    log_time, bar_end
                ):
                    raise TickExecutionUnavailable(
                        "OHLC indicates a protective level was reached, but the bar "
                        "has no tick observations; candle fallback is disabled"
                    )
                # Intrabar OHLC extrema are not executable bid/ask evidence.
                # When quotes exist but none crosses the position's executable
                # level, keep the position open instead of inventing a fill.
            self._maybe_ema_exit(bar, idx, log_time)
            return

        sl_hit = False
        if self.position.sl_price is not None:
            if self.position.direction == 1 and float(bar.low) <= self.position.sl_price:
                sl_hit = True
            elif self.position.direction == -1 and float(bar.high) >= self.position.sl_price:
                sl_hit = True

        if sl_hit:
            assert self.position.sl_price is not None
            pnl, fill_price = self.broker.close_position(
                self.account,
                self.position,
                self._stop_fill_quote(
                    bar, int(self.position.direction), float(self.position.sl_price)
                ),
            )
            record = {
                "type": "stop_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_sl",
                "bar": idx,
                "time": log_time,
                "fill_time": None,
                "event_time_start": log_time,
                "event_time_end": log_time
                + pd.Timedelta(minutes=getattr(self, "_minutes_per_step", 1)),
                "time_precision": "bar_interval",
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
            return

        has_multi_levels = self.position is not None and (
            self.position.tp1_price is not None
            or self.position.tp2_price is not None
            or self.position.tp3_price is not None
        )
        if has_multi_levels and self.position is not None:
            pos = self.position
            for slot_num in (1, 2, 3):
                if self.position is None or pos.size <= 0:
                    break
                slot_px = getattr(pos, f"tp{slot_num}_price")
                if slot_px is None:
                    continue
                hit_mask_bit = 1 << (slot_num - 1)
                if pos.tp_hit_mask & hit_mask_bit:
                    continue

                slot_hit = False
                if pos.direction == 1 and float(bar.high) >= float(slot_px):
                    slot_hit = True
                elif pos.direction == -1 and float(bar.low) <= float(slot_px):
                    slot_hit = True

                if slot_hit:
                    # Check remaining active slots after this one
                    remaining_slots = [
                        s
                        for s in (1, 2, 3)
                        if s > slot_num and getattr(pos, f"tp{s}_price") is not None
                    ]
                    is_final = len(remaining_slots) == 0
                    if is_final:
                        close_size = pos.size
                    else:
                        init_size = float(pos.tp_initial_size or pos.size)
                        active_slots_at_entry = []
                        for s in (1, 2, 3):
                            if (pos.tp_hit_mask & (1 << (s - 1))) or getattr(
                                pos, f"tp{s}_price"
                            ) is not None:
                                active_slots_at_entry.append(s)
                        if slot_num in active_slots_at_entry:
                            idx_in_active = active_slots_at_entry.index(slot_num)
                            frac = (
                                pos.tp_lot_fractions[idx_in_active]
                                if idx_in_active < len(pos.tp_lot_fractions)
                                else (1.0 / len(active_slots_at_entry))
                            )
                        else:
                            frac = 1.0 / max(1, len(pos.tp_lot_fractions))
                        close_size = round(frac * init_size, 2)
                        close_size = min(close_size, pos.size)
                        if close_size <= 0:
                            close_size = pos.size

                    quote = self._sl_tp_fill_quote(int(pos.direction), float(slot_px))
                    pnl, fill_price = self._close_at_quote(pos, quote, size=close_size)
                    pos.tp_hit_mask |= hit_mask_bit
                    setattr(pos, f"tp{slot_num}_price", None)
                    record = {
                        "type": "tp_close" if pos.size <= 0 else "tp_partial_close",
                        "pnl": pnl,
                        "price": fill_price,
                        "reason": f"structure_tp{slot_num}",
                        "bar": idx,
                        "time": log_time,
                        "fill_time": None,
                        "event_time_start": log_time,
                        "event_time_end": log_time
                        + pd.Timedelta(minutes=getattr(self, "_minutes_per_step", 1)),
                        "time_precision": "bar_interval",
                        "equity": self.account.equity,
                        "tp_slot": slot_num,
                        "size": close_size,
                    }
                    self._stamp_exit(record, fill_price)
                    self.trade_log.append(record)
                    self.sessions_with_trades.add(self._current_session_id())

                    if pos.size <= 0:
                        self.position = None
                        self._note_close(pnl)
                        return
                    else:
                        pos.tp_price = (
                            pos.tp1_price
                            if pos.tp1_price is not None
                            else (pos.tp2_price if pos.tp2_price is not None else pos.tp3_price)
                        )
                        if self._breakeven_trigger_r > 0.0:
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
            if self.position is None:
                return
        elif self.position.tp_price is not None:
            tp_hit = False
            if self.position.direction == 1 and float(bar.high) >= self.position.tp_price:
                tp_hit = True
            elif self.position.direction == -1 and float(bar.low) <= self.position.tp_price:
                tp_hit = True

            if tp_hit:
                assert self.position.tp_price is not None
                pnl, fill_price = self._close_at_quote(
                    self.position,
                    self._sl_tp_fill_quote(
                        int(self.position.direction), float(self.position.tp_price)
                    ),
                )
                record = {
                    "type": "tp_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "structure_tp",
                    "bar": idx,
                    "time": log_time,
                    "fill_time": None,
                    "event_time_start": log_time,
                    "event_time_end": log_time
                    + pd.Timedelta(minutes=getattr(self, "_minutes_per_step", 1)),
                    "time_precision": "bar_interval",
                    "equity": self.account.equity,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.position = None
                self.sessions_with_trades.add(self._current_session_id())
                self._note_close(pnl)
                return

        self._maybe_ema_exit(bar, idx, log_time)

    def _check_multi_tp_ticks(self, bar_time: pd.Timestamp, idx: int) -> bool:
        """Replay observed bid/ask crossings in order, respecting remaining TP lots.

        Return True when a tick produced any exit. The caller must then avoid
        applying the bar's ambiguous OHLC crossings to the same position.
        """
        book = self._tickbook
        pos = self.position
        if book is None or pos is None:
            return False
        had_fill = False
        last_fill_ns = -1
        randomized_by_trigger: dict[int, tuple[Any, dict[str, Any]]] = {}
        bar_minutes = max(1, int(round(getattr(self, "_minutes_per_step", 1.0))))
        for tick_time, bid, ask in book.bar_quotes(bar_time, bar_minutes=bar_minutes):
            if self.position is not pos or pos.size <= 0:
                break
            if tick_time.value < last_fill_ns:
                continue
            if pos.entry_timestamp is not None and tick_time < pd.Timestamp(
                cast(Any, pos.entry_timestamp)
            ):
                continue
            exit_side = bid if pos.direction == 1 else ask
            sl = pos.sl_price
            stop_hit = sl is not None and (
                (pos.direction == 1 and exit_side <= sl)
                or (pos.direction == -1 and exit_side >= sl)
            )
            if stop_hit:
                execution_meta: dict[str, Any] = {}
                if (
                    getattr(self, "tick_execution_required", False)
                    or getattr(self, "randomized_fill_latency_ms", 0.0) > 0
                ):
                    cached = randomized_by_trigger.get(tick_time.value)
                    if cached is None:
                        quote, execution_meta = self._resolve_randomized_tick_quote(tick_time)
                        execution_meta["order_submission_time"] = tick_time
                        cached = (quote, execution_meta)
                        randomized_by_trigger[tick_time.value] = cached
                    quote, execution_meta = cached
                    bid, ask = quote.bid, quote.ask
                    execution_time = quote.tick_time
                else:
                    fill_time = tick_time + pd.Timedelta(milliseconds=self.fill_delay_ms)
                    quote_event = book.quote_event_at(fill_time)
                    if quote_event is None:
                        continue
                    quote_time, bid, ask = quote_event
                    execution_time = max(fill_time, quote_time)
                    execution_meta = {
                        "order_submission_time": tick_time,
                        "execution_order_time": tick_time,
                        "execution_arrival_time": fill_time,
                        "execution_latency_ns": int(
                            pd.Timedelta(milliseconds=self.fill_delay_ms).value
                        ),
                        "actual_tick_time": quote_time,
                        "fill_quote_source": "tick",
                    }
                remaining = pos.size
                pnl, fill_price = self._close_at_quote(pos, (bid, ask), tick_fill=True)
                record = {
                    "type": "stop_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "structure_sl",
                    "fill_quote_source": "tick",
                    "bar": idx,
                    "time": execution_time,
                    "equity": self.account.equity,
                    "size": remaining,
                    **execution_meta,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.position = None
                self.sessions_with_trades.add(self._current_session_id())
                self._note_close(pnl)
                return True

            for slot_num in (1, 2, 3):
                if self.position is None or pos.size <= 0:
                    return True
                bit = 1 << (slot_num - 1)
                slot_px = getattr(pos, f"tp{slot_num}_price")
                if slot_px is None or pos.tp_hit_mask & bit:
                    continue
                target_hit = (pos.direction == 1 and exit_side >= slot_px) or (
                    pos.direction == -1 and exit_side <= slot_px
                )
                if not target_hit:
                    continue
                execution_meta = {}
                if (
                    getattr(self, "tick_execution_required", False)
                    or getattr(self, "randomized_fill_latency_ms", 0.0) > 0
                ):
                    cached = randomized_by_trigger.get(tick_time.value)
                    if cached is None:
                        quote, execution_meta = self._resolve_randomized_tick_quote(tick_time)
                        execution_meta["order_submission_time"] = tick_time
                        cached = (quote, execution_meta)
                        randomized_by_trigger[tick_time.value] = cached
                    quote, execution_meta = cached
                    bid, ask = quote.bid, quote.ask
                    execution_time = quote.tick_time
                else:
                    fill_time = tick_time + pd.Timedelta(milliseconds=self.fill_delay_ms)
                    quote_event = book.quote_event_at(fill_time)
                    if quote_event is None:
                        continue
                    quote_time, bid, ask = quote_event
                    execution_time = max(fill_time, quote_time)
                    execution_meta = {
                        "order_submission_time": tick_time,
                        "execution_order_time": tick_time,
                        "execution_arrival_time": fill_time,
                        "execution_latency_ns": int(
                            pd.Timedelta(milliseconds=self.fill_delay_ms).value
                        ),
                        "actual_tick_time": quote_time,
                        "fill_quote_source": "tick",
                    }
                last_fill_ns = max(last_fill_ns, execution_time.value)
                remaining_slots = [
                    slot
                    for slot in (1, 2, 3)
                    if slot > slot_num
                    and getattr(pos, f"tp{slot}_price") is not None
                    and not (pos.tp_hit_mask & (1 << (slot - 1)))
                ]
                if not remaining_slots:
                    close_size = pos.size
                else:
                    init_size = float(pos.tp_initial_size or pos.initial_size or pos.size)
                    active = [
                        slot
                        for slot in (1, 2, 3)
                        if (pos.tp_hit_mask & (1 << (slot - 1)))
                        or getattr(pos, f"tp{slot}_price") is not None
                    ]
                    frac_idx = active.index(slot_num)
                    frac = (
                        pos.tp_lot_fractions[frac_idx]
                        if frac_idx < len(pos.tp_lot_fractions)
                        else 1.0 / len(active)
                    )
                    close_size = min(round(float(frac) * init_size, 2), pos.size)
                    if close_size <= 0:
                        close_size = pos.size

                pnl, fill_price = self._close_at_quote(
                    pos, (bid, ask), size=close_size, tick_fill=True
                )
                pos.tp_hit_mask |= bit
                setattr(pos, f"tp{slot_num}_price", None)
                record = {
                    "type": "tp_close" if pos.size <= 0 else "tp_partial_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": f"structure_tp{slot_num}",
                    "fill_quote_source": "tick",
                    "bar": idx,
                    "time": execution_time,
                    "equity": self.account.equity,
                    "tp_slot": slot_num,
                    "size": close_size,
                    **execution_meta,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.sessions_with_trades.add(self._current_session_id())
                had_fill = True
                if pos.size <= 0:
                    self.position = None
                    self._note_close(pnl)
                    return True

                pos.tp_price = next(
                    (
                        getattr(pos, f"tp{slot}_price")
                        for slot in (1, 2, 3)
                        if getattr(pos, f"tp{slot}_price") is not None
                    ),
                    None,
                )
                if self._breakeven_trigger_r > 0.0:
                    buffer = self._breakeven_buffer_pts
                    if pos.direction == 1:
                        new_sl = pos.entry_price + buffer
                        if pos.sl_price is None or new_sl > pos.sl_price:
                            pos.sl_price = new_sl
                            pos.breakeven_done = True
                    else:
                        new_sl = pos.entry_price - buffer
                        if pos.sl_price is None or new_sl < pos.sl_price:
                            pos.sl_price = new_sl
                            pos.breakeven_done = True
        return had_fill
