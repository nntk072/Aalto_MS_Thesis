# ruff: noqa: F403, F405
from __future__ import annotations

from trading.execution.quotes import quote_from_bar_price

from .runtime_dependencies import *


class ExecutionRuntimeMixin(RuntimeMixinBase):
    position: Position | None
    _last_realized_close_pnl: float | None
    _last_realized_close_r: float | None
    _pending_realized_r: float | None

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
            pnl, fill_price = self.broker.close_position(self.account, self.position, quote)
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
                    pnl, fill_price = self.broker.close_position(
                        self.account,
                        pos,
                        quote,
                        size=close_size,
                    )
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
                pnl, fill_price = self.broker.close_position(
                    self.account,
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
                    fill_time = tick_time + pd.Timedelta(milliseconds=self._fill_delay_ms)
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
                            pd.Timedelta(milliseconds=self._fill_delay_ms).value
                        ),
                        "actual_tick_time": quote_time,
                        "fill_quote_source": "tick",
                    }
                remaining = pos.size
                pnl, fill_price = self.broker.close_position(self.account, pos, (bid, ask))
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
                    fill_time = tick_time + pd.Timedelta(milliseconds=self._fill_delay_ms)
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
                            pd.Timedelta(milliseconds=self._fill_delay_ms).value
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

                pnl, fill_price = self.broker.close_position(
                    self.account, pos, (bid, ask), size=close_size
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

    def _tick_exit_quote(
        self, bar_time: pd.Timestamp
    ) -> tuple[str, tuple[float, float], pd.Timestamp, dict[str, Any]] | None:
        """Delayed tick quote for a stop or target, or ``None`` to use the level."""
        book = self._tickbook
        pos = self.position
        if book is None or pos is None:
            return None
        if pos.sl_price is None and pos.tp_price is None:
            return None
        if getattr(self, "tick_execution_required", False) or self.randomized_fill_latency_ms > 0:
            hit = resolve_tick_exit(
                book,
                pd.Timestamp(bar_time),
                int(pos.direction),
                None if pos.sl_price is None else float(pos.sl_price),
                None if pos.tp_price is None else float(pos.tp_price),
                self.np_random,
                bar_minutes=max(1, int(round(self._minutes_per_step))),
                max_latency_ns_exclusive=int(self.randomized_fill_latency_ms * 1_000_000),
                max_quote_wait_ns=self.max_tick_quote_wait_ms * 1_000_000,
                earliest_time=(
                    None
                    if pos.entry_timestamp is None
                    else pd.Timestamp(cast(Any, pos.entry_timestamp))
                ),
            )
            if hit is None:
                return None
            quote = hit.quote
            meta = {
                "order_submission_time": hit.trigger_time,
                "execution_order_time": quote.order_time,
                "execution_arrival_time": quote.arrival_time,
                "execution_latency_ns": quote.latency_ns,
                "actual_tick_time": quote.tick_time,
                "fill_quote_source": "tick",
            }
            return hit.kind, (quote.bid, quote.ask), quote.tick_time, meta
        legacy_hit = book.delayed_exit_event(
            pd.Timestamp(bar_time),
            int(pos.direction),
            None if pos.sl_price is None else float(pos.sl_price),
            None if pos.tp_price is None else float(pos.tp_price),
            pd.Timedelta(milliseconds=self._fill_delay_ms),
            earliest_time=(
                None
                if pos.entry_timestamp is None
                else pd.Timestamp(cast(Any, pos.entry_timestamp))
            ),
        )
        if legacy_hit is None:
            return None
        kind, fill_time, bid, ask = legacy_hit
        return (
            kind,
            (bid, ask),
            fill_time,
            {
                "order_submission_time": pd.Timestamp(bar_time),
                "execution_order_time": pd.Timestamp(bar_time),
                "execution_arrival_time": pd.Timestamp(bar_time),
                "execution_latency_ns": 0,
                "actual_tick_time": fill_time,
                "fill_quote_source": "tick",
            },
        )

    def _resolve_randomized_tick_quote(
        self, order_time: pd.Timestamp
    ) -> tuple[Any, dict[str, Any]]:
        """Resolve the first eligible tick and return auditable execution fields.

        A configured latency of zero is an immediate tick lookup, not a candle
        fill. Positive values sample integer nanoseconds below the configured
        exclusive upper bound.
        """
        if self._tickbook is None:
            raise TickExecutionUnavailable("tick-only execution requires a TickBook")
        quote = resolve_tick_execution(
            self._tickbook,
            pd.Timestamp(order_time),
            self.np_random,
            max_latency_ns_exclusive=int(self.randomized_fill_latency_ms * 1_000_000),
            max_quote_wait_ns=self.max_tick_quote_wait_ms * 1_000_000,
        )
        if quote is None:
            raise TickExecutionUnavailable(
                f"no executable quote within {self.max_tick_quote_wait_ms}ms of {order_time}"
            )
        return quote, {
            "order_submission_time": quote.order_time,
            "execution_order_time": quote.order_time,
            "execution_arrival_time": quote.arrival_time,
            "execution_latency_ns": quote.latency_ns,
            "actual_tick_time": quote.tick_time,
            "fill_quote_source": "tick",
        }

    def _sl_tp_fill_quote(self, direction: int, hit_price: float) -> tuple[float, float]:
        """Bid/ask that closes at *hit_price*, worsened only by adverse slippage.

        Longs sell the bid and shorts buy the ask, so that side is the stop
        or target. The other side is one model spread away so
        ``Broker.close_position`` still sees a two-sided quote.
        """
        # Configured spread is already in price units. point_size is only
        # used to convert raw MT5 spread-column values and for price precision.
        spread = float(self.cost_model.spread_points)
        # Slippage is quoted in price points, the same unit as ``min_sl_points``,
        # ``sl_buffer_pts`` and ``swing_buffer_pts``, so it is *not* scaled by
        # ``point_size`` the way the tick-denominated spread fallback is. It was
        # scaled before, which meant ``slippage_points: 10`` bought 0.1 of a
        # point — an order of magnitude below the slippage the sizing floor is
        # written to tolerate. Positive only ever widens the fill against us.
        slip = float(self.cost_model.slippage_points)
        if direction == 1:
            bid = hit_price - slip
            ask = bid + spread
        else:
            ask = hit_price + slip
            bid = ask - spread
        return bid, ask

    def _stop_fill_quote(
        self, bar: BarView, direction: int, stop_price: float
    ) -> tuple[float, float]:
        """Price a bar stop at its level unless the bar opens through it."""
        raw_spread = float(bar.spread)
        spread = (
            raw_spread * self.cost_model.point_size
            if np.isfinite(raw_spread)
            else self.cost_model.spread_points
        )
        if direction == 1:
            trigger_price = min(stop_price, float(bar.open))
        else:
            opening_ask = float(bar.open) + spread
            trigger_price = max(stop_price, opening_ask)
        return self._sl_tp_fill_quote(direction, trigger_price)

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

    def _bar_quote(self, bar: BarView) -> tuple[float, float]:
        """Get (bid, ask) from a bar using cost model.

        The bar's raw ``spread`` column is in MT5 broker points, not price
        units, so it must be scaled by ``point_size`` before being used as a
        price-unit spread (see ``simulation.backtest.engine._bar_spread_price_units``).
        """
        return quote_from_bar_price(self.cost_model, float(bar.close), float(bar.spread))

    def _bar_open_quote(self, bar: BarView) -> tuple[float, float]:
        """Get an executable bid/ask from a bar's opening price."""
        return quote_from_bar_price(self.cost_model, float(bar.open), float(bar.spread))
