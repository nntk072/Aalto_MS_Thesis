"""Focused simulation execution service."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from market_data.loading.ticks import (
    TickExecutionUnavailable,
    resolve_tick_execution,
    resolve_tick_exit,
)

from .mixin_base import RuntimeMixinBase


class TickQuoteRuntimeMixin(RuntimeMixinBase):
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
            pd.Timedelta(milliseconds=self.fill_delay_ms),
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
