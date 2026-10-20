"""Focused simulation initialization service."""

from __future__ import annotations

from market_data.loading.ticks import TickBook

from .mixin_base import RuntimeMixinBase


class ExecutionControlsInitializationMixin(RuntimeMixinBase):
    def _initialize_execution_controls(
        self,
        tickbook: TickBook | None,
        tick_execution_required: bool,
        fill_delay_ms: int,
        randomized_fill_latency_ms: float,
        max_tick_quote_wait_ms: int,
    ) -> None:
        self._tickbook = tickbook
        self.tick_execution_required = bool(tick_execution_required)
        self.fill_delay_ms = int(fill_delay_ms)
        self.randomized_fill_latency_ms = float(randomized_fill_latency_ms)
        self.max_tick_quote_wait_ms = int(max_tick_quote_wait_ms)
        if self.randomized_fill_latency_ms < 0:
            raise ValueError("randomized_fill_latency_ms cannot be negative")
        if self.randomized_fill_latency_ms > 30.0:
            raise ValueError("randomized_fill_latency_ms must be at most 30ms exclusive")
        if self.max_tick_quote_wait_ms <= 0:
            raise ValueError("max_tick_quote_wait_ms must be positive")
        if (
            self.tick_execution_required or self.randomized_fill_latency_ms > 0
        ) and tickbook is None:
            raise ValueError("tick-only execution requires a TickBook")


__all__ = ["ExecutionControlsInitializationMixin"]
