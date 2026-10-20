"""Focused simulation execution service."""

from __future__ import annotations

import numpy as np

from trading.execution.quotes import quote_from_bar_price

from .market_view import BarView
from .mixin_base import RuntimeMixinBase


class FillQuoteRuntimeMixin(RuntimeMixinBase):
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
