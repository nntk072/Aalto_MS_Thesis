"""Order → fill broker model.

All execution methods accept a ``(bid, ask)`` quote tuple so that fills
honour real bid/ask pricing:

* Long open  → fill at ask  (buy order)
* Long close → fill at bid  (sell order)
* Short open → fill at bid  (sell order)
* Short close→ fill at ask  (buy order)

Mark-to-market also uses the close-side price:
* Long MTM   → valued at bid
* Short MTM  → valued at ask
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .account import AccountState
from .costs import CostModel


Direction = Literal[-1, 0, 1]
Quote = tuple[float, float]   # (bid, ask)


@dataclass
class Position:
    direction: int          # +1 long, -1 short
    size: float             # lots
    entry_price: float
    margin_used: float
    sl_price: float | None = None       # stop loss price level
    tp_price: float | None = None       # take profit price level
    risk_frac: float | None = None      # risk fraction used for sizing
    rr_ratio: float | None = None       # reward/risk ratio


@dataclass
class Broker:
    leverage: int = 50
    margin_pct: float = 0.02    # 2%
    contract_size: float = 1.0  # multiplier per lot
    cost_model: CostModel = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cost_model is None:
            self.cost_model = CostModel()

    def required_margin(self, price: float, lots: float) -> float:
        return price * lots * self.contract_size * self.margin_pct

    def open_position(
        self,
        acc: AccountState,
        quote: Quote,
        lots: float,
        direction: int,
    ) -> Position | None:
        """Attempt to open a position.

        Long opens fill at ask; short opens fill at bid.
        Returns the new ``Position`` or ``None`` if margin is insufficient or
        *direction* is not a valid trade side (must be ``+1`` or ``-1``).
        """
        if direction not in (1, -1):
            # Regression guard: engine action codes like exit_action (e.g. 2)
            # must never reach open_position. Reject rather than open a
            # bogus position with an undefined side.
            return None
        bid, ask = quote
        fill = self.cost_model.fill_price(bid, ask, direction)
        margin = self.required_margin(fill, lots)
        if acc.equity < margin:
            return None
        cost = self.cost_model.total_cost(lots)
        acc.balance -= cost
        return Position(
            direction=direction,
            size=lots,
            entry_price=fill,
            margin_used=margin,
        )

    def close_position(
        self,
        acc: AccountState,
        position: Position,
        quote: Quote,
    ) -> tuple[float, float]:
        """Close an open position and return (pnl, fill_price).

        Long closes fill at bid; short closes fill at ask.
        Commission is deducted from P&L before booking to the account.
        
        Returns:
            tuple of (realised_pnl, fill_price)
        """
        bid, ask = quote
        # Closing direction is opposite to the position direction
        fill = self.cost_model.fill_price(bid, ask, -position.direction)
        cost = self.cost_model.total_cost(position.size)
        pnl = (
            (fill - position.entry_price)
            * position.direction
            * position.size
            * self.contract_size
            - cost
        )
        acc.close_trade(pnl)
        return pnl, fill

    def mark_to_market(
        self,
        acc: AccountState,
        position: Position,
        quote: Quote,
    ) -> float:
        """Recompute and record unrealised P&L using the close-side price.

        * Long positions are valued at bid (what you'd receive if you sold now).
        * Short positions are valued at ask (what you'd pay to buy back now).
        """
        bid, ask = quote
        mtm_price = bid if position.direction == 1 else ask
        unrealised = (
            (mtm_price - position.entry_price)
            * position.direction
            * position.size
            * self.contract_size
        )
        acc.update_equity(unrealised)
        return unrealised

