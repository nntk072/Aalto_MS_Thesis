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
from math import isfinite
from typing import Literal

from trading.account.state import AccountState

from .costs import CostModel

Direction = Literal[-1, 0, 1]
Quote = tuple[float, float]  # (bid, ask)


def _validate_quote(quote: Quote) -> Quote:
    """Refuse impossible or non-finite executable prices before account mutation."""
    bid, ask = quote
    if not isfinite(bid) or not isfinite(ask) or bid <= 0.0 or ask < bid:
        raise ValueError("Invalid bid/ask quote: prices must be finite, positive and ask >= bid")
    return bid, ask


@dataclass
class Position:
    direction: int  # +1 long, -1 short
    size: float  # lots
    entry_price: float
    margin_used: float
    # Identifiers and quantities describe the original position and its fills.
    # They remain stable as size shrinks through partial target execution.
    position_id: str = ""
    initial_size: float = 0.0
    filled_size: float = 0.0
    fill_count: int = 0
    position_net_pnl: float = 0.0
    sl_price: float | None = None  # stop loss price level
    tp_price: float | None = None  # take profit price level
    risk_frac: float | None = None  # risk fraction used for sizing
    rr_ratio: float | None = None  # reward/risk ratio
    entry_timestamp: object | None = None
    entry_atr: float | None = None
    best_favorable: float = 0.0
    last_progress_favorable: float = 0.0
    stale_counter: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    #: Planned loss at the structural stop in account currency. Set by the
    #: environment at entry from the lot size and stop distance. The
    #: intraday adverse-excursion guard uses it as its floor so sizing and
    #: the guard cannot disagree.
    planned_risk_usd: float | None = None
    #: Stop level as placed at entry. Breakeven management compares against
    #: this, not against a rewritten ``sl_price``.
    sl_initial_price: float | None = None
    #: True once the stop has been moved to the breakeven level.
    breakeven_done: bool = False
    #: Structural geometry chosen at entry. ``planned_rr`` is empty for an
    #: EMA-21 exit. ``rr_ratio`` is not reused for that meaning.
    sl_ref: str = ""
    tp_ref: str = ""
    planned_rr: float | None = None
    exit_mode: str = ""
    stop_distance: float = 0.0
    sl_buffer: float = 0.0
    manipulation: str = ""
    n_sl: int = 0
    n_tp: int = 0
    entry_sl_ref_price: float | None = None
    entry_last_swing_high: float | None = None
    entry_last_swing_low: float | None = None
    sl_mode: str = "fixed"
    sl_mode_overridden: bool = False
    tp_initial_size: float | None = None
    tp_mode: str = "fixed"
    tp_mode_overridden: bool = False
    tp_breakeven_done: bool = False
    tp_initial_price: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    tp3_price: float | None = None
    tp1_ref: str = ""
    tp2_ref: str = ""
    tp3_ref: str = ""
    tp_lot_fractions: tuple[float, ...] = ()
    tp_hit_mask: int = 0
    entry_tp_ref_price: float | None = None
    entry_setup_types: tuple[str, ...] = ()
    entry_origin_bar: int | None = None


@dataclass
class Broker:
    leverage: int = 50
    margin_pct: float = 0.02  # 2%
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
        if direction not in (1, -1) or not isfinite(lots) or lots <= 0.0:
            # Regression guard: engine action codes like exit_action (e.g. 2)
            # must never reach open_position. Reject rather than open a
            # bogus position with an undefined side.
            return None
        bid, ask = _validate_quote(quote)
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
            initial_size=lots,
        )

    def close_position(
        self,
        acc: AccountState,
        position: Position,
        quote: Quote,
        size: float | None = None,
    ) -> tuple[float, float]:
        """Close an open position (or part of it) and return (pnl, fill_price).

        Long closes fill at bid; short closes fill at ask.
        Commission is deducted from P&L before booking to the account.

        Returns:
            tuple of (realised_pnl, fill_price)
        """
        if not isfinite(position.size) or position.size <= 0.0:
            raise ValueError("Cannot close a position with no remaining quantity")
        if size is not None and (not isfinite(float(size)) or float(size) <= 0.0):
            raise ValueError("Requested close quantity must be finite and positive")
        bid, ask = _validate_quote(quote)
        # Closing direction is opposite to the position direction
        fill = self.cost_model.fill_price(bid, ask, -position.direction)
        close_size = position.size if size is None else min(float(size), position.size)
        cost = self.cost_model.total_cost(close_size)
        pnl = (
            fill - position.entry_price
        ) * position.direction * close_size * self.contract_size - cost
        acc.close_trade(pnl)
        position.size -= close_size
        if position.size <= 1e-9:
            position.size = 0.0
            position.margin_used = 0.0
        else:
            position.margin_used = self.required_margin(position.entry_price, position.size)
            self.mark_to_market(acc, position, quote)
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
        bid, ask = _validate_quote(quote)
        mtm_price = bid if position.direction == 1 else ask
        unrealised = (
            (mtm_price - position.entry_price)
            * position.direction
            * position.size
            * self.contract_size
        )
        acc.update_equity(unrealised)
        return unrealised
