"""Order → fill broker model.

Handles margin calculation, leverage, and order execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .account import AccountState
from .costs import CostModel


Direction = Literal[-1, 0, 1]


@dataclass
class Position:
    direction: int          # +1 long, -1 short
    size: float             # lots
    entry_price: float
    margin_used: float


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
        price: float,
        lots: float,
        direction: int,
        spread_points: float | None = None,
    ) -> Position | None:
        """Attempt to open a position; returns Position or None if insufficient margin."""
        fill_price = self.cost_model.fill_price(price, direction, spread_points=spread_points)
        margin = self.required_margin(fill_price, lots)
        if acc.equity < margin:
            return None
        cost = self.cost_model.total_cost(lots, direction)
        acc.balance -= cost
        return Position(
            direction=direction,
            size=lots,
            entry_price=fill_price,
            margin_used=margin,
        )

    def close_position(
        self,
        acc: AccountState,
        position: Position,
        price: float,
        spread_points: float | None = None,
    ) -> float:
        """Close position at *price*, return realised P&L."""
        fill_price = self.cost_model.fill_price(price, -position.direction, spread_points=spread_points)
        cost = self.cost_model.total_cost(position.size, -position.direction)
        pnl = (
            (fill_price - position.entry_price)
            * position.direction
            * position.size
            * self.contract_size
            - cost
        )
        acc.close_trade(pnl)
        return pnl

    def mark_to_market(self, acc: AccountState, position: Position, price: float) -> float:
        """Compute unrealised P&L for an open position."""
        unrealised = (
            (price - position.entry_price)
            * position.direction
            * position.size
            * self.contract_size
        )
        acc.update_equity(unrealised)
        return unrealised
