"""Transaction cost model: spread, slippage, commission."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Holds cost parameters and computes fill costs."""

    spread_points: float = 0.6         # US100 default
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0    # 0 as per spec
    point_value: float = 1.0           # $ per point per unit of contract_size

    def total_cost(self, lots: float, direction: int) -> float:
        """Return the cost (deducted from P&L) for opening *or* closing a trade.

        Parameters
        ----------
        lots:
            Positive trade size in lots.
        direction:
            +1 = long, -1 = short (unused here but provided for extensibility).
        """
        half_spread = (self.spread_points + self.slippage_points) / 2.0
        cost = half_spread * lots * self.point_value
        cost += self.commission_per_lot * lots
        return cost

    def fill_price(self, mid_price: float, direction: int, spread_points: float | None = None) -> float:
        """Return the fill price for a market order."""
        sp = spread_points if spread_points is not None else self.spread_points
        slip = self.slippage_points
        half = (sp + slip) / 2.0
        return mid_price + direction * half


COST_US100 = CostModel(spread_points=0.6)
COST_US500 = CostModel(spread_points=1.53)
