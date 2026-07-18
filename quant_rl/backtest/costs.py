"""Transaction cost model: spread, slippage, commission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Holds cost parameters and computes fill costs."""

    spread_points: float = 0.6  # US100 default
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0  # 0 as per spec
    point_value: float = 1.0  # $ per point per unit of contract_size
    point_size: float = 0.01  # minimum price increment (e.g., 0.01 for US100)

    def total_cost(self, lots: float, direction: int | None = None) -> float:
        """Return the explicit transaction cost for one side of a trade.

        Spread/slippage are modeled in :meth:`fill_price` (bid/ask execution),
        so this method should only include explicit broker fees (commission).

        Parameters
        ----------
        lots:
            Positive trade size in lots.
        direction:
            +1 = long, -1 = short (unused here but provided for extensibility).
        """
        return self.commission_per_lot * lots

    def fill_price(self, bid: float, ask: float, direction: int) -> float:
        """Return the fill price for a market order.

        Direction mapping:
        - ``+1``: buy side (ask)
        - ``-1``: sell side (bid)

        Parameters
        ----------
        bid : float
            Bid price
        ask : float
            Ask price
        direction : int
            +1 for long (buy at ask), -1 for short (sell at bid)
        """
        if direction == 1:
            return ask
        else:  # direction == -1
            return bid

    def bar_quote(self, close: float, bar_spread: float | None = None) -> tuple[float, float]:
        """Derive bid/ask from close price using spread.

        In MT5 data, close is typically the bid price.
        Ask is derived as bid + spread.

        Parameters
        ----------
        close : float
            Bar close price (MT5 close = bid)
        bar_spread : float | None
            Spread in points from data, or use default

        Returns
        -------
        tuple[float, float]
            (bid, ask) prices
        """
        sp = bar_spread if bar_spread is not None else self.spread_points
        bid = close
        ask = close + sp
        return bid, ask


COST_US100 = CostModel(spread_points=0.6)
COST_US500 = CostModel(spread_points=1.53)
