"""Transaction cost model: spread, slippage, commission.

Bid/ask semantics
-----------------
MT5 bar ``close`` prices are **bid-based**.  The ask is derived by adding the
spread.  All order fills obey:

* **Buy**  (long open / short close) → fill at **ask** + slippage
* **Sell** (short open / long close) → fill at **bid** − slippage

The spread is therefore charged exactly **once** per round-trip via the
``fill_price`` method.  ``total_cost`` covers commissions only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Holds cost parameters and computes bid/ask fill prices.

    Parameters
    ----------
    spread_points:
        Fixed bid-ask spread in price units (e.g. 0.6 USD for US100).
        Used when no per-bar or tick-level spread is available.
    slippage_points:
        Extra market-impact slippage in price units added on top of the spread.
    commission_per_lot:
        Round-trip commission per lot in account currency.
    point_size:
        Price increment per "MT5 spread point" in the bar CSV spread column.
        Converts the integer ``spread`` column to price units.
        Typical value: 0.01 for US100 (prices quoted to 2 d.p.).
    """

    spread_points: float = 0.6       # price units (USD for US100)
    slippage_points: float = 0.0     # price units
    commission_per_lot: float = 0.0
    point_size: float = 0.01         # CSV spread integer → price units

    # ------------------------------------------------------------------
    # Fill price
    # ------------------------------------------------------------------

    def fill_price(self, bid: float, ask: float, direction: int) -> float:
        """Return the execution price for a market order.

        Parameters
        ----------
        bid, ask:
            Current best bid and ask prices.
        direction:
            ``+1`` = buy (long open or short close) → fills at ask + slippage.
            ``-1`` = sell (short open or long close) → fills at bid − slippage.
        """
        if direction > 0:
            return ask + self.slippage_points
        return bid - self.slippage_points

    # ------------------------------------------------------------------
    # Commission cost
    # ------------------------------------------------------------------

    def total_cost(self, lots: float) -> float:
        """Return commission cost for one leg (open *or* close).

        The bid-ask spread is already captured by ``fill_price``; this method
        covers explicit fees only.
        """
        return self.commission_per_lot * lots

    # ------------------------------------------------------------------
    # Bar-based quote construction (fallback when no tick data)
    # ------------------------------------------------------------------

    def bar_quote(
        self,
        close: float,
        bar_spread: float | None = None,
    ) -> tuple[float, float]:
        """Build a ``(bid, ask)`` quote from an MT5 bar close price.

        MT5 bars are bid-based, so ``close`` is the bid.  The ask is derived
        by adding the spread.

        Parameters
        ----------
        close:
            Bar close price (= last bid of the bar in MT5).
        bar_spread:
            Raw integer spread from the CSV ``spread`` column.  Converted to
            price units via ``point_size``.  If ``None``, ``spread_points``
            (the fixed model spread) is used instead.
        """
        if bar_spread is not None:
            sp = bar_spread * self.point_size
        else:
            sp = self.spread_points
        return close, close + sp


# ---------------------------------------------------------------------------
# Pre-built instrument defaults
# ---------------------------------------------------------------------------

COST_US100 = CostModel(spread_points=0.6,  point_size=0.01)
COST_US500 = CostModel(spread_points=1.53, point_size=0.01)
