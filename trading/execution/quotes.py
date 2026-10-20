"""Bar-to-executable quote conversion."""

from __future__ import annotations

import math

from trading.execution.costs import CostModel


def quote_from_bar_price(
    cost_model: CostModel, price: float, spread_points: float
) -> tuple[float, float]:
    spread = (
        float(spread_points) * cost_model.point_size
        if math.isfinite(float(spread_points))
        else None
    )
    return cost_model.bar_quote(float(price), bar_spread=spread)
