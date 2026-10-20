"""Multi-target position invariants."""

from __future__ import annotations

import math

from trading.execution.broker import Position


def reorder_remaining_targets(
    pos: Position, *, point_size: float, price_tolerance: float = 1e-8
) -> None:
    """Disable invalid remaining TP slots without fabricating target prices."""
    tol = max(float(price_tolerance), float(point_size) / 2.0)
    last_price: float | None = None
    for slot in (1, 2, 3):
        bit = 1 << (slot - 1)
        attr = f"tp{slot}_price"
        level = getattr(pos, attr)
        if level is None or (pos.tp_hit_mask & bit):
            continue
        price = float(level)
        valid = math.isfinite(price) and (
            (pos.direction == 1 and price > pos.entry_price + tol)
            or (pos.direction == -1 and price < pos.entry_price - tol)
        )
        if valid and last_price is not None:
            valid = price > last_price + tol if pos.direction == 1 else price < last_price - tol
        if not valid:
            setattr(pos, attr, None)
        else:
            last_price = price
    if hasattr(pos, "tp_price"):
        pos.tp_price = next(
            (
                getattr(pos, f"tp{slot}_price")
                for slot in (1, 2, 3)
                if getattr(pos, f"tp{slot}_price") is not None
            ),
            None,
        )
