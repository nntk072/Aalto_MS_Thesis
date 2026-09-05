"""Idea 1 strategy: PO3 manipulation/distribution + IFVG entry zones."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TradingStrategy


class PO3IFVGStrategy(TradingStrategy):
    """Entry chain: Asian context + sweep + manipulation + distribution + IFVG.

    The long candidate chain (Agent.md §17) requires Asian context available,
    a sell-side sweep, manipulation state detected, manipulation ended with
    distribution confirmed, a bullish IFVG zone active and price inside or
    retesting the zone. The short side mirrors. ``enforce_gate=False`` keeps
    the chain diagnostic-only (for debugging); ``True`` hard-gates entries.

    Structural SL (environment risk rule): long SL at the manipulation low,
    short SL at the manipulation high, with an optional configured buffer.
    """

    name = "po3_ifvg"
    required_features = (
        "asian_high",
        "asian_low",
        "sweep_high",
        "sweep_low",
        "po3_manipulation_low",
        "po3_manipulation_high",
        "po3_manipulation_end",
        "po3_distribution",
        "ifvg_bull_low",
        "ifvg_bull_high",
        "ifvg_bear_low",
        "ifvg_bear_high",
    )
    raw_columns = (
        "asian_high",
        "asian_low",
        "sweep_high_level",
        "sweep_low_level",
        "po3_manipulation_high",
        "po3_manipulation_low",
        "ifvg_bull_low",
        "ifvg_bull_high",
        "ifvg_bear_low",
        "ifvg_bear_high",
        "last_swing_high",
        "last_swing_low",
    )

    def __init__(self, *, enforce_gate: bool = True, require_asian_context: bool = True):
        self.enforce_gate = enforce_gate
        self.require_asian_context = require_asian_context

    def _asian_ok(self, row: pd.Series) -> bool:
        if not self.require_asian_context:
            return True
        return bool(
            np.isfinite(float(row.get("asian_high", np.nan)))
            and np.isfinite(float(row.get("asian_low", np.nan)))
        )

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Full PO3+IFVG chain when the gate is enforced (Agent.md §17)."""
        if not self.enforce_gate:
            return True
        if not self._asian_ok(row):
            return False
        if direction == 1:
            return bool(
                float(row.get("sweep_low", 0.0)) > 0
                and float(row.get("po3_distribution", 0.0)) > 0
                and float(row.get("po3_distribution_direction", 0.0)) == 1
                and float(row.get("ifvg_bull_active", 0.0)) > 0
                and float(row.get("price_in_ifvg_bull", 0.0)) > 0
            )
        if direction == -1:
            return bool(
                float(row.get("sweep_high", 0.0)) > 0
                and float(row.get("po3_distribution", 0.0)) > 0
                and float(row.get("po3_distribution_direction", 0.0)) == -1
                and float(row.get("ifvg_bear_active", 0.0)) > 0
                and float(row.get("price_in_ifvg_bear", 0.0)) > 0
            )
        return False

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Long -> manipulation low, short -> manipulation high (Agent.md §14)."""
        col = "po3_manipulation_low" if direction == 1 else "po3_manipulation_high"
        level = float(row.get(col, np.nan))
        valid = np.isfinite(level) and (
            level < float(row["close"]) if direction == 1 else level > float(row["close"])
        )
        return level if valid else None

    def target_candidates(self, *, direction: int, row: pd.Series) -> dict[str, float]:
        """Named structural targets; the TP resolver validates each side."""

        def _get(col: str) -> float:
            value = float(row.get(col, np.nan))
            return value if np.isfinite(value) else np.nan

        if direction == 1:
            return {
                "buyside_liquidity": _get("sweep_high_level"),
                "previous_day_high_low": _get("prev_day_high"),
            }
        return {
            "sellside_liquidity": _get("sweep_low_level"),
            "previous_day_high_low": _get("prev_day_low"),
        }
