"""Idea 2 strategy: distribution + swing high/low (sweep -> BOS chains)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TradingStrategy


class DistributionStrategy(TradingStrategy):
    """Entry chain: sweep -> reclaim -> BOS -> distribution candidate.

    Long candidates need a reclaimed sell-side sweep followed by a
    close-through break of structure up; the short side mirrors. The
    structural stop references the swept extreme (Agent.md §14, §18).
    """

    name = "distribution"
    required_features = (
        "sweep_high",
        "sweep_low",
        "sweep_high_level",
        "sweep_low_level",
        "sweep_high_reclaimed",
        "sweep_low_reclaimed",
        "bos_up",
        "bos_down",
    )
    raw_columns = (
        "sweep_high_level",
        "sweep_low_level",
        "bos_up_level",
        "bos_down_level",
        "last_swing_high",
        "last_swing_low",
    )

    def __init__(self, *, enforce_gate: bool = True):
        self.enforce_gate = enforce_gate

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Directionally consistent sweep->reclaim->BOS chain when gated."""
        if not self.enforce_gate:
            return True
        if direction == 1:
            return bool(
                float(row.get("sweep_low_reclaimed", 0.0)) > 0 and float(row.get("bos_up", 0.0)) > 0
            )
        if direction == -1:
            return bool(
                float(row.get("sweep_high_reclaimed", 0.0)) > 0
                and float(row.get("bos_down", 0.0)) > 0
            )
        return False

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Long -> swept low, short -> swept high (Agent.md §14)."""
        col = "sweep_low_level" if direction == 1 else "sweep_high_level"
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
