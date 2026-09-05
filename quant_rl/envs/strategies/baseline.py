"""Baseline (Idea 3) strategy: autonomous behaviour, unchanged."""

from __future__ import annotations

import pandas as pd

from .base import TradingStrategy


class BaselineStrategy(TradingStrategy):
    """Control-group strategy (Idea 3, P2).

    Adds no gates, no structural stop references and no strategy features —
    the environment behaves exactly as before when this strategy is active.
    """

    name = "baseline"
    required_features = ()
    raw_columns = ()

    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:  # noqa: ARG002
        """No gate: the baseline agent decides freely."""
        return True

    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:  # noqa: ARG002
        """No structural stop: baseline keeps its existing SL behaviour."""
        return None

    def target_candidates(self, *, direction: int, row: pd.Series) -> dict[str, float]:  # noqa: ARG002
        """No structural targets: baseline keeps its existing TP behaviour."""
        return {}
