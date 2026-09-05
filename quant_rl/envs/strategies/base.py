"""Base interface for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class TradingStrategy(ABC):
    """Strategy semantics consumed by :class:`TradingEnv`.

    Implementations must be stateless with respect to time: all methods
    receive the feature row of the decision bar and answer from it.
    """

    name: str = "base"

    #: Feature columns that must exist in the feature matrix (Agent.md §11).
    required_features: tuple[str, ...] = ()

    #: Raw price-level columns kept for execution/risk but excluded from the
    #: model's normalised sequence observation (Agent.md §11).
    raw_columns: tuple[str, ...] = ()

    @abstractmethod
    def validate_entry(self, *, direction: int, row: pd.Series) -> bool:
        """Whether an entry in ``direction`` is allowed at this bar."""

    @abstractmethod
    def sl_reference(self, *, direction: int, row: pd.Series) -> float | None:
        """Structural SL reference price, or None if unavailable/invalid."""

    @abstractmethod
    def target_candidates(self, *, direction: int, row: pd.Series) -> dict[str, float]:
        """Named take-profit candidate levels for the TP-target resolver."""
