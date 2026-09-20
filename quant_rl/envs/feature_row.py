"""Lightweight numpy-backed row views for TradingEnv hot path.

Avoids per-step ``pd.Series`` / ``bars.iloc`` construction while remaining
duck-compatible with strategy APIs that expect Series-like ``.get`` /
``__getitem__`` / ``.index`` membership.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BarView:
    """OHLC (+ optional spread) for a single bar without pandas."""

    open: float
    high: float
    low: float
    close: float
    spread: float = float("nan")

    def __getitem__(self, key: str) -> float:
        if key == "open":
            return self.open
        if key == "high":
            return self.high
        if key == "low":
            return self.low
        if key == "close":
            return self.close
        if key == "spread":
            return self.spread
        raise KeyError(key)

    @property
    def index(self) -> frozenset[str]:
        keys = {"open", "high", "low", "close"}
        if np.isfinite(self.spread):
            keys = keys | {"spread"}
        return frozenset(keys)


class FeatureRow:
    """Zero-copy view of one feature-matrix row (numpy + column map)."""

    __slots__ = ("_values", "_col_to_idx", "_index")

    def __init__(self, values: np.ndarray[Any, Any], col_to_idx: dict[str, int]) -> None:
        self._values = values
        self._col_to_idx = col_to_idx
        # Strategies use ``col not in row.index`` / ``col in row.index``.
        self._index = col_to_idx

    @property
    def index(self) -> dict[str, int]:
        return self._index

    def get(self, key: str, default: Any = np.nan) -> Any:
        idx = self._col_to_idx.get(key)
        if idx is None:
            return default
        return float(self._values[idx])

    def __getitem__(self, key: str) -> float:
        idx = self._col_to_idx[key]
        return float(self._values[idx])

    def __contains__(self, key: object) -> bool:
        return key in self._col_to_idx

    def __iter__(self) -> Iterator[str]:
        return iter(self._col_to_idx)
