"""Broker-neutral contracts for live inference and execution adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import pandas as pd


class LiveSignal(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NONE = "none"


class MarketDataSource(Protocol):
    def get_symbol(self) -> str: ...

    def get_data(self) -> pd.DataFrame: ...


class LiveStrategy(Protocol):
    def signal(self) -> tuple[str, LiveSignal]: ...


__all__ = ["LiveSignal", "LiveStrategy", "MarketDataSource"]
