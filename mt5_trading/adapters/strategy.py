from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mt5_trading.domain.signal import Signal


class TradingStrategy(ABC):
    @abstractmethod
    def signal(self) -> tuple[str, Signal]:
        raise NotImplementedError
