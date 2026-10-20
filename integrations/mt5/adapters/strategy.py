from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from integrations.mt5.domain.signal import Signal


class TradingStrategy(ABC):
    @abstractmethod
    def signal(self) -> tuple[str, Signal]:
        raise NotImplementedError
