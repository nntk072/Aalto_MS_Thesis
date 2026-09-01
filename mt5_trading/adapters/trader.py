from abc import ABC, abstractmethod
from typing import Any


class Trader(ABC):
    @abstractmethod
    def open_position(self, *args: Any, **kwargs: Any) -> Any:
        """Open a position; returns the broker result object or None on failure."""
        raise NotImplementedError

    @abstractmethod
    def close_positions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_opened_positions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_all_positions(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def send_to_break_even(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def calculate_position_size(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
