"""Baseline strategy interface.

Strategies are step-indexed: the environment calls ``act()`` exactly once
per bar in order, so a strategy can pre-compute a signal array from the
bars DataFrame at construction time and emit ``signals[idx]`` on each call.
This keeps strategies independent of the observation layout and trivially
testable with synthetic data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStrategy(ABC):
    """Common interface for all non-learned and learned baselines.

    Subclasses implement :meth:`_signal` (one action per bar index) or
    override :meth:`act` directly. Actions follow the continuous contract
    of ``TradingEnv``: positive = long fraction, negative = short fraction,
    0 = hold/exit.
    """

    def __init__(self, n_bars: int) -> None:
        """Store the expected number of bars and reset the step counter.

        Args:
            n_bars: Length of the pre-computed signal array.
        """
        self._n_bars = n_bars
        self._idx = 0

    def reset(self) -> None:
        """Reset the internal step counter for a new episode."""
        self._idx = 0

    def act(self, obs: dict[str, Any]) -> float:
        """Return the action for the current bar.

        Args:
            obs: Environment observation (ignored by signal-based
                strategies; kept for interface compatibility).

        Returns:
            Continuous action in ``[-1.0, 1.0]``.
        """
        signal = float(self._signal(min(self._idx, self._n_bars - 1)))
        self._idx += 1
        return signal

    @abstractmethod
    def _signal(self, idx: int) -> float:
        """Return the action for bar ``idx``."""
