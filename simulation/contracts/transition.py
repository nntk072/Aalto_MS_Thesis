"""Framework-neutral result of one simulation state transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class SimulationTransition(Generic[ObservationT]):
    """Semantic outcome of advancing a simulator by one action."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    def as_gymnasium(
        self,
    ) -> tuple[ObservationT, float, bool, bool, dict[str, Any]]:
        """Adapt this transition to Gymnasium's five-value step protocol."""
        return (
            self.observation,
            self.reward,
            self.terminated,
            self.truncated,
            dict(self.info),
        )


__all__ = ["SimulationTransition"]
