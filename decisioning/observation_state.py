"""Stable coordinates shared by simulation observations and encoders."""

from __future__ import annotations

from typing import Final, Literal

EntryState = Literal["FLAT", "CANDIDATE", "ARMED", "IN_POSITION"]
BASE_ACCOUNT_DIM: Final[int] = 6
ENTRY_STATES: Final[tuple[EntryState, ...]] = ("FLAT", "CANDIDATE", "ARMED", "IN_POSITION")
ENTRY_STATE_DIM: Final[int] = len(ENTRY_STATES)


def entry_state_one_hot(state: str) -> tuple[float, float, float, float]:
    """Encode the entry lifecycle state using the stable observation order."""
    try:
        index = ENTRY_STATES.index(state)
    except ValueError as exc:
        raise ValueError(f"unknown entry state: {state!r}") from exc
    return tuple(float(index == i) for i in range(ENTRY_STATE_DIM))  # type: ignore[return-value]
