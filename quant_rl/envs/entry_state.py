"""Causal lifecycle for persistent entry opportunities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

EntryState = Literal["FLAT", "CANDIDATE", "ARMED", "IN_POSITION"]
BASE_ACCOUNT_DIM: Final[int] = 6
ENTRY_STATES: Final[tuple[EntryState, ...]] = ("FLAT", "CANDIDATE", "ARMED", "IN_POSITION")
ENTRY_STATE_DIM: Final[int] = len(ENTRY_STATES)


def entry_state_one_hot(state: str) -> tuple[float, float, float, float]:
    """Return the observation encoding for one entry lifecycle state."""
    try:
        index = ENTRY_STATES.index(state)
    except ValueError as exc:
        raise ValueError(f"unknown entry state: {state!r}") from exc
    return tuple(float(index == i) for i in range(ENTRY_STATE_DIM))  # type: ignore[return-value]


def immutable_setup_levels(
    values: Mapping[str, float] | tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Copy finite setup geometry into a stable, immutable tuple."""
    items = values.items() if isinstance(values, Mapping) else values
    copied = []
    for key, raw_value in items:
        value = float(raw_value)
        if math.isfinite(value):
            copied.append((str(key), value))
    return tuple(sorted(copied))


@dataclass(frozen=True, slots=True)
class EntryEvidence:
    """Market interpretation supplied to the lifecycle state machine."""

    bar: int
    evidence_detected: bool
    invalidated: bool
    arm_condition: bool
    retest_confirmed: bool
    direction: int
    setup_types: tuple[str, ...]
    setup_levels_snapshot: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar", int(self.bar))
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "setup_types", tuple(self.setup_types))
        object.__setattr__(
            self,
            "setup_levels_snapshot",
            immutable_setup_levels(self.setup_levels_snapshot),
        )


@dataclass(frozen=True, slots=True)
class EntryCandidate:
    """Immutable opportunity details captured on its origin bar."""

    direction: int
    setup_types: tuple[str, ...]
    origin_bar: int
    setup_levels_snapshot: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if int(self.direction) not in (-1, 1):
            raise ValueError("candidate direction must be -1 or 1")
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "setup_types", tuple(self.setup_types))
        object.__setattr__(self, "origin_bar", int(self.origin_bar))
        object.__setattr__(
            self,
            "setup_levels_snapshot",
            immutable_setup_levels(self.setup_levels_snapshot),
        )


class EntryStateMachine:
    """Manage candidate, armed, and executed entry opportunity states."""

    def __init__(self, candidate_max_age_bars: int, arm_requires_retest: bool = False) -> None:
        if int(candidate_max_age_bars) < 0:
            raise ValueError("candidate_max_age_bars must be non-negative")
        self.candidate_max_age_bars = int(candidate_max_age_bars)
        self.arm_requires_retest = bool(arm_requires_retest)
        self.state: EntryState = "FLAT"
        self.candidate: EntryCandidate | None = None
        # Per-episode lifecycle counters. Kept here (not on the env) so the
        # FSM owns its own reporting surface; the env only reads them out.
        self.visits: dict[EntryState, int] = {
            "FLAT": 0,
            "CANDIDATE": 0,
            "ARMED": 0,
            "IN_POSITION": 0,
        }
        self.candidate_created: int = 0
        self.candidate_expired: int = 0
        self.candidate_invalidated: int = 0
        self.trigger_requested: int = 0
        self.trigger_refused: int = 0
        self.entered: int = 0
        self.closed: int = 0
        self.armed_bars: int = 0

    def reset(self) -> None:
        """Return to the initial state at an episode boundary."""
        self.state = "FLAT"
        self.candidate = None
        self.visits = {"FLAT": 0, "CANDIDATE": 0, "ARMED": 0, "IN_POSITION": 0}
        self.candidate_created = 0
        self.candidate_expired = 0
        self.candidate_invalidated = 0
        self.trigger_requested = 0
        self.trigger_refused = 0
        self.entered = 0
        self.closed = 0
        self.armed_bars = 0

    def observe(self, evidence: EntryEvidence) -> EntryState:
        """Apply one bar of evidence without changing executed trade state."""
        if self.state == "IN_POSITION":
            return self.state
        if self.state == "FLAT":
            if (
                evidence.evidence_detected
                and not evidence.invalidated
                and evidence.direction in (-1, 1)
            ):
                self.candidate = EntryCandidate(
                    direction=evidence.direction,
                    setup_types=evidence.setup_types,
                    origin_bar=evidence.bar,
                    setup_levels_snapshot=evidence.setup_levels_snapshot,
                )
                self.candidate_created += 1
                self._enter("CANDIDATE")
            return self.state

        candidate = self.candidate
        if candidate is None:
            self._clear()
            return self.state
        if evidence.bar < candidate.origin_bar:
            raise ValueError("entry evidence bars must be monotonic")

        opposing_evidence = evidence.direction == -candidate.direction and (
            evidence.evidence_detected or evidence.retest_confirmed
        )
        if evidence.invalidated or opposing_evidence:
            self.candidate_invalidated += 1
            self._clear()
            return self.state

        if self.state == "CANDIDATE":
            age = evidence.bar - candidate.origin_bar
            if age > self.candidate_max_age_bars:
                self.candidate_expired += 1
                self._clear()
            elif evidence.arm_condition and (
                not self.arm_requires_retest
                or (evidence.retest_confirmed and evidence.direction == candidate.direction)
            ):
                self._enter("ARMED")
        return self.state

    def request_trigger(self) -> EntryCandidate | None:
        """Return the armed candidate for an attempt without consuming it."""
        if self.state != "ARMED":
            return None
        self.trigger_requested += 1
        return self.candidate

    def mark_entered(self) -> None:
        """Record successful position creation for the armed candidate."""
        if self.state != "ARMED" or self.candidate is None:
            raise RuntimeError("only an armed candidate can enter a position")
        self.entered += 1
        self._enter("IN_POSITION")

    def mark_closed(self) -> None:
        """Clear lifecycle state after the environment closes the position."""
        if self.state != "IN_POSITION":
            raise RuntimeError("only an open position can be marked closed")
        self.closed += 1
        self._clear()

    def invalidate(self) -> None:
        """Clear an unexecuted opportunity."""
        if self.state in ("CANDIDATE", "ARMED"):
            self.candidate_invalidated += 1
            self._clear()

    def is_armed(self) -> bool:
        """Whether PPO may request an execution attempt."""
        return self.state == "ARMED"

    def is_candidate(self) -> bool:
        """Whether an unarmed candidate is waiting for confirmation."""
        return self.state == "CANDIDATE"

    def _clear(self) -> None:
        self.state = "FLAT"
        self.candidate = None

    def _enter(self, state: EntryState) -> None:
        """Record a state transition and count bars spent in ARMED."""
        if self.state == "ARMED":
            self.armed_bars += 1
        self.state = state
        self.visits[state] += 1
