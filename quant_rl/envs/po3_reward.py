"""Idea 1 strategy-alignment reward (Agent.md §21).

Event-based, never continuous: it rewards (or penalises) the *entry*
transition only, driving the agent toward entries inside a confirmed IFVG
zone during the distribution phase and away from entries during active
manipulation. It intentionally stays small relative to the base economic
reward so the strategy signal shapes behaviour without dominating PnL.
"""

from __future__ import annotations


class PO3Reward:
    """Reward for a single position-entry event under the Idea 1 chain.

    Parameters
    ----------
    entry_bonus:
        Positive bonus for entering inside a confirmed IFVG zone.
    manipulation_penalty:
        Negative reward for entering while manipulation is still running.
    invalid_ifvg_penalty:
        Negative reward for entering outside any confirmed IFVG zone.
    distribution_bonus:
        Positive bonus for entering during the distribution phase.
    """

    #: Keyword inputs this reward consumes (for CompositeReward filtering).
    required_inputs: tuple[str, ...] = (
        "position_changed",
        "direction",
        "in_ifvg",
        "manipulation_active",
        "manipulation_end",
        "distribution_phase",
    )

    def __init__(
        self,
        entry_bonus: float,
        manipulation_penalty: float,
        invalid_ifvg_penalty: float,
        distribution_bonus: float,
    ) -> None:
        self.entry_bonus = float(entry_bonus)
        self.manipulation_penalty = float(manipulation_penalty)
        self.invalid_ifvg_penalty = float(invalid_ifvg_penalty)
        self.distribution_bonus = float(distribution_bonus)

    def reset(self) -> None:
        """Reset per-episode state (no-op: the reward is stateless)."""

    def __call__(
        self,
        *,
        position_changed: bool,
        direction: int,
        in_ifvg: bool,
        manipulation_active: bool,
        manipulation_end: bool,
        distribution_phase: bool,
    ) -> float:
        """Compute the entry-event reward for the current step.

        Only a position *transition* (``position_changed`` and a non-zero
        ``direction``) earns anything — a position that merely stays open is
        never rewarded here (Agent.md §21).
        """
        if not position_changed or direction == 0:
            return 0.0

        reward = 0.0
        if manipulation_active and not manipulation_end:
            reward -= self.manipulation_penalty
        if in_ifvg:
            reward += self.entry_bonus
        else:
            reward -= self.invalid_ifvg_penalty
        if distribution_phase:
            reward += self.distribution_bonus
        return reward
