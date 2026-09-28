"""Idea 2 strategy-alignment reward (Agent.md §22).

Event-based, never continuous: rewards the *entry* transition only when the
distribution leg is already past the opposing-gap failure and the trade
points with that leg. A break of structure is not required.
"""

from __future__ import annotations


class DistributionReward:
    """Reward for a single position-entry event under the Idea 2 chain.

    Parameters
    ----------
    entry_bonus:
        Positive bonus for an entry on a distribution bar after the gap failed.
    sweep_penalty:
        Negative reward for an entry that is not on that bar.
    distribution_bonus:
        Positive bonus for an entry during the distribution phase.
    """

    #: Keyword inputs this reward consumes (for CompositeReward filtering).
    required_inputs: tuple[str, ...] = (
        "position_changed",
        "direction",
        "distribution_after_ifvg",
        "distribution_phase",
    )

    def __init__(
        self,
        entry_bonus: float,
        sweep_penalty: float,
        distribution_bonus: float,
    ) -> None:
        self.entry_bonus = float(entry_bonus)
        self.sweep_penalty = float(sweep_penalty)
        self.distribution_bonus = float(distribution_bonus)

    def reset(self) -> None:
        """Reset per-episode state (no-op: the reward is stateless)."""

    def __call__(
        self,
        *,
        position_changed: bool,
        direction: int,
        distribution_after_ifvg: bool,
        distribution_phase: bool,
    ) -> float:
        """Compute the entry-event reward for the current step.

        Only a position *transition* (``position_changed`` and a non-zero
        ``direction``) earns anything. The entry is consistent when the
        open is on a distribution bar after the opposing gap failed.
        """
        if not position_changed or direction == 0:
            return 0.0

        reward = 0.0
        if distribution_after_ifvg:
            reward += self.entry_bonus
        else:
            reward -= self.sweep_penalty
        if distribution_phase:
            reward += self.distribution_bonus
        return reward
