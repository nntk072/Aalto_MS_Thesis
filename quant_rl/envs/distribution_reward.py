"""Idea 2 strategy-alignment reward (Agent.md §22).

Event-based, never continuous: rewards the *entry* transition only when it
follows a directionally consistent sweep -> reclaim -> break-of-structure
chain (long: sweep_low + BOS_up; short: sweep_high + BOS_down). Rewards
only the matching mapping and never the opposite one.
"""

from __future__ import annotations


class DistributionReward:
    """Reward for a single position-entry event under the Idea 2 chain.

    Parameters
    ----------
    entry_bonus:
        Positive bonus for an entry on a consistent sweep->BOS chain.
    sweep_penalty:
        Negative reward for an entry with no supporting sweep/BOS context.
    distribution_bonus:
        Positive bonus for an entry during the distribution phase.
    """

    #: Keyword inputs this reward consumes (for CompositeReward filtering).
    required_inputs: tuple[str, ...] = (
        "position_changed",
        "direction",
        "sweep_high",
        "sweep_low",
        "bos_up",
        "bos_down",
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
        sweep_high: bool,
        sweep_low: bool,
        bos_up: bool,
        bos_down: bool,
        distribution_phase: bool,
    ) -> float:
        """Compute the entry-event reward for the current step.

        Only a position *transition* (``position_changed`` and a non-zero
        ``direction``) earns anything. Only the directionally consistent
        chain is rewarded; the opposite mapping is never rewarded
        (Agent.md §22).
        """
        if not position_changed or direction == 0:
            return 0.0

        reward = 0.0
        # Long: sweep_low then BOS_up. Short: sweep_high then BOS_down.
        consistent = (direction == 1 and sweep_low and bos_up) or (
            direction == -1 and sweep_high and bos_down
        )
        if consistent:
            reward += self.entry_bonus
        else:
            reward -= self.sweep_penalty
        if distribution_phase:
            reward += self.distribution_bonus
        return reward
