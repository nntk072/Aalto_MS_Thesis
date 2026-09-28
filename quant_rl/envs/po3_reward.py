"""Idea 1 strategy-alignment reward (Agent.md §21).

Event-based, never continuous: it rewards the *entry* transition when the
bar is a liquidity sweep or a protected-swing confirmation in the trade
direction. An entry on neither is penalised. A sweep is often still inside
manipulation, so that penalty is not charged on a sweep or swing entry.
"""

from __future__ import annotations


class PO3Reward:
    """Reward for a single position-entry event under the Idea 1 chain.

    Parameters
    ----------
    entry_bonus:
        Positive bonus for a sweep or protected-swing entry.
    manipulation_penalty:
        Negative reward for an entry that is not a sweep or swing while
        manipulation is still running.
    invalid_ifvg_penalty:
        Negative reward for an entry that is not a sweep or protected swing.
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
        "liquidity",
        "in_gap",
        "entry_setup",
    )

    def __init__(
        self,
        entry_bonus: float,
        manipulation_penalty: float,
        invalid_ifvg_penalty: float,
        distribution_bonus: float,
        sweep_penalty: float = 0.0,
    ) -> None:
        self.entry_bonus = float(entry_bonus)
        self.manipulation_penalty = float(manipulation_penalty)
        self.invalid_ifvg_penalty = float(invalid_ifvg_penalty)
        self.distribution_bonus = float(distribution_bonus)
        self.sweep_penalty = float(sweep_penalty)

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
        liquidity: bool = True,
        in_gap: bool | None = None,
        entry_setup: bool = False,
    ) -> float:
        """Compute the entry-event reward for the current step.

        Only a position *transition* (``position_changed`` and a non-zero
        ``direction``) earns anything — a position that merely stays open is
        never rewarded here (Agent.md §21).
        """
        if not position_changed or direction == 0:
            return 0.0

        del in_ifvg, in_gap
        reward = 0.0
        if entry_setup:
            reward += self.entry_bonus
        else:
            if manipulation_active and not manipulation_end:
                reward -= self.manipulation_penalty
            reward -= self.invalid_ifvg_penalty
            if self.sweep_penalty and not liquidity:
                reward -= self.sweep_penalty
        if distribution_phase:
            reward += self.distribution_bonus
        return reward
