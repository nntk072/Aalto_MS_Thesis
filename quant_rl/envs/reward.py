"""Differential Sharpe Ratio (DSR) and PnL rewards.

Reference: Moody & Saffell (2001) "Learning to trade via direct reinforcement".
Reward = dS_t / dF_t · ΔF_t  (first-order approximation)

We also add:
- Soft penalty for FTMO daily-loss proximity
- Soft penalty for year max-loss proximity (loss_from_initial)
- Cost term (already captured in PnL but can be explicit)
- Hard breach → terminal negative reward
"""

from __future__ import annotations

import numpy as np


def _soft_band_penalty(
    value: float,
    soft: float | None,
    hard: float,
    *,
    weight: float = 0.5,
) -> float:
    """Linear penalty in ``(soft, hard]``; 0 below soft."""
    soft_v = float(soft) if soft is not None and soft > 0.0 else 0.0
    if soft_v <= 0.0 or hard <= soft_v or value <= soft_v:
        return 0.0
    excess = (value - soft_v) / (hard - soft_v + 1e-9)
    return weight * float(np.clip(excess, 0.0, 1.0))


class DSRReward:
    """Online differential Sharpe reward with FTMO soft penalties."""

    def __init__(self, eta: float = 0.01) -> None:
        self.eta = eta  # EMA damping for A and B estimates
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0

    def __call__(
        self,
        step_pnl: float,
        *,
        daily_loss: float = 0.0,
        daily_loss_limit: float = 5_000.0,
        soft_daily_loss_limit: float | None = 2_000.0,
        loss_from_initial: float = 0.0,
        soft_max_loss_limit: float | None = 5_000.0,
        max_loss_limit: float = 10_000.0,
        initial_balance: float = 100_000.0,
        breach: bool = False,
        realized_close_pnl: float | None = None,  # noqa: ARG002
        equity: float | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the DSR reward for one step.

        Parameters
        ----------
        step_pnl:
            Realised + unrealised P&L change for this bar.
        daily_loss, daily_loss_limit, soft_daily_loss_limit:
            Soft FTMO penalty ramps from the soft brick toward the hard limit.
        loss_from_initial, soft_max_loss_limit, max_loss_limit:
            Soft year-loss shaping from soft floor toward absolute max loss.
        initial_balance:
            For normalisation.
        breach:
            Hard FTMO guardrail breached → large negative terminal reward.
        """
        if breach:
            return -10.0

        # Normalise P&L to relative return
        r = step_pnl / initial_balance

        # Update EMA estimates
        A_prev = self._A
        B_prev = self._B
        self._A = A_prev + self.eta * (r - A_prev)
        self._B = B_prev + self.eta * (r**2 - B_prev)

        denom = self._B - self._A**2
        if denom <= 1e-10:
            dsr = 0.0
        else:
            dsr = (self._B * (r - A_prev) - 0.5 * self._A * (r**2 - B_prev)) / (denom**1.5)

        # Soft brick → hard limit: linear penalty between soft and hard daily loss.
        soft = (
            float(soft_daily_loss_limit)
            if soft_daily_loss_limit is not None and soft_daily_loss_limit > 0.0
            else 0.8 * daily_loss_limit
        )
        dsr -= _soft_band_penalty(daily_loss, soft, daily_loss_limit)
        dsr -= _soft_band_penalty(loss_from_initial, soft_max_loss_limit, max_loss_limit)

        return float(np.clip(dsr, -10.0, 10.0))


class PnLReward:
    """Realized-PnL reward with soft daily-loss penalty from 1% of equity.

    Used for Idea 1/2 trader-like training (``env.reward_mode: pnl``).
    Baseline Idea 3 keeps :class:`DSRReward`.
    """

    def __init__(self, *, soft_loss_frac: float = 0.01, dsr_weight: float = 0.0) -> None:
        self.soft_loss_frac = soft_loss_frac
        self.dsr_weight = float(dsr_weight)
        self._dsr = DSRReward() if self.dsr_weight > 0 else None

    def reset(self) -> None:
        if self._dsr is not None:
            self._dsr.reset()

    def __call__(
        self,
        step_pnl: float,
        *,
        daily_loss: float = 0.0,
        daily_loss_limit: float = 5_000.0,
        soft_daily_loss_limit: float | None = 2_000.0,
        loss_from_initial: float = 0.0,
        soft_max_loss_limit: float | None = 5_000.0,
        max_loss_limit: float = 10_000.0,
        initial_balance: float = 100_000.0,
        breach: bool = False,
        realized_close_pnl: float | None = None,
        equity: float | None = None,
    ) -> float:
        if breach:
            return -10.0

        reward = 0.0
        if realized_close_pnl is not None:
            reward += float(realized_close_pnl) / initial_balance

        eq = float(equity) if equity is not None else initial_balance
        soft_start = (
            float(soft_daily_loss_limit)
            if soft_daily_loss_limit is not None and soft_daily_loss_limit > 0.0
            else self.soft_loss_frac * max(eq, 1.0)
        )
        reward -= _soft_band_penalty(daily_loss, soft_start, daily_loss_limit)
        reward -= _soft_band_penalty(loss_from_initial, soft_max_loss_limit, max_loss_limit)

        if self._dsr is not None:
            dsr = self._dsr(
                step_pnl,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                soft_daily_loss_limit=soft_daily_loss_limit,
                loss_from_initial=loss_from_initial,
                soft_max_loss_limit=soft_max_loss_limit,
                max_loss_limit=max_loss_limit,
                initial_balance=initial_balance,
                breach=False,
            )
            reward += self.dsr_weight * dsr

        return float(np.clip(reward, -10.0, 10.0))
