"""Differential Sharpe Ratio (DSR) reward.

Reference: Moody & Saffell (2001) "Learning to trade via direct reinforcement".
Reward = dS_t / dF_t · ΔF_t  (first-order approximation)

We also add:
- Soft penalty for FTMO daily-loss proximity
- Cost term (already captured in PnL but can be explicit)
- Hard breach → terminal negative reward
"""

from __future__ import annotations

import numpy as np


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
        initial_balance: float = 100_000.0,
        breach: bool = False,
    ) -> float:
        """Compute the DSR reward for one step.

        Parameters
        ----------
        step_pnl:
            Realised + unrealised P&L change for this bar.
        daily_loss, daily_loss_limit:
            Used for soft FTMO penalty.
        initial_balance:
            For normalisation.
        breach:
            Hard FTMO guardrail breached → large negative terminal reward.
        """
        if breach:
            return -1.0

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

        # Soft FTMO daily-loss proximity penalty (linear ramp in last 20% of limit)
        threshold = 0.8 * daily_loss_limit
        if daily_loss > threshold:
            excess = (daily_loss - threshold) / (daily_loss_limit - threshold + 1e-9)
            dsr -= 0.5 * float(np.clip(excess, 0, 1))

        return float(np.clip(dsr, -10.0, 10.0))
