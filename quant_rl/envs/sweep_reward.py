"""Sweep Confirmation Reward for liquidity-sweep trading.

Implements: R_t = ΔPnL_t - Cost_t + α·C_t - β·T_t

Where:
- ΔPnL_t: Change in unrealised P&L
- Cost_t: Transaction costs
- C_t: Sweep Confirmation Score (automated heuristic)
- T_t: Time decay penalty
"""

from __future__ import annotations

import numpy as np


class SweepConfirmationReward:
    """Dense, label-free reward with sweep confirmation scoring.

    This reward function enables zero-human-label training by providing
    immediate feedback based on automated sweep detection heuristics.

    Parameters
    ----------
    alpha : float
        Weight for sweep confirmation score C_t (default: 0.1)
    beta : float
        Weight for time decay penalty T_t (default: 0.01)
    hold_bars : int
        Number of bars price must hold beyond level for confirmation (default: 3)
    cost_per_trade : float
        Estimated transaction cost per trade (default: 0.0)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.01,
        hold_bars: int = 3,
        cost_per_trade: float = 0.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.hold_bars = hold_bars
        self.cost_per_trade = cost_per_trade

        # State for sweep tracking (reset per episode)
        self.prev_price: float | None = None

        # London sweep tracking
        self.in_london_long_sweep: bool = False
        self.in_london_short_sweep: bool = False
        self.london_long_counter: int = 0
        self.london_short_counter: int = 0

        # Asian sweep tracking
        self.in_asian_long_sweep: bool = False
        self.in_asian_short_sweep: bool = False
        self.asian_long_counter: int = 0
        self.asian_short_counter: int = 0

    def reset(self) -> None:
        """Reset state for new episode."""
        self.prev_price = None
        self.in_london_long_sweep = False
        self.in_london_short_sweep = False
        self.london_long_counter = 0
        self.london_short_counter = 0
        self.in_asian_long_sweep = False
        self.in_asian_short_sweep = False
        self.asian_long_counter = 0
        self.asian_short_counter = 0

    def __call__(
        self,
        pnl_step: float,
        cost: float,
        price: float,
        london_high: float,
        london_low: float,
        asian_high: float,
        asian_low: float,
        minutes_since_open: float,
        position_changed: bool = False,
    ) -> float:
        """Compute reward for current step.

        Parameters
        ----------
        pnl_step : float
            Change in unrealised P&L from previous step
        cost : float
            Transaction cost for this step
        price : float
            Current price
        london_high : float
            London session high (liquidity level)
        london_low : float
            London session low (liquidity level)
        asian_high : float
            Asian session high (liquidity level)
        asian_low : float
            Asian session low (liquidity level)
        minutes_since_open : float
            Minutes elapsed since NY open (16:30 UTC+3)
        position_changed : bool
            Whether position was opened/closed/changed this step

        Returns
        -------
        float
            Reward value for this step
        """
        # Compute sweep confirmation score C_t
        c_t = self._compute_sweep_score(price, london_high, london_low, asian_high, asian_low)

        # Compute time decay T_t
        t_t = max(0.0, minutes_since_open - 20.0)

        # Compute transaction cost
        if position_changed:
            total_cost = cost + self.cost_per_trade
        else:
            total_cost = cost

        # Final reward: R_t = ΔPnL_t - Cost_t + α·C_t - β·T_t
        reward = pnl_step - total_cost + (self.alpha * c_t) - (self.beta * t_t)

        # Update previous price
        self.prev_price = price

        return float(reward)

    def _compute_sweep_score(
        self,
        price: float,
        london_high: float,
        london_low: float,
        asian_high: float,
        asian_low: float,
    ) -> float:
        """Compute C_t: Liquidity Sweep Confirmation Score.

        Returns:
            +1.0: London level swept and held for hold_bars
            +0.5: Asian level swept and held for hold_bars
            -0.5: False breakout (price closes back inside)
            0.0: No sweep detected
        """
        if self.prev_price is None:
            self.prev_price = price
            return 0.0

        prev = self.prev_price

        # Handle NaN values in levels
        if np.isnan(london_high) or np.isnan(london_low):
            london_high = london_low = float("nan")
        if np.isnan(asian_high) or np.isnan(asian_low):
            asian_high = asian_low = float("nan")

        # --- LONDON LEVEL SWEEPS (Priority) ---

        # Long sweep: price crosses ABOVE London High
        if not np.isnan(london_high) and price > london_high and prev <= london_high:
            self.in_london_long_sweep = True
            self.london_long_counter = 1
            self.in_asian_long_sweep = False  # London takes priority
            self.asian_long_counter = 0

        # Short sweep: price crosses BELOW London Low
        elif not np.isnan(london_low) and price < london_low and prev >= london_low:
            self.in_london_short_sweep = True
            self.london_short_counter = 1
            self.in_asian_short_sweep = False
            self.asian_short_counter = 0

        # Track London LONG sweep hold
        if self.in_london_long_sweep and not np.isnan(london_high):
            if price > london_high:
                self.london_long_counter += 1
                if self.london_long_counter >= self.hold_bars:
                    return 1.0  # Strongest signal: London level confirmed
            else:
                # False breakout: closed back below London High
                self.in_london_long_sweep = False
                self.london_long_counter = 0
                return -0.5

        # Track London SHORT sweep hold
        if self.in_london_short_sweep and not np.isnan(london_low):
            if price < london_low:
                self.london_short_counter += 1
                if self.london_short_counter >= self.hold_bars:
                    return 1.0  # Strongest signal: London level confirmed
            else:
                # False breakout: closed back above London Low
                self.in_london_short_sweep = False
                self.london_short_counter = 0
                return -0.5

        # --- ASIAN LEVEL SWEEPS (only if not in London sweep) ---

        if not (self.in_london_long_sweep or self.in_london_short_sweep):
            # Long sweep: price crosses ABOVE Asian High
            if not np.isnan(asian_high) and price > asian_high and prev <= asian_high:
                self.in_asian_long_sweep = True
                self.asian_long_counter = 1

            # Short sweep: price crosses BELOW Asian Low
            elif not np.isnan(asian_low) and price < asian_low and prev >= asian_low:
                self.in_asian_short_sweep = True
                self.asian_short_counter = 1

            # Track Asian LONG sweep hold
            if self.in_asian_long_sweep and not np.isnan(asian_high):
                if price > asian_high:
                    self.asian_long_counter += 1
                    if self.asian_long_counter >= self.hold_bars:
                        return 0.5  # Weaker signal: Asian level confirmed
                else:
                    # False breakout: closed back below Asian High
                    self.in_asian_long_sweep = False
                    self.asian_long_counter = 0
                    return -0.5

            # Track Asian SHORT sweep hold
            if self.in_asian_short_sweep and not np.isnan(asian_low):
                if price < asian_low:
                    self.asian_short_counter += 1
                    if self.asian_short_counter >= self.hold_bars:
                        return 0.5  # Weaker signal: Asian level confirmed
                else:
                    # False breakout: closed back above Asian Low
                    self.in_asian_short_sweep = False
                    self.asian_short_counter = 0
                    return -0.5

        # No sweep detected
        return 0.0


class CompositeReward:
    """Composite reward combining multiple reward components.

    This is a wrapper that allows combining SweepConfirmationReward
    with other reward signals (e.g., DSR from existing code).
    """

    def __init__(
        self,
        sweep_reward: SweepConfirmationReward,
        dsr_weight: float = 0.5,
        sweep_weight: float = 0.5,
        dsr_eta: float = 0.01,
    ):
        self.sweep_reward = sweep_reward
        self.dsr_weight = dsr_weight
        self.sweep_weight = sweep_weight
        # Persist DSR state across steps so the EMA Sharpe estimate
        # actually accumulates. A fresh DSRReward() per call would reset
        # _A/_B every step and kill the signal.
        from ..envs.reward import DSRReward

        self._dsr_fn = DSRReward(eta=dsr_eta)

    def reset(self) -> None:
        """Reset all component reward functions."""
        self.sweep_reward.reset()
        self._dsr_fn.reset()

    def __call__(
        self,
        pnl_step: float,
        *,
        daily_loss: float = 0.0,
        daily_loss_limit: float = 5_000.0,
        initial_balance: float = 100_000.0,
        breach: bool = False,
        # Additional parameters for sweep reward (optional)
        cost: float = 0.0,
        price: float | None = None,
        london_high: float | None = None,
        london_low: float | None = None,
        asian_high: float | None = None,
        asian_low: float | None = None,
        minutes_since_open: float = 0.0,
        position_changed: bool = False,
        dsr_reward: float | None = None,
    ) -> float:
        """Compute composite reward.

        Supports both DSR-style arguments and sweep-style arguments.
        For backward compatibility, accepts DSR arguments and uses sweep
        reward only when sweep parameters are provided.
        """
        # Compute DSR reward if not provided (reuses persistent state)
        if dsr_reward is None:
            dsr_reward = self._dsr_fn(
                pnl_step,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                initial_balance=initial_balance,
                breach=breach,
            )

        # If sweep parameters are provided, compute sweep reward
        if all(p is not None for p in [price, london_high, london_low, asian_high, asian_low]):
            sweep_r = self.sweep_reward(
                pnl_step,
                cost,
                float(price),  # type: ignore[arg-type]
                float(london_high),  # type: ignore[arg-type]
                float(london_low),  # type: ignore[arg-type]
                float(asian_high),  # type: ignore[arg-type]
                float(asian_low),  # type: ignore[arg-type]
                minutes_since_open,
                position_changed,
            )
            return (self.dsr_weight * dsr_reward) + (self.sweep_weight * sweep_r)
        else:
            # Fallback to DSR only
            return dsr_reward
