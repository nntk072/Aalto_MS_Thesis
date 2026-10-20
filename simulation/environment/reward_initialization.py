"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any

from .mixin_base import RuntimeMixinBase
from .rewards import DSRReward, EquityDeltaReward, PnLReward, RMultipleReward
from .sweep_reward import CompositeReward, SweepConfirmationReward


class RewardInitializationMixin(RuntimeMixinBase):
    def _initialize_reward_runtime(
        self,
        trader_actions: bool | None,
        strategy_actions: bool,
        use_sweep_reward: bool,
        strategy_reward: Any,
        dsr_eta: float,
        sweep_alpha: float,
        sweep_beta: float,
        sweep_hold_bars: int,
        dsr_weight: float,
        sweep_weight: float,
        strategy_weight: float,
    ) -> bool:
        # Resolve trader/strategy action flag early (needed for reward + spaces).
        _strategy_actions = (
            bool(trader_actions) if trader_actions is not None else bool(strategy_actions)
        )

        # CompositeReward zeroes both dsr and sweep weights in pnl mode, so sweep
        # inputs would drop the realized-PnL term entirely. Refuse the combination
        # instead of training on a signal that is only strategy shaping (plan E5).
        if self.reward_mode == "pnl" and use_sweep_reward:
            raise ValueError(
                "reward_mode='pnl' is incompatible with use_sweep_reward=True: the "
                "composite reward would drop the PnL term. Use reward_mode='dsr' for "
                "sweep reward."
            )

        # A hard risk breach receives the same terminal penalty in training and
        # evaluation. The guardrail also ends the episodic training trajectory.
        self.reward_fn: (
            DSRReward | CompositeReward | PnLReward | RMultipleReward | EquityDeltaReward
        )
        breach_penalty = -10.0
        if self.reward_mode == "equity_delta_r":
            if self.risk_mode != "fixed" or self.fixed_risk_usd <= 0.0:
                raise ValueError(
                    "reward_mode='equity_delta_r' requires risk_mode='fixed' and fixed_risk_usd > 0"
                )
            if use_sweep_reward or strategy_reward is not None:
                raise ValueError(
                    "reward_mode='equity_delta_r' cannot be combined with sweep or "
                    "strategy reward shaping"
                )
            self.reward_fn = EquityDeltaReward(self.fixed_risk_usd, breach_penalty=breach_penalty)
            self.dsr_reward = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)
        else:
            self.reward_fn = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)
        base_pnl = (
            PnLReward(soft_loss_frac=0.01, dsr_weight=0.0, breach_penalty=breach_penalty)
            if self.reward_mode in ("pnl", "rr")
            else None
        )
        if self.reward_mode == "equity_delta_r":
            pass
        elif (
            use_sweep_reward
            or (_strategy_actions and strategy_reward is not None)
            or (self.reward_mode == "pnl" and strategy_reward is not None)
        ):
            sweep_reward = SweepConfirmationReward(
                alpha=sweep_alpha, beta=sweep_beta, hold_bars=sweep_hold_bars
            )
            self.reward_fn = CompositeReward(
                sweep_reward=sweep_reward,
                dsr_weight=0.0 if self.reward_mode == "pnl" else dsr_weight,
                sweep_weight=0.0 if self.reward_mode == "pnl" else sweep_weight,
                strategy_reward=strategy_reward,
                strategy_weight=strategy_weight,
                base_reward=base_pnl,
            )
            self.dsr_reward = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)
        elif self.reward_mode == "rr":
            # R-multiple objective. Only a plain RMultipleReward: no sweep or
            # strategy shaping, so the agent is scored purely on cumulative R.
            self.reward_fn = RMultipleReward(breach_penalty=breach_penalty)
            self.dsr_reward = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)
        elif self.reward_mode == "pnl":
            self.reward_fn = PnLReward(
                soft_loss_frac=0.01, dsr_weight=0.0, breach_penalty=breach_penalty
            )
            self.dsr_reward = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)
        else:
            self.reward_fn = DSRReward(eta=dsr_eta, breach_penalty=breach_penalty)

        return _strategy_actions


__all__ = ["RewardInitializationMixin"]
