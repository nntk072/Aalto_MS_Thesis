"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from gymnasium import spaces

from decisioning.actions import (
    ActionSchema,
    DiscreteActionSchema,
    build_action_schema,
    build_discrete_trading_action_schema,
)
from decisioning.observations import OBSERVATION_SCHEMA_V2, OBSERVATION_SCHEMA_V3

from .mixin_base import RuntimeMixinBase


class ActionEpisodeInitializationMixin(RuntimeMixinBase):
    action_space: Any

    def _initialize_action_and_episode_contracts(
        self,
        continuous_actions: bool,
        max_risk_frac: float,
        use_vae: bool,
        vae: Any,
        pre_ny_by_date: dict[Any, np.ndarray[Any, Any]] | None,
        pre_ny_data: pd.DataFrame | None,
        fill_latency_bars: int,
        max_episode_steps: int | None,
        normalize_account: bool,
    ) -> None:
        # Action space: strategy actions, continuous, or discrete
        self.continuous_actions = continuous_actions
        self.max_risk_frac = (
            self.trading_policy.max_agent_risk_frac
            if self.repeated_payout_policy
            else max_risk_frac
        )
        if self.repeated_payout_policy and not (self.strategy_actions or self.continuous_actions):
            raise ValueError(
                "repeated-payout mode requires a continuous risk action coordinate "
                "(strategy_actions=true or continuous_actions=true)"
            )
        self.action_schema: ActionSchema | DiscreteActionSchema

        # VAE for narrative embedding
        self.use_vae = use_vae
        self.vae = vae
        self.pre_ny_by_date: dict[Any, np.ndarray[Any, Any]] = (
            dict(pre_ny_by_date) if pre_ny_by_date is not None else {}
        )
        self.pre_ny_data = pre_ny_data  # legacy; prefer pre_ny_by_date
        # Chain C: decision-to-fill latency, expressed in whole M1 bars.
        # 0 = idealised next-bar fill (previous behaviour); 1 = 1-bar delay, etc.
        self.fill_latency_bars = max(0, int(fill_latency_bars))

        # Hard cap on episode length. None → run to data end; int → truncate
        # at that step count (Gymnasium standard: truncated=True).
        self.max_episode_steps: int | None = (
            int(max_episode_steps) if max_episode_steps is not None else None
        )
        # Whether to rescale the account vector so its per-element magnitudes
        # are comparable with the z-scored seq features. See __init__ docstring.
        self.normalize_account = bool(normalize_account)
        if (
            self.observation_schema.version in (OBSERVATION_SCHEMA_V2, OBSERVATION_SCHEMA_V3)
            and not self.normalize_account
        ):
            raise ValueError(f"{self.observation_schema.version} requires normalize_account=True")

        if use_vae:
            if vae is None:
                raise ValueError("vae must be provided when use_vae=True")
            if not self.pre_ny_by_date and pre_ny_data is None:
                raise ValueError(
                    "pre_ny_by_date (or legacy pre_ny_data) must be provided when use_vae=True"
                )
            # Freeze VAE encoder
            for param in vae.parameters():
                param.requires_grad = False

        if self.strategy_actions:
            # Trader-like Box (no free long/short dim).
            # PPO's unsquashed Gaussian mean sits near 0; a [0,1] box clipped
            # deterministic mean to 0 (always hold). Affine-map to unit interval
            # at decode: u = 0.5 * (clip(a, -1, 1) + 1).
            #
            # Default 5-D layout (strategy owns the side). Exit mode is the
            # last dimension:
            #   [0] entry intensity (hold only if threshold > 0 and u[0] < it;
            #       default 0 → context_direction decides enter vs hold)
            #   [1] stop distance ratio: 1.0 = at sl_reference, <1 tighter,
            #       >1 wider (mapped from raw[-1,1] to [0,2])
            #   [2] risk_frac -> risk_frac_range
            #   [3] target fraction among structural prices at least 1R away
            #   [4] exit mode: u >= 0.5 is ema_21, u < 0.5 is structural
            #
            # Opt-in 6-D layout (``agent_direction_control=True``) — dim 0 is
            # signed, so the agent can confirm, veto, or invert the heuristic
            # side when the pattern is failing. Exit mode stays last:
            #   [0] direction conviction +1 long .. -1 short (sign is the side;
            #       |u| < context_direction_threshold defers to the strategy)
            #   [1] entry intensity
            #   [2] stop distance ratio
            #   [3] risk_frac -> risk_frac_range
            #   [4] target fraction among structural prices at least 1R away
            #   [5] exit mode: u >= 0.5 is ema_21, u < 0.5 is structural
            #
            # Opt-in SL-mode dim (``allow_agent_sl_mode=True``) inserts one
            # dimension before exit so the invariant "exit mode is the last
            # dimension" survives:
            #   5-D -> 6-D: [intensity, stop, risk, target, sl_mode, exit]
            #   6-D -> 7-D: [direction, intensity, stop, risk, target, sl_mode, exit]
            self.action_schema = build_action_schema(
                has_direction=self.agent_direction_control,
                has_sl_mode=self.allow_agent_sl_mode,
                has_tp_mode=self.allow_agent_tp_mode,
                has_multi_tp=self.allow_multi_tp,
                has_simplex=self.allow_multi_tp and self.allow_simplex,
            )
            n_dims = self.action_schema.width
            self.action_space = spaces.Box(
                low=np.full(n_dims, -1.0, dtype=np.float32),
                high=np.full(n_dims, 1.0, dtype=np.float32),
                dtype=np.float32,
            )
        elif continuous_actions:
            # Continuous action space: Box(-1, 1) for position sizing
            # -1.0 = max short, +1.0 = max long, 0 = hold
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # Discrete action space: 0=hold, 1-9=enter_long, 10-18=enter_short, 19=exit
            self.action_space = spaces.Discrete(20)
            self.action_schema = build_discrete_trading_action_schema(20)


__all__ = ["ActionEpisodeInitializationMixin"]
