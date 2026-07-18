"""Gymnasium TradingEnv.

Observation space
-----------------
``Dict``:
  - ``"seq"``: Box float32 [T, F] – rolling feature window
  - ``"account"``: Box float32 [5] – normalised account state

Action space
------------
Discrete(3): 0=flat, 1=long, 2=short  (mapped internally to {0, +1, -1})

Episode termination
-------------------
- FTMO hard breach (daily loss / max drawdown)
- End of data
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ..backtest.account import AccountState
from ..backtest.broker import Broker, Position
from ..backtest.costs import CostModel, COST_US100
from ..backtest.guardrails import FTMOGuardrails
from .reward import DSRReward


_ACTION_MAP = {0: 0, 1: 1, 2: -1}   # gym discrete → internal direction


class TradingEnv(gym.Env):
    """Event-driven single-instrument trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        obs_window: int = 60,
        cost_model: CostModel = COST_US100,
        initial_balance: float = 100_000.0,
        lots: float = 1.0,
        ftmo_kwargs: dict | None = None,
        dsr_eta: float = 0.01,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()

        # Align to common index
        common = bars.index.intersection(features.index)
        self._bars = bars.loc[common].reset_index(drop=False)
        self._features = features.loc[common].values.astype(np.float32)
        self._features = np.nan_to_num(self._features, nan=0.0)

        self._obs_window = obs_window
        self._initial_balance = initial_balance
        self._lots = lots
        self._cost_model = cost_model
        self._ftmo_kwargs = ftmo_kwargs or {}
        self._dsr_eta = dsr_eta
        self._max_steps = max_episode_steps

        n_feat = self._features.shape[1]
        acc_dim = 5

        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(
                    low=-10.0, high=10.0, shape=(obs_window, n_feat), dtype=np.float32
                ),
                "account": spaces.Box(
                    low=-1.0, high=1.0, shape=(acc_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short

        # Will be set in reset()
        self._idx: int = obs_window
        self._acc: AccountState | None = None
        self._broker: Broker | None = None
        self._guardrails: FTMOGuardrails | None = None
        self._reward_fn: DSRReward | None = None
        self._position: Position | None = None
        self._prev_equity: float = initial_balance
        self._prev_session: int = -1
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._idx = self._obs_window
        self._acc = AccountState(initial_balance=self._initial_balance)
        self._broker = Broker(cost_model=self._cost_model)
        self._guardrails = FTMOGuardrails(**self._ftmo_kwargs)
        self._reward_fn = DSRReward(eta=self._dsr_eta)
        self._position = None
        self._prev_equity = self._initial_balance
        self._prev_session = -1
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._acc is not None, "Call reset() before step()"

        row = self._bars.iloc[self._idx]
        price = float(row["close"])
        spread_points = float(row["spread"]) if "spread" in row.index and pd.notna(row["spread"]) else None
        session = int(row["session_id"]) if "session_id" in row.index else 0

        # Session reset
        if session != self._prev_session:
            self._acc.reset_daily()
            self._prev_session = session

        # Mark-to-market
        if self._position is not None:
            self._broker.mark_to_market(self._acc, self._position, price)  # type: ignore[union-attr]

        # FTMO check
        breach_reason = self._guardrails.breach_reason(self._acc)  # type: ignore[union-attr]
        if breach_reason:
            if self._position is not None:
                self._broker.close_position(  # type: ignore[union-attr]
                    self._acc, self._position, price, spread_points=spread_points
                )
                self._position = None
            reward = self._reward_fn(  # type: ignore[call-arg]
                0.0,
                daily_loss=self._acc.daily_loss,
                daily_loss_limit=self._guardrails.daily_loss_limit,
                initial_balance=self._initial_balance,
                breach=True,
            )
            self._idx += 1
            obs = self._get_obs()
            return obs, reward, True, False, {"breach": breach_reason}

        # Execute action
        direction = _ACTION_MAP[int(action)]
        if direction != 0:
            if self._position is not None and self._position.direction != direction:
                self._broker.close_position(  # type: ignore[union-attr]
                    self._acc, self._position, price, spread_points=spread_points
                )
                self._position = None
            if self._position is None:
                self._position = self._broker.open_position(  # type: ignore[union-attr]
                    self._acc, price, self._lots, direction, spread_points=spread_points
                )
        elif direction == 0 and self._position is not None:
            self._broker.close_position(  # type: ignore[union-attr]
                self._acc, self._position, price, spread_points=spread_points
            )
            self._position = None

        # Reward
        step_pnl = self._acc.equity - self._prev_equity
        self._prev_equity = self._acc.equity
        reward = self._reward_fn(  # type: ignore[call-arg]
            step_pnl,
            daily_loss=self._acc.daily_loss,
            daily_loss_limit=self._guardrails.daily_loss_limit,
            initial_balance=self._initial_balance,
        )

        self._idx += 1
        self._step_count += 1
        done = self._idx >= len(self._bars)
        truncated = (self._max_steps is not None) and (self._step_count >= self._max_steps)

        obs = self._get_obs() if not done else self._get_obs(last=True)
        info: dict[str, Any] = {
            "equity": self._acc.equity,
            "balance": self._acc.balance,
        }
        return obs, float(reward), done, truncated, info

    def _get_obs(self, last: bool = False) -> dict[str, np.ndarray]:
        if last:
            end = min(self._idx, len(self._features))
            start = max(0, end - self._obs_window)
            seq = self._features[start:end]
            if seq.shape[0] < self._obs_window:
                pad = np.zeros((self._obs_window - seq.shape[0], self._features.shape[1]), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)
        else:
            seq = self._features[self._idx - self._obs_window : self._idx]

        acc_arr = np.array(self._acc.to_array() if self._acc else [0.0] * 5, dtype=np.float32)
        return {"seq": seq, "account": acc_arr}

    def render(self) -> None:
        pass
