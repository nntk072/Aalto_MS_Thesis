"""Gymnasium trading environment with hybrid PPO action space.

The environment encapsulates:
- State: sequential features (obs_window bars) + account state (position, unrealised PnL, risk)
- Action: discrete (hold/enter_long/enter_short/exit) + continuous (risk_frac, rr_ratio on entry)
- Observation: dict with 'seq' (feature array) + scalar account metrics
- Reward: DSR (Differential Sharpe Ratio) per step, penalties for FTMO breaches
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.engine import run_backtest
from ..backtest.risk import compute_sl_price, compute_tp_price, compute_lots
from ..envs.reward import DSRReward


class TradingEnv(gym.Env):
    """Gymnasium environment for RL-based trading with structure-aware SL/TP.
    
    Observation space:
    - 'seq': feature array shape (obs_window, n_features)
    - 'equity': current equity
    - 'position': 0=flat, +1=long, -1=short
    - 'unrealised': unrealised PnL
    - 'risk_used': fraction of risk limit used
    
    Action space (hybrid):
    - action[0]: discrete action in {0=hold, 1=enter_long, 2=enter_short, 3=exit}
    - action[1]: continuous risk_frac ∈ [0.005, 0.02] (when entering)
    - action[2]: continuous rr_ratio ∈ [1.0, 3.0] (when entering)
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        obs_window: int = 60,
        initial_balance: float = 100_000.0,
        max_loss_per_trade: float = 100.0,
        contract_size: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
    ):
        """Initialize the trading environment.
        
        Parameters
        ----------
        bars : pd.DataFrame
            M1 price bars with high, low, close.
        features : pd.DataFrame
            Feature matrix (same length as bars).
        obs_window : int
            Number of past bars to include in observation.
        initial_balance : float
            Starting account equity.
        max_loss_per_trade : float
            Hard USD cap per trade.
        contract_size : float
            Multiplier for lot sizing.
        min_lot, max_lot : float
            Lot size bounds.
        """
        super().__init__()
        self.bars = bars
        self.features = features
        self.obs_window = obs_window
        self.initial_balance = initial_balance
        self.max_loss_per_trade = max_loss_per_trade
        self.contract_size = contract_size
        self.min_lot = min_lot
        self.max_lot = max_lot

        n_bars = len(bars)
        n_features = features.shape[1] if features.shape[1] > 0 else 1

        # Observation space: dict with seq array and scalars
        self.observation_space = spaces.Dict({
            "seq": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_window, n_features),
                dtype=np.float32,
            ),
            "equity": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "unrealised": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "risk_used": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

        # Action space: hybrid discrete + continuous
        # action[0]: 0=hold, 1=enter_long, 2=enter_short, 3=exit
        # action[1]: risk_frac (ignored if not entering)
        # action[2]: rr_ratio (ignored if not entering)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.005, 1.0], dtype=np.float32),
            high=np.array([3.0, 0.02, 3.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reward_fn = DSRReward()
        self.step_count = 0
        self.current_bar_idx = obs_window
        self.equity = initial_balance
        self.position = None
        self.position_entry = None
        self.unrealised_pnl = 0.0

    def reset(self, seed: int | None = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.step_count = 0
        self.current_bar_idx = self.obs_window
        self.equity = self.initial_balance
        self.position = None
        self.position_entry = None
        self.unrealised_pnl = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> dict[str, Any]:
        """Get current observation."""
        feat_seq = self.features.iloc[
            self.current_bar_idx - self.obs_window : self.current_bar_idx
        ].values.astype(np.float32)
        
        pos_value = 0.0 if self.position is None else float(self.position.direction)
        
        return {
            "seq": feat_seq,
            "equity": np.array([self.equity], dtype=np.float32),
            "position": np.array([pos_value], dtype=np.int32),
            "unrealised": np.array([self.unrealised_pnl], dtype=np.float32),
            "risk_used": np.array([min(abs(self.unrealised_pnl) / self.max_loss_per_trade, 1.0)], dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> tuple:
        """Execute one step of the environment.
        
        Parameters
        ----------
        action : np.ndarray
            [discrete_action, risk_frac, rr_ratio]
            discrete_action: 0=hold, 1=enter_long, 2=enter_short, 3=exit
        
        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self.current_bar_idx >= len(self.bars):
            return self._get_obs(), 0.0, True, False, {}

        # Parse action
        discrete_act = int(np.clip(action[0], 0, 3))
        risk_frac = float(np.clip(action[1], 0.005, 0.02))
        rr_ratio = float(np.clip(action[2], 1.0, 3.0))

        bar = self.bars.iloc[self.current_bar_idx]
        current_price = float(bar["close"])
        
        # Update unrealised PnL if in position
        if self.position is not None:
            if self.position.direction == 1:
                self.unrealised_pnl = (current_price - self.position_entry) * self.position.size * self.contract_size
            else:
                self.unrealised_pnl = (self.position_entry - current_price) * self.position.size * self.contract_size
        
        # Execute action
        action_pnl = 0.0
        
        if discrete_act == 1 and self.position is None:  # Enter long
            sl_price = compute_sl_price(
                current_price, 1,
                float(self.features.iloc[self.current_bar_idx].get("last_swing_low_price", np.nan)),
                float(self.features.iloc[self.current_bar_idx].get("last_swing_high_price", np.nan)),
            )
            lots = compute_lots(self.equity, risk_frac, abs(current_price - sl_price), self.contract_size, self.max_loss_per_trade)
            self.position = type('Pos', (), {'direction': 1, 'size': lots, 'sl_price': sl_price})()
            self.position_entry = current_price
            
        elif discrete_act == 2 and self.position is None:  # Enter short
            sl_price = compute_sl_price(
                current_price, -1,
                float(self.features.iloc[self.current_bar_idx].get("last_swing_low_price", np.nan)),
                float(self.features.iloc[self.current_bar_idx].get("last_swing_high_price", np.nan)),
            )
            lots = compute_lots(self.equity, risk_frac, abs(current_price - sl_price), self.contract_size, self.max_loss_per_trade)
            self.position = type('Pos', (), {'direction': -1, 'size': lots, 'sl_price': sl_price})()
            self.position_entry = current_price
            
        elif discrete_act == 3 and self.position is not None:  # Exit
            action_pnl = self.unrealised_pnl
            self.equity += action_pnl
            self.position = None
            self.position_entry = None
            self.unrealised_pnl = 0.0

        # Compute reward (simple: realised PnL)
        reward = action_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        
        # Check termination (max loss or end of data)
        done = (self.unrealised_pnl <= -self.max_loss_per_trade) or (self.current_bar_idx >= len(self.bars) - 1)
        
        self.step_count += 1
        self.current_bar_idx += 1

        return self._get_obs(), reward, done, False, {}

    def render(self, mode: str = "human") -> None:
        """Render environment state (placeholder)."""
        pass
