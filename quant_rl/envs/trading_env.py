"""Gymnasium environment for RL-based trading with structure-aware SL/TP.

Wraps the backtest engine with a standard Gym interface. Actions are hybrid:
- Discrete: {hold=0, enter_long=1, enter_short=2, exit=3}
- Continuous (only when entering): risk_frac, rr_ratio

Observation: Dict space with time-series features (60-bar window) + account state.
Reward: Differential Sharpe Ratio (DSR) + small penalties for poor structure usage.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.broker import Broker, Position
from ..backtest.costs import CostModel, COST_US100
from ..backtest.guardrails import FTMOGuardrails
from ..backtest.risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short
from ..envs.reward import DSRReward


class TradingEnv(gym.Env):
    """RL trading environment with structure-aware SL/TP and hybrid action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        obs_window: int = 60,
        initial_balance: float = 100_000.0,
        cost_model: CostModel = COST_US100,
        broker_kwargs: dict | None = None,
        guardrail_kwargs: dict | None = None,
        risk_frac_range: tuple[float, float] = (0.005, 0.02),
        rr_ratio_range: tuple[float, float] = (1.0, 3.0),
        swing_buffer_pts: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        contract_size: float = 1.0,
        max_loss_per_trade_usd: float = 100.0,
        dsr_eta: float = 0.01,
    ):
        """Initialize trading environment.

        Parameters
        ----------
        bars : pd.DataFrame
            OHLC price data with DatetimeIndex.
        features : pd.DataFrame
            Feature matrix aligned to bars, including structure features.
        obs_window : int
            Number of bars to include in observation window.
        initial_balance : float
            Starting account balance.
        cost_model : CostModel
            Broker cost model for fills.
        broker_kwargs : dict | None
            Kwargs for Broker initialization.
        guardrail_kwargs : dict | None
            Kwargs for FTMOGuardrails initialization.
        risk_frac_range : tuple[float, float]
            Min/max risk fraction for continuous action normalization.
        rr_ratio_range : tuple[float, float]
            Min/max R:R ratio for continuous action normalization.
        swing_buffer_pts : float
            Buffer in price points for SL placement beyond swing.
        min_lot, max_lot : float
            Lot size bounds.
        contract_size : float
            Contract multiplier.
        max_loss_per_trade_usd : float
            Safety cap on per-trade loss.
        dsr_eta : float
            Differential Sharpe Ratio damping factor.
        """
        self.bars = bars
        self.features = features
        self.obs_window = obs_window
        self.initial_balance = initial_balance
        self.cost_model = cost_model
        self.broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
        self.guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))

        self.risk_frac_range = risk_frac_range
        self.rr_ratio_range = rr_ratio_range
        self.swing_buffer_pts = swing_buffer_pts
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.contract_size = contract_size
        self.max_loss_per_trade_usd = max_loss_per_trade_usd

        self.reward_fn = DSRReward(eta=dsr_eta)

        # Action space: hybrid discrete + continuous
        # discrete: 0=hold, 1=enter_long, 2=enter_short, 3=exit
        # continuous (when entering): [risk_frac, rr_ratio]
        self.action_space = spaces.Dict(
            {
                "discrete": spaces.Discrete(4),
                "continuous": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        # Observation space: dict with time-series + account state
        # features: (obs_window, n_features)
        # account: [equity, position_direction, open_pnl, unrealised_r, dist_to_sl]
        n_features = features.shape[1] if len(features) > 0 else 1
        self.observation_space = spaces.Dict(
            {
                "seq": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_window, n_features),
                    dtype=np.float32,
                ),
                "account": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)

        self.step_idx = self.obs_window
        self.account = self._create_account()
        self.position: Position | None = None
        self.equity_curve = [self.initial_balance]
        self.pnl_history = [0.0]
        self.trade_log = []

        obs = self._get_observation()
        return obs, {}

    def _create_account(self):
        """Factory for fresh account state."""
        from ..backtest.account import AccountState

        return AccountState(initial_balance=self.initial_balance)

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action : dict
            {"discrete": int, "continuous": np.ndarray}
            discrete: 0=hold, 1=enter_long, 2=enter_short, 3=exit
            continuous: [risk_frac_normalized, rr_ratio_normalized] in [0, 1]
        """
        if self.step_idx >= len(self.bars):
            done = True
            truncated = True
            return self._get_observation(), 0.0, done, truncated, {}

        bar = self.bars.iloc[self.step_idx]
        feat_row = self.features.iloc[self.step_idx]
        bar_time = self.bars.index[self.step_idx]

        # Mark-to-market
        bid, ask = self._bar_quote(bar)
        if self.position is not None:
            self.broker.mark_to_market(self.account, self.position, (bid, ask))

        # Fill quote for next action
        if self.step_idx + 1 < len(self.bars):
            fill_instant = self.bars.index[self.step_idx + 1]
            next_bar = self.bars.iloc[self.step_idx + 1]
            fill_bid, fill_ask = self._bar_quote(next_bar)
        else:
            fill_bid, fill_ask = bid, ask

        # Parse action
        discrete_action = int(action["discrete"])
        risk_frac_norm, rr_ratio_norm = action["continuous"]

        # Denormalize continuous parameters
        risk_frac = self.risk_frac_range[0] + risk_frac_norm * (
            self.risk_frac_range[1] - self.risk_frac_range[0]
        )
        rr_ratio = self.rr_ratio_range[0] + rr_ratio_norm * (
            self.rr_ratio_range[1] - self.rr_ratio_range[0]
        )

        # Check guardrails
        reason = self.guardrails.breach_reason(self.account)
        if reason:
            if self.position is not None:
                pnl = self.broker.close_position(self.account, self.position, (fill_bid, fill_ask))
                self.trade_log.append(
                    {"type": "forced_close", "pnl": pnl, "reason": reason, "time": bar_time}
                )
                self.position = None
            done = True
            truncated = True
        else:
            done = False
            truncated = False

            # Check SL/TP hits
            if self.position is not None:
                sl_hit = False
                if self.position.sl_price is not None:
                    if self.position.direction == 1 and float(bar["low"]) <= self.position.sl_price:
                        sl_hit = True
                    elif self.position.direction == -1 and float(bar["high"]) >= self.position.sl_price:
                        sl_hit = True

                if sl_hit:
                    pnl = self.broker.close_position(self.account, self.position, (fill_bid, fill_ask))
                    self.trade_log.append(
                        {"type": "stop_close", "pnl": pnl, "reason": "structure_sl", "time": bar_time}
                    )
                    self.position = None
                elif self.position.tp_price is not None:
                    tp_hit = False
                    if self.position.direction == 1 and float(bar["high"]) >= self.position.tp_price:
                        tp_hit = True
                    elif self.position.direction == -1 and float(bar["low"]) <= self.position.tp_price:
                        tp_hit = True

                    if tp_hit:
                        pnl = self.broker.close_position(self.account, self.position, (fill_bid, fill_ask))
                        self.trade_log.append(
                            {
                                "type": "tp_close",
                                "pnl": pnl,
                                "reason": "structure_tp",
                                "time": bar_time,
                            }
                        )
                        self.position = None

            # Action handling
            if not done:
                if discrete_action != 0:
                    if self.position is not None and self.position.direction != discrete_action:
                        pnl = self.broker.close_position(self.account, self.position, (fill_bid, fill_ask))
                        self.trade_log.append({"type": "close", "pnl": pnl, "time": bar_time})
                        self.position = None

                    if self.position is None and discrete_action in [1, -1]:
                        # Try to compute structure SL/TP
                        sl_price = None
                        tp_price = None

                        last_swing_low = (
                            float(feat_row["last_swing_low"])
                            if "last_swing_low" in feat_row.index and pd.notna(feat_row["last_swing_low"])
                            else np.nan
                        )
                        last_swing_high = (
                            float(feat_row["last_swing_high"])
                            if "last_swing_high" in feat_row.index and pd.notna(feat_row["last_swing_high"])
                            else np.nan
                        )

                        entry_price = float(fill_ask if discrete_action == 1 else fill_bid)

                        if discrete_action == 1 and not np.isnan(last_swing_low):
                            sl_price, tp_price = compute_sl_tp_long(
                                entry_price,
                                last_swing_low,
                                buffer_pts=self.swing_buffer_pts,
                                rr_ratio=rr_ratio,
                            )
                            lots = compute_lots(
                                self.account.equity,
                                risk_frac,
                                entry_price,
                                sl_price,
                                contract_size=self.contract_size,
                                min_lot=self.min_lot,
                                max_lot=self.max_lot,
                                max_loss_cap=self.max_loss_per_trade_usd,
                            )
                        elif discrete_action == -1 and not np.isnan(last_swing_high):
                            sl_price, tp_price = compute_sl_tp_short(
                                entry_price,
                                last_swing_high,
                                buffer_pts=self.swing_buffer_pts,
                                rr_ratio=rr_ratio,
                            )
                            lots = compute_lots(
                                self.account.equity,
                                risk_frac,
                                entry_price,
                                sl_price,
                                contract_size=self.contract_size,
                                min_lot=self.min_lot,
                                max_lot=self.max_lot,
                                max_loss_cap=self.max_loss_per_trade_usd,
                            )
                        else:
                            lots = 1.0  # Fallback if no swings available

                        self.position = self.broker.open_position(
                            self.account, (fill_bid, fill_ask), lots, discrete_action
                        )
                        if self.position:
                            self.position.sl_price = sl_price
                            self.position.tp_price = tp_price
                            self.position.risk_frac = risk_frac
                            self.position.rr_ratio = rr_ratio
                            self.trade_log.append(
                                {
                                    "type": "open",
                                    "direction": discrete_action,
                                    "price": self.position.entry_price,
                                    "lots": self.position.size,
                                    "sl_price": sl_price,
                                    "tp_price": tp_price,
                                    "time": bar_time,
                                }
                            )

                elif discrete_action == 0 and self.position is not None:
                    # Exit action
                    pnl = self.broker.close_position(self.account, self.position, (fill_bid, fill_ask))
                    self.trade_log.append({"type": "close", "pnl": pnl, "time": bar_time})
                    self.position = None

        self.equity_curve.append(self.account.equity)
        pnl_step = self.account.equity - self.equity_curve[-2]
        self.pnl_history.append(pnl_step)

        # Compute reward using DSR
        daily_loss = self.initial_balance - self.account.equity
        reward = self.reward_fn(
            pnl_step,
            daily_loss=daily_loss,
            daily_loss_limit=self.guardrails.daily_loss_limit,
            initial_balance=self.initial_balance,
            breach=done and truncated,
        )

        obs = self._get_observation()
        info = {"equity": self.account.equity, "position": self.position is not None}

        self.step_idx += 1

        return obs, float(reward), done, truncated, info

    def _bar_quote(self, bar: pd.Series) -> tuple[float, float]:
        """Get (bid, ask) from a bar using cost model."""
        bar_spread = float(bar["spread"]) if "spread" in bar.index else None
        return self.cost_model.bar_quote(float(bar["close"]), bar_spread=bar_spread)

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Construct observation dict."""
        # Time-series features
        start_idx = max(0, self.step_idx - self.obs_window)
        seq = self.features.iloc[start_idx : self.step_idx].values.astype(np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        # Pad if needed
        if len(seq) < self.obs_window:
            pad_width = ((self.obs_window - len(seq), 0), (0, 0))
            seq = np.pad(seq, pad_width, mode="constant", constant_values=0.0)

        # Account state
        pos_dir = float(self.position.direction) if self.position is not None else 0.0
        open_pnl = float(self.account.open_pnl) if self.position is not None else 0.0
        unrealised_r = (
            (open_pnl / self.account.equity * 100) if self.account.equity > 0 else 0.0
        )
        dist_to_sl = 0.0
        if self.position is not None and self.position.sl_price is not None:
            dist_to_sl = (
                self.position.entry_price - self.position.sl_price
                if self.position.direction == 1
                else self.position.sl_price - self.position.entry_price
            )

        account_state = np.array(
            [
                self.account.equity,
                pos_dir,
                open_pnl,
                unrealised_r,
                dist_to_sl,
            ],
            dtype=np.float32,
        )

        return {"seq": seq, "account": account_state}
