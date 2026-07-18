"""Gymnasium environment for RL-based trading with structure-aware SL/TP.

Wraps the backtest engine with a standard Gym interface. Actions are hybrid:
- Discrete: {hold=0, enter_long=1, enter_short=2, exit=3}
- Continuous (only when entering): risk_frac, rr_ratio

Observation: Dict space with time-series features (60-bar window) + account state.
Reward: Differential Sharpe Ratio (DSR) + small penalties for poor structure usage.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.account import AccountState
from ..backtest.broker import Broker, Position
from ..backtest.costs import COST_US100, CostModel
from ..backtest.guardrails import FTMOGuardrails
from ..backtest.risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short
from ..envs.reward import DSRReward


class TradingEnv(gym.Env[dict[str, np.ndarray[Any, Any]], int]):
    """RL trading environment with structure-aware SL/TP and hybrid action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        obs_window: int = 60,
        initial_balance: float = 100_000.0,
        cost_model: CostModel = COST_US100,
        broker_kwargs: dict[str, Any] | None = None,
        guardrail_kwargs: dict[str, Any] | None = None,
        risk_frac_range: tuple[float, float] = (0.005, 0.02),
        rr_ratio_range: tuple[float, float] = (1.0, 3.0),
        swing_buffer_pts: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        contract_size: float = 1.0,
        max_loss_per_trade_usd: float = 100.0,
        dsr_eta: float = 0.01,
        episodic: bool = True,
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
        episodic : bool
            If ``True`` (default, used for PPO training), a guardrail breach
            ends the episode (``done=True``) exactly as before. If ``False``
            (used for walk-forward evaluation/rollout), a breach force-closes
            any open position and blocks new trades for the rest of that
            ``session_id`` (calendar day), then trading resumes on the next
            session — mirroring ``quant_rl.backtest.engine.run_backtest`` — so
            a single call to ``reset()`` + repeated ``step()`` calls can walk
            the *entire* test set without terminating early.
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
        self.episodic = episodic

        self.reward_fn = DSRReward(eta=dsr_eta)

        # Action space: simplified discrete
        # 0=hold, 1-9=enter_long with risk/rr variants, 10-18=enter_short variants, 19=exit
        # This avoids Dict space which PPO doesn't support natively
        self.action_space = spaces.Discrete(20)

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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)

        self.step_idx = self.obs_window
        self.account = self._create_account()
        self.position: Position | None = None
        self.equity_curve = [self.initial_balance]
        self.pnl_history = [0.0]
        self.trade_log: list[dict[str, Any]] = []

        # Eval-mode (episodic=False) walk-forward bookkeeping. Harmless but
        # unused when episodic=True (training).
        self.prev_session: int | None = None
        self.breached_sessions: set[int] = set()
        self.sessions_with_trades: set[int] = set()
        self.all_sessions: set[int] = set()
        self.breach_log: list[str] = []
        self.breach_events: list[dict[str, Any]] = []

        obs = self._get_observation()
        return obs, {}

    def _create_account(self) -> AccountState:
        """Factory for fresh account state."""
        return AccountState(initial_balance=self.initial_balance)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Parameters
        ----------
        action : int
            0=hold, 1-9=enter_long variants, 10-18=enter_short variants, 19=exit
        """
        if self.step_idx >= len(self.bars):
            done = True
            truncated = True
            return self._get_observation(), 0.0, done, truncated, {}

        bar = self.bars.iloc[self.step_idx]
        feat_row = self.features.iloc[self.step_idx]
        bar_time = self.bars.index[self.step_idx]
        session_id = int(bar["session_id"]) if "session_id" in bar.index else 0

        if not self.episodic:
            self.all_sessions.add(session_id)
            if session_id != self.prev_session:
                self.account.reset_daily()
                self.prev_session = session_id

        # Mark-to-market
        bid, ask = self._bar_quote(bar)
        if self.position is not None:
            self.broker.mark_to_market(self.account, self.position, (bid, ask))

        # Fill quote for next action
        if self.step_idx + 1 < len(self.bars):
            next_bar = self.bars.iloc[self.step_idx + 1]
            fill_bid, fill_ask = self._bar_quote(next_bar)
        else:
            fill_bid, fill_ask = bid, ask

        # Decode discrete action (0-19)
        # 0: hold
        # 1-9: enter_long with different risk/rr (3x3 grid)
        # 10-18: enter_short with different risk/rr (3x3 grid)
        # 19: exit
        discrete_action = 0  # default hold
        risk_frac = self.risk_frac_range[0]  # default
        rr_ratio = self.rr_ratio_range[0]  # default

        if action == 0:
            discrete_action = 0  # hold
        elif 1 <= action <= 9:
            discrete_action = 1  # enter_long
            # Map to risk/rr: low/med/high × low/med/high
            idx = action - 1
            risk_variant = idx // 3  # 0, 1, 2
            rr_variant = idx % 3  # 0, 1, 2
            risk_levels = [
                self.risk_frac_range[0],
                (self.risk_frac_range[0] + self.risk_frac_range[1]) / 2,
                self.risk_frac_range[1],
            ]
            rr_levels = [
                self.rr_ratio_range[0],
                (self.rr_ratio_range[0] + self.rr_ratio_range[1]) / 2,
                self.rr_ratio_range[1],
            ]
            risk_frac = risk_levels[risk_variant]
            rr_ratio = rr_levels[rr_variant]
        elif 10 <= action <= 18:
            discrete_action = -1  # enter_short
            idx = action - 10
            risk_variant = idx // 3
            rr_variant = idx % 3
            risk_levels = [
                self.risk_frac_range[0],
                (self.risk_frac_range[0] + self.risk_frac_range[1]) / 2,
                self.risk_frac_range[1],
            ]
            rr_levels = [
                self.rr_ratio_range[0],
                (self.rr_ratio_range[0] + self.rr_ratio_range[1]) / 2,
                self.rr_ratio_range[1],
            ]
            risk_frac = risk_levels[risk_variant]
            rr_ratio = rr_levels[rr_variant]
        else:  # action == 19
            discrete_action = 0  # exit action mapped to hold, exit handled below

        # Check guardrails
        if self.episodic:
            reason = self.guardrails.breach_reason(self.account)
            session_blocked = False
        else:
            # Eval mode: a breach blocks new trading for the rest of this
            # session (calendar day) instead of ending the whole rollout —
            # mirrors run_backtest's `breached_sessions` handling so a fresh
            # breach is recorded/force-closed exactly once per session.
            session_blocked = session_id in self.breached_sessions
            reason = None if session_blocked else self.guardrails.breach_reason(self.account)
            if reason:
                self.breached_sessions.add(session_id)
                session_blocked = True

        if reason:
            if self.position is not None:
                pnl, fill_price = self.broker.close_position(
                    self.account, self.position, (fill_bid, fill_ask)
                )
                self.trade_log.append(
                    {
                        "type": "forced_close",
                        "pnl": pnl,
                        "price": fill_price,
                        "reason": reason,
                        "bar": self.step_idx,
                        "time": bar_time,
                        "equity": self.account.equity,
                    }
                )
                self.position = None
                self.sessions_with_trades.add(session_id)
            if not self.episodic:
                self.breach_log.append(reason)
                self.breach_events.append(
                    {
                        "time": bar_time,
                        "session_id": session_id,
                        "reason": reason,
                        "equity": self.account.equity,
                    }
                )
            done = self.episodic
            truncated = self.episodic
        elif session_blocked:
            # Already breached earlier today (eval mode only): no new
            # trading until the next session, but keep the rollout going.
            done = False
            truncated = False
        else:
            done = False
            truncated = False

            # Check SL/TP hits
            if self.position is not None:
                sl_hit = False
                if self.position.sl_price is not None:
                    if self.position.direction == 1 and float(bar["low"]) <= self.position.sl_price:
                        sl_hit = True
                    elif (
                        self.position.direction == -1
                        and float(bar["high"]) >= self.position.sl_price
                    ):
                        sl_hit = True

                if sl_hit:
                    pnl, fill_price = self.broker.close_position(
                        self.account, self.position, (fill_bid, fill_ask)
                    )
                    self.trade_log.append(
                        {
                            "type": "stop_close",
                            "pnl": pnl,
                            "price": fill_price,
                            "reason": "structure_sl",
                            "bar": self.step_idx,
                            "time": bar_time,
                            "equity": self.account.equity,
                        }
                    )
                    self.position = None
                    self.sessions_with_trades.add(session_id)
                elif self.position.tp_price is not None:
                    tp_hit = False
                    if (
                        self.position.direction == 1
                        and float(bar["high"]) >= self.position.tp_price
                    ):
                        tp_hit = True
                    elif (
                        self.position.direction == -1
                        and float(bar["low"]) <= self.position.tp_price
                    ):
                        tp_hit = True

                    if tp_hit:
                        pnl, fill_price = self.broker.close_position(
                            self.account, self.position, (fill_bid, fill_ask)
                        )
                        self.trade_log.append(
                            {
                                "type": "tp_close",
                                "pnl": pnl,
                                "price": fill_price,
                                "reason": "structure_tp",
                                "bar": self.step_idx,
                                "time": bar_time,
                                "equity": self.account.equity,
                            }
                        )
                        self.position = None
                        self.sessions_with_trades.add(session_id)

            # Action handling
            if not done:
                if action == 19:  # exit action
                    if self.position is not None:
                        pnl, fill_price = self.broker.close_position(
                            self.account, self.position, (fill_bid, fill_ask)
                        )
                        self.trade_log.append(
                            {
                                "type": "close",
                                "pnl": pnl,
                                "price": fill_price,
                                "bar": self.step_idx,
                                "time": bar_time,
                                "equity": self.account.equity,
                            }
                        )
                        self.position = None
                        self.sessions_with_trades.add(session_id)
                elif discrete_action != 0:  # enter_long or enter_short
                    if self.position is not None and self.position.direction != discrete_action:
                        pnl, fill_price = self.broker.close_position(
                            self.account, self.position, (fill_bid, fill_ask)
                        )
                        self.trade_log.append(
                            {
                                "type": "close",
                                "pnl": pnl,
                                "price": fill_price,
                                "bar": self.step_idx,
                                "time": bar_time,
                                "equity": self.account.equity,
                            }
                        )
                        self.position = None
                        self.sessions_with_trades.add(session_id)

                    if self.position is None and discrete_action in [1, -1]:
                        # Try to compute structure SL/TP
                        sl_price = None
                        tp_price = None

                        last_swing_low = (
                            float(feat_row["last_swing_low"])
                            if "last_swing_low" in feat_row.index
                            and pd.notna(feat_row["last_swing_low"])
                            else np.nan
                        )
                        last_swing_high = (
                            float(feat_row["last_swing_high"])
                            if "last_swing_high" in feat_row.index
                            and pd.notna(feat_row["last_swing_high"])
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
                                    "risk_frac": risk_frac,
                                    "rr_ratio": rr_ratio,
                                    "bar": self.step_idx,
                                    "time": bar_time,
                                    "equity": self.account.equity,
                                }
                            )
                            self.sessions_with_trades.add(session_id)

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
        """Get (bid, ask) from a bar using cost model.

        The bar's raw ``spread`` column is in MT5 broker points, not price
        units, so it must be scaled by ``point_size`` before being used as a
        price-unit spread (see ``quant_rl.backtest.engine._bar_spread_price_units``).
        """
        if "spread" in bar.index and pd.notna(bar["spread"]):
            bar_spread = float(bar["spread"]) * self.cost_model.point_size
        else:
            bar_spread = None
        return self.cost_model.bar_quote(float(bar["close"]), bar_spread=bar_spread)

    def _get_observation(self) -> dict[str, np.ndarray[Any, Any]]:
        """Construct observation dict."""
        # Time-series features
        start_idx = max(0, self.step_idx - self.obs_window)
        seq = np.asarray(self.features.iloc[start_idx : self.step_idx].values, dtype=np.float32)
        seq = cast(np.ndarray[Any, Any], np.nan_to_num(seq, nan=0.0))
        # Pad if needed
        if len(seq) < self.obs_window:
            pad_width = ((self.obs_window - len(seq), 0), (0, 0))
            seq = cast(
                np.ndarray[Any, Any], np.pad(seq, pad_width, mode="constant", constant_values=0.0)
            )

        # Account state
        pos_dir = float(self.position.direction) if self.position is not None else 0.0
        open_pnl = float(self.account.open_pnl) if self.position is not None else 0.0
        unrealised_r = (open_pnl / self.account.equity * 100) if self.account.equity > 0 else 0.0
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
