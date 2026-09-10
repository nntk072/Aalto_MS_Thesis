"""Gymnasium environment for RL-based trading with structure-aware SL/TP.

Wraps the backtest engine with a standard Gym interface. Actions support:
- Discrete: {hold=0, enter_long=1-9, enter_short=10-18, exit=19}
- Continuous: Box(-1, 1) for proportional position sizing

Observation: Dict space with time-series features (60-bar window) + account state.
Reward: Differential Sharpe Ratio (DSR) or Sweep Confirmation Reward.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.account import AccountState
from ..backtest.broker import Broker, Position
from ..backtest.costs import COST_US100, CostModel
from ..backtest.guardrails import FTMOGuardrails
from ..backtest.risk import (
    compute_lots,
    compute_sl_tp_from_structure,
    compute_sl_tp_long,
    compute_sl_tp_short,
    resolve_tp_target,
)
from ..envs.reward import DSRReward
from ..envs.strategies import BaselineStrategy, TradingStrategy
from ..envs.sweep_reward import CompositeReward, SweepConfirmationReward
from ..features.build import MTF_RAW_SUFFIXES
from ..models.vae import VAE


class TradingEnv(gym.Env[dict[str, np.ndarray[Any, Any]], int | np.ndarray[Any, Any]]):
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
        use_sweep_reward: bool = False,
        sweep_alpha: float = 0.1,
        sweep_beta: float = 0.01,
        sweep_hold_bars: int = 3,
        dsr_weight: float = 0.3,
        sweep_weight: float = 0.7,
        continuous_actions: bool = False,
        max_risk_frac: float = 0.01,
        use_vae: bool = False,
        vae: VAE | None = None,
        pre_ny_data: pd.DataFrame | None = None,
        ny_session_start_idx: int | None = None,
        fill_latency_bars: int = 0,
        max_episode_steps: int | None = None,
        normalize_account: bool = True,
        strategy: TradingStrategy | None = None,
        strategy_actions: bool = False,
        sl_buffer_pts: float = 0.0,
        strategy_reward: Any = None,
        strategy_weight: float = 0.0,
        block_overnight: bool = True,
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
        use_sweep_reward : bool
            If True, use SweepConfirmationReward instead of DSRReward.
        sweep_alpha : float
            Weight for sweep confirmation score C_t.
        sweep_beta : float
            Weight for time decay penalty T_t.
        sweep_hold_bars : int
            Number of bars price must hold beyond level for confirmation.
        dsr_weight : float
            Weight for DSR reward in composite.
        sweep_weight : float
            Weight for sweep reward in composite.
        continuous_actions : bool
            If True, use Box(-1, 1) continuous action space for position sizing.
        max_risk_frac : float
            Maximum risk fraction per trade when using continuous actions (default: 0.01 = 1%).
        use_vae : bool
            If True, use VAE to extract latent narrative embedding from pre-NY sequence.
        vae : VAE | None
            Pre-trained VAE model for narrative embedding. Required if use_vae=True.
        pre_ny_data : pd.DataFrame | None
            Pre-NY session data (01:05-16:29 UTC+3) for VAE input. Required if use_vae=True.
        ny_session_start_idx : int | None
            Index of the first NY session bar (16:30 UTC+3). Used to compute
            minutes_since_open for the sweep time-decay penalty. If None, the
            observation window length is used as the default.
        max_episode_steps : int | None
            Hard cap on episode length in environment steps. When the agent
            has taken this many steps, the episode ends with
            ``truncated=True`` (Gymnasium standard). ``None`` means no
            truncation, so episodes run to the end of the data. PPO's
            ``n_steps`` is the dominant rollout-length control; this is a
            safety net against infinite episodes on small datasets.
        normalize_account : bool
            If True (default), the ``account`` vector in the observation is
            rescaled so the per-element magnitudes are roughly comparable
            with the z-scored ``seq`` features:
              - equity:    log(equity / initial_balance)
              - pos_dir:   -1 / 0 / +1
              - open_pnl:  open_pnl / initial_balance
              - unrealised_r: unchanged (already a percentage)
              - dist_to_sl: dist_to_sl / current_bar_close
            Without this, the equity and pnl fields sit at ~1e5 while the
            z-scored seq features sit at ~1.0, dominating the policy's
            MLP heads. Disable with ``False`` to recover the raw values.
        strategy : TradingStrategy | None
            Strategy semantics (entry gates, structural SL reference,
            TP targets). Defaults to the no-op :class:`BaselineStrategy`,
            which preserves existing behaviour exactly.
        strategy_actions : bool
            If True, use the Idea 1/2 four-dimensional Box action
            ``[direction, risk, RR, TP-target]`` (Agent.md §12) and route
            entries through the strategy's gates / structural SL / TP-target
            resolver instead of the legacy discrete/1-D spaces and swing
            levels. Baseline (Idea 3) keeps ``False``.
        sl_buffer_pts : float
            Buffer in price points for the structural stop in strategy
            modes. ``0.0`` = ``sl_mode: exact``; positive = ``buffered``.
        block_overnight : bool
            If True (default), any position still open on the **last bar of
            a session** (``session_id`` changes on the next step) is
            force-closed at that bar's close quote — the position never
            carries across the overnight gap. Mirrors
            ``run_backtest``'s no-overnight contract (its ``eod_close``)
            and keeps evaluation/training free of gap risk. Set ``False``
            to restore gap-hold behaviour (mark-to-market across the
            session boundary; SL/TP checked on the next session's bars).
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
        self.block_overnight = bool(block_overnight)
        self.episodic = episodic
        self.use_sweep_reward = use_sweep_reward

        # Initialize reward function
        self.reward_fn: DSRReward | CompositeReward
        if use_sweep_reward or (strategy_actions and strategy_reward is not None):
            sweep_reward = SweepConfirmationReward(
                alpha=sweep_alpha, beta=sweep_beta, hold_bars=sweep_hold_bars
            )
            self.reward_fn = CompositeReward(
                sweep_reward=sweep_reward,
                dsr_weight=dsr_weight,
                sweep_weight=sweep_weight,
                strategy_reward=strategy_reward,
                strategy_weight=strategy_weight,
            )
            self.dsr_reward = DSRReward(eta=dsr_eta)  # Keep for composite
        else:
            self.reward_fn = DSRReward(eta=dsr_eta)

        # Entry-gate: warn once (not per-step) if required features are missing.
        self._entry_gate_warned: bool = False

        # Strategy semantics (Idea 1/2) vs. legacy behaviour (Idea 3 baseline).
        self.strategy: TradingStrategy = strategy if strategy is not None else BaselineStrategy()
        self.strategy_actions = bool(strategy_actions)
        self.sl_buffer_pts = float(sl_buffer_pts)
        self._selected_tp_mode = "rr"
        if self.strategy_actions:
            missing = [c for c in self.strategy.required_features if c not in features.columns]
            if missing:
                raise ValueError(f"Missing strategy features: {missing}")

        # Model-facing observation frame: the strategy's raw price-level
        # columns stay in ``features`` for execution/risk but must not enter
        # the normalised ``seq`` tensor (Agent.md §11). MTF blocks reuse the
        # same stems with a ``{TF}_`` prefix (e.g. ``M5_last_swing_high``),
        # matched via MTF_RAW_SUFFIXES so unnormalised HTF price magnitudes
        # never reach the encoder.
        drop_cols = [c for c in self.strategy.raw_columns if c in features.columns]
        mtf_raw = {
            c
            for c in features.columns
            if any(str(c).endswith(f"_{stem}") for stem in MTF_RAW_SUFFIXES)
            and str(c) not in self.strategy.raw_columns
        }
        self._obs_features = features.drop(columns=drop_cols + sorted(mtf_raw))

        # Action space: strategy actions, continuous, or discrete
        self.continuous_actions = continuous_actions
        self.max_risk_frac = max_risk_frac

        # VAE for narrative embedding
        self.use_vae = use_vae
        self.vae = vae
        self.pre_ny_data = pre_ny_data
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

        if use_vae:
            if vae is None:
                raise ValueError("vae must be provided when use_vae=True")
            if pre_ny_data is None:
                raise ValueError("pre_ny_data must be provided when use_vae=True")
            # Freeze VAE encoder
            for param in vae.parameters():
                param.requires_grad = False

        if self.strategy_actions:
            # 4-D strategy action (Agent.md §12):
            #   action[0] in [-1, 1]: direction/entry intensity (|x| < 0.25 = hold)
            #   action[1] in [0, 1]: risk selector -> risk_frac_range
            #   action[2] in [0, 1]: RR selector -> rr_ratio_range
            #   action[3] in [0, 1]: TP-target selector -> discrete modes
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif continuous_actions:
            # Continuous action space: Box(-1, 1) for position sizing
            # -1.0 = max short, +1.0 = max long, 0 = hold
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # Discrete action space: 0=hold, 1-9=enter_long, 10-18=enter_short, 19=exit
            self.action_space = spaces.Discrete(20)

        # NY session start index for time decay penalty
        self.ny_session_start_idx = (
            ny_session_start_idx if ny_session_start_idx is not None else self.obs_window
        )

        # Observation space: dict with time-series + account state (+ VAE latent if enabled)
        # features: (obs_window, n_features)
        # account: [equity, position_direction, open_pnl, unrealised_r, dist_to_sl]
        # vae_z: latent embedding from VAE (if use_vae=True)
        n_features = self._obs_features.shape[1] if len(self._obs_features) > 0 else 1
        vae_latent_dim = vae.encoder.latent_dim if use_vae and vae is not None else 0

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
        # Add VAE latent to observation space if enabled
        if use_vae and vae_latent_dim > 0:
            self.observation_space.spaces["vae_z"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vae_latent_dim,),
                dtype=np.float32,
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
        # Step counter within the current episode (0 at reset). Distinct
        # from ``step_idx`` (the absolute bar index); used to enforce
        # ``max_episode_steps`` truncation.
        self.episode_step_count = 0
        # Warn-once flag for the entry gate; re-arm per episode so a new
        # feature matrix (possibly missing gate columns) warns again.
        self._entry_gate_warned = False

        # Reset reward function
        self.reward_fn.reset()

        # Eval-mode (episodic=False) walk-forward bookkeeping. Harmless but
        # unused when episodic=True (training).
        self.prev_session: int | None = None
        self.breached_sessions: set[int] = set()
        self.sessions_with_trades: set[int] = set()
        self.all_sessions: set[int] = set()
        self.breach_log: list[str] = []
        self.breach_events: list[dict[str, Any]] = []
        self._level_crosses: dict[tuple[int, str], Any] = {}

        obs = self._get_observation()
        return obs, {}

    def _create_account(self) -> AccountState:
        """Factory for fresh account state."""
        return AccountState(initial_balance=self.initial_balance)

    def _check_entry_gate(
        self,
        price: float,
        discrete_action: int,
        feat_row: pd.Series,
    ) -> bool:
        """Check if action is allowed based on multi-liquidity entry gate.

        Entry Gate Rules:
        - Long: (price > LondonHigh OR price > AsianHigh) AND volume_spike > 1.5
        - Short: (price < LondonLow OR price < AsianLow) AND volume_spike > 1.5
        - Exit/Hold: Always allowed

        Parameters
        ----------
        price : float
            Current price
        discrete_action : int
            -1 = short, 0 = hold/exit, 1 = long
        feat_row : pd.Series
            Feature row with liquidity levels and volume_spike

        Returns
        -------
        bool
            True if action is allowed, False otherwise
        """
        if discrete_action == 0:  # Hold or exit
            return True

        # Check if we have the required features
        if not all(
            col in feat_row.index
            for col in ["london_high", "london_low", "asian_high", "asian_low", "volume_spike"]
        ):
            # Features missing: the gate cannot evaluate, so it would silently
            # allow every entry. Warn once rather than spamming every step.
            if not self._entry_gate_warned:
                warnings.warn(
                    "Entry-gate features (london/asian levels, volume_spike) missing "
                    "from feature matrix — gate falls back to allowing all entries. "
                    "Check that session-liquidity and volume-spike features are enabled.",
                    stacklevel=2,
                )
                self._entry_gate_warned = True
            return True  # Fallback: allow if features missing

        london_high = float(feat_row["london_high"])
        london_low = float(feat_row["london_low"])
        asian_high = float(feat_row["asian_high"])
        asian_low = float(feat_row["asian_low"])
        volume_spike = float(feat_row["volume_spike"])

        if discrete_action == 1:  # Long
            # Long allowed if price > LondonHigh OR price > AsianHigh AND volume_spike > 1.5
            return (price > london_high or price > asian_high) and volume_spike > 1.5
        elif discrete_action == -1:  # Short
            # Short allowed if price < LondonLow OR price < AsianLow AND volume_spike > 1.5
            return (price < london_low or price < asian_low) and volume_spike > 1.5

        return True

    def _decode_action(
        self, action: int | float | np.ndarray[Any, Any]
    ) -> tuple[int, float, float, str]:
        """Decode an action into (discrete_action, risk_frac, rr_ratio, tp_mode).

        Handles three action formats:
        - Strategy (4-D Box): [direction, risk, rr, tp_mode]
        - Continuous (Box(-1,1)): proportional position sizing
        - Discrete (Discrete(20)): 0=hold, 1-9=long variants, 10-18=short variants, 19=exit
        """
        tp_mode = "rr"
        risk_frac = self.risk_frac_range[0]
        rr_ratio = self.rr_ratio_range[0]
        if self.strategy_actions:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            direction_val = float(arr[0]) if arr.size else 0.0
            entry_threshold = 0.25
            if direction_val > entry_threshold:
                discrete_action = 1
            elif direction_val < -entry_threshold:
                discrete_action = -1
            else:
                discrete_action = 0
            if arr.size >= 4:
                r_lo, r_hi = self.risk_frac_range
                rr_lo, rr_hi = self.rr_ratio_range
                risk_frac = r_lo + float(arr[1]) * (r_hi - r_lo)
                rr_ratio = rr_lo + float(arr[2]) * (rr_hi - rr_lo)
                tp_modes = [
                    "rr",
                    "buyside_liquidity",
                    "sellside_liquidity",
                    "previous_day_high_low",
                ]
                tp_idx = min(int(round(float(arr[3]) * (len(tp_modes) - 1))), len(tp_modes) - 1)
                tp_mode = tp_modes[tp_idx]
            else:
                risk_frac = self.risk_frac_range[0]
                rr_ratio = self.rr_ratio_range[0]
        elif self.continuous_actions:
            if isinstance(action, np.ndarray):
                action_value = float(action[0]) if action.size > 0 else 0.0
            else:
                action_value = float(action)
            if action_value > 0:
                discrete_action = 1
                risk_frac = self.max_risk_frac * action_value
            elif action_value < 0:
                discrete_action = -1
                risk_frac = self.max_risk_frac * abs(action_value)
            else:
                discrete_action = 0
                risk_frac = 0.0
            rr_ratio = self.rr_ratio_range[0]
        else:
            # Discrete actions
            if action == 0:
                discrete_action = 0
            elif 1 <= action <= 9:
                discrete_action = 1
                idx = int(action) - 1
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
            elif 10 <= action <= 18:
                discrete_action = -1
                idx = int(action) - 10
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
                discrete_action = 0
                risk_frac = self.risk_frac_range[0]
                rr_ratio = self.rr_ratio_range[0]
        return discrete_action, risk_frac, rr_ratio, tp_mode

    def step(
        self,
        action: int | float | np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Parameters
        ----------
        action : int or np.ndarray
            Discrete: 0=hold, 1-9=enter_long variants, 10-18=enter_short variants, 19=exit
            Continuous: Box(-1, 1) for proportional position sizing
        """
        if self.step_idx >= len(self.bars):
            done = True
            truncated = True
            return self._get_observation(), 0.0, done, truncated, {}

        bar = self.bars.iloc[self.step_idx]
        feat_row = self.features.iloc[self.step_idx]
        bar_time = self.bars.index[self.step_idx]
        session_id = int(bar["session_id"]) if "session_id" in bar.index else 0

        # Fill quote for next action (latency-shifted: decision at bar t
        # fills at the quote of bar t + fill_latency_bars, not t + 1)
        fill_idx = self.step_idx + 1 + self.fill_latency_bars
        fill_bid, fill_ask, next_bar = self._progress_market(
            bar, feat_row, bar_time, session_id, fill_idx
        )

        # Decode action (discrete or continuous)
        discrete_action, risk_frac, rr_ratio, tp_mode = self._decode_action(action)
        self._selected_tp_mode = tp_mode

        # Check entry gate for new positions
        # Suppress new entries on the last bar of a session (overnight block
        # just closed a position or would have if there was one).
        entering_new_session = (
            self.block_overnight
            and next_bar is not None
            and "session_id" in next_bar.index
            and int(next_bar["session_id"]) != session_id
        )
        if entering_new_session:
            discrete_action = 0

        # Check entry gate for new positions
        if discrete_action != 0:  # Only check for long/short entries
            if self.strategy_actions:
                gate_ok = self.strategy.validate_entry(direction=discrete_action, row=feat_row)
            else:
                gate_ok = self._check_entry_gate(float(bar["close"]), discrete_action, feat_row)
            if not gate_ok:
                # Gate not satisfied, force to hold
                discrete_action = 0
                risk_frac = 0.0

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
                if not self.strategy_actions and action == 19:  # exit action
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

                        # Track whether structural SL/TP levels were available
                        # AND geometrically valid so we never open a naked
                        # position (no SL/TP) as a fallback and never crash.
                        has_levels = False
                        if self.strategy_actions:
                            # Structural SL from the strategy object
                            # (manipulation extreme for Idea 1, swept level for
                            # Idea 2 — Agent.md §14), TP from the resolver
                            # (Agent.md §15). Never from the normalised obs.
                            sl_ref = self.strategy.sl_reference(
                                direction=discrete_action, row=feat_row
                            )
                            if sl_ref is not None and np.isfinite(float(sl_ref)):
                                try:
                                    sl_price, _tp_rr = compute_sl_tp_from_structure(
                                        direction=discrete_action,
                                        entry_price=entry_price,
                                        structure_level=float(sl_ref),
                                        rr_ratio=rr_ratio,
                                        buffer_pts=self.sl_buffer_pts,
                                    )
                                    tp_price = resolve_tp_target(
                                        direction=discrete_action,
                                        entry_price=entry_price,
                                        sl_price=sl_price,
                                        rr_ratio=rr_ratio,
                                        target_mode=getattr(self, "_selected_tp_mode", "rr"),
                                        row=feat_row,
                                    )
                                except ValueError:
                                    sl_price, tp_price = None, None
                            if sl_price is not None and tp_price is not None:
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
                                has_levels = True
                            else:
                                lots = 0.0
                        # Long needs the swing low strictly below entry; short
                        # needs the swing high strictly above entry. Stale/
                        # equal levels would make compute_sl_tp_* raise.
                        elif (
                            discrete_action == 1
                            and not np.isnan(last_swing_low)
                            and last_swing_low < entry_price
                        ):
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
                            has_levels = True
                        elif (
                            discrete_action == -1
                            and not np.isnan(last_swing_high)
                            and last_swing_high > entry_price
                        ):
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
                            has_levels = True
                        else:
                            # No swing levels available: reject the entry
                            # (treat as hold) rather than open a naked position.
                            lots = 0.0

                        if has_levels:
                            sl_already_hit = False
                            if sl_price is not None:
                                if discrete_action == 1 and fill_bid <= sl_price:
                                    sl_already_hit = True
                                elif discrete_action == -1 and fill_ask >= sl_price:
                                    sl_already_hit = True
                            if not sl_already_hit:
                                self.position = self.broker.open_position(
                                    self.account, (fill_bid, fill_ask), lots, discrete_action
                                )
                            else:
                                self.position = None
                        if self.position:
                            self.position.sl_price = sl_price
                            self.position.tp_price = tp_price
                            self.position.risk_frac = risk_frac
                            self.position.rr_ratio = rr_ratio
                            entry_level, cross_time = self._matched_entry_level(
                                session_id, discrete_action
                            )
                            sweep_delay = (
                                float("nan")
                                if cross_time is None
                                else float((bar_time - cross_time).total_seconds())
                            )
                            self.trade_log.append(
                                {
                                    "type": "open",
                                    "strategy": self.strategy.name,
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
                                    "level_type": entry_level,
                                    "sweep_delay_s": sweep_delay,
                                    "action_tp_mode": getattr(self, "_selected_tp_mode", "rr"),
                                    "asian_high": float(feat_row.get("asian_high", float("nan"))),
                                    "asian_low": float(feat_row.get("asian_low", float("nan"))),
                                    "sweep_high": float(feat_row.get("sweep_high", float("nan"))),
                                    "sweep_low": float(feat_row.get("sweep_low", float("nan"))),
                                    "manipulation_high": float(
                                        feat_row.get("po3_manipulation_high", float("nan"))
                                    ),
                                    "manipulation_low": float(
                                        feat_row.get("po3_manipulation_low", float("nan"))
                                    ),
                                    "manipulation_end": float(
                                        feat_row.get("po3_manipulation_end", float("nan"))
                                    ),
                                    "distribution_phase": float(
                                        feat_row.get("po3_distribution", float("nan"))
                                    ),
                                    "ifvg_zone_low": float(
                                        feat_row.get(
                                            "ifvg_bull_low",
                                            feat_row.get("ifvg_bear_low", float("nan")),
                                        )
                                    ),
                                    "ifvg_zone_high": float(
                                        feat_row.get(
                                            "ifvg_bull_high",
                                            feat_row.get("ifvg_bear_high", float("nan")),
                                        )
                                    ),
                                }
                            )
                            self.sessions_with_trades.add(session_id)

        self.equity_curve.append(self.account.equity)
        pnl_step = self.account.equity - self.equity_curve[-2]
        self.pnl_history.append(pnl_step)

        reward = self._calculate_reward(pnl_step, bar, feat_row, done, truncated)

        obs = self._get_observation()
        info = {"equity": self.account.equity, "position": self.position is not None}

        # Increment step counters: ``step_idx`` is the absolute bar index into
        # the data; ``episode_step_count`` is the within-episode counter used
        # to enforce ``max_episode_steps``. Bumping the per-episode counter
        # *before* the truncation check lets a configured cap of N allow
        # exactly N step() returns.
        self.step_idx += 1
        self.episode_step_count += 1

        # Apply max_episode_steps truncation last so it overrides
        # data-end (done=True) and guardrail-breach (done=True) signals
        # consistently: when the cap fires, we report truncated=True and
        # keep the rest of the per-step accounting intact.
        if self.max_episode_steps is not None and self.episode_step_count >= self.max_episode_steps:
            truncated = True

        return obs, float(reward), done, truncated, info

    def _matched_entry_level(self, session_id: int, discrete_action: int) -> tuple[str | None, Any]:
        """Match an entry to the most recently crossed liquidity level.

        Args:
            session_id: Current trading session identifier.
            discrete_action: Signed action (+1 long, -1 short).

        Returns:
            Tuple of (level name or None, crossing time or None). Longs
            match high levels, shorts match low levels.
        """
        candidates = {
            name
            for (sid, name) in self._level_crosses
            if sid == session_id
            and (name.endswith("high") if discrete_action > 0 else name.endswith("low"))
        }
        if not candidates:
            return None, None
        latest_name = max(candidates, key=lambda n: self._level_crosses[(session_id, n)])
        return latest_name, self._level_crosses[(session_id, latest_name)]

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

    def _progress_market(
        self,
        bar: pd.Series,
        feat_row: pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
    ) -> tuple[float, float, pd.DataFrame | None]:
        """Advance market state for the current bar.

        Tracks liquidity-level crossings, updates eval-mode session
        bookkeeping, marks open positions to market, computes the latency-shifted
        fill quote, and force-closes positions at session boundaries when
        ``block_overnight`` is enabled.

        Returns ``(fill_bid, fill_ask, next_bar)`` where ``next_bar`` is the
        latency-shifted bar or ``None`` when ``fill_idx`` is past the data end.
        """
        price = float(bar["close"])
        for level_name, level_value in (
            ("london_high", feat_row.get("london_high", float("nan"))),
            ("asian_high", feat_row.get("asian_high", float("nan"))),
            ("london_low", feat_row.get("london_low", float("nan"))),
            ("asian_low", feat_row.get("asian_low", float("nan"))),
        ):
            level_value = float(level_value)
            key = (session_id, level_name)
            if key in self._level_crosses or not np.isfinite(level_value):
                continue
            crossed = level_name.endswith("high") and price > level_value
            crossed = crossed or (level_name.endswith("low") and price < level_value)
            if crossed:
                self._level_crosses[key] = bar_time

        if not self.episodic:
            self.all_sessions.add(session_id)
            if session_id != self.prev_session:
                self.account.reset_daily()
                self.prev_session = session_id

        bid, ask = self._bar_quote(bar)
        if self.position is not None:
            self.broker.mark_to_market(self.account, self.position, (bid, ask))

        if fill_idx < len(self.bars):
            next_bar = self.bars.iloc[fill_idx]
            fill_bid, fill_ask = self._bar_quote(next_bar)
        else:
            next_bar = None
            fill_bid, fill_ask = bid, ask

        if self.block_overnight and self.position is not None:
            next_sid = (
                int(next_bar["session_id"])
                if next_bar is not None and "session_id" in next_bar.index
                else session_id
            )
            if next_sid != session_id:
                pnl, fill_price = self.broker.close_position(
                    self.account, self.position, (fill_bid, fill_ask)
                )
                self.trade_log.append(
                    {
                        "type": "eod_close",
                        "pnl": pnl,
                        "price": fill_price,
                        "reason": "session_end",
                        "bar": self.step_idx,
                        "time": bar_time,
                        "equity": self.account.equity,
                    }
                )
                self.position = None
                self.sessions_with_trades.add(session_id)

        return fill_bid, fill_ask, next_bar

    def _calculate_reward(
        self,
        pnl_step: float,
        bar: pd.Series,
        feat_row: pd.Series,
        done: bool,
        truncated: bool,
    ) -> float:
        """Compute the step reward from PnL and market state.

        Delegates to the active reward implementation (DSR or composite)
        with the appropriate kwargs.
        """
        daily_loss = self.initial_balance - self.account.equity
        minutes_since_open = max(0, (self.step_idx - self.ny_session_start_idx)) * 5.0 / 60.0

        reward_kwargs: dict[str, Any] = {
            "daily_loss": daily_loss,
            "daily_loss_limit": self.guardrails.daily_loss_limit,
            "initial_balance": self.initial_balance,
            "breach": done and truncated,
        }

        if self.use_sweep_reward and isinstance(self.reward_fn, CompositeReward):
            position_changed = (
                self.position is not None
                and len(self.trade_log) > 0
                and self.trade_log[-1].get("type") in ("open", "close")
            )
            reward_kwargs.update(
                {
                    "cost": 0.0,
                    "price": float(bar["close"]),
                    "london_high": float(feat_row.get("london_high", float("nan"))),
                    "london_low": float(feat_row.get("london_low", float("nan"))),
                    "asian_high": float(feat_row.get("asian_high", float("nan"))),
                    "asian_low": float(feat_row.get("asian_low", float("nan"))),
                    "minutes_since_open": float(minutes_since_open),
                    "position_changed": bool(position_changed),
                }
            )

        if self.strategy_actions and isinstance(self.reward_fn, CompositeReward):
            if getattr(self.reward_fn, "strategy_reward", None) is not None:
                pos_dir = 0
                in_ifvg = False
                if self.position is not None:
                    pos_dir = int(self.position.direction)
                    in_ifvg = (
                        float(feat_row.get("price_in_ifvg_bull", 0.0)) > 0
                        if pos_dir == 1
                        else float(feat_row.get("price_in_ifvg_bear", 0.0)) > 0
                    )
                reward_kwargs["strategy_context"] = {
                    "position_changed": bool(
                        self.position is not None
                        and len(self.trade_log) > 0
                        and self.trade_log[-1].get("type") in ("open", "close")
                    ),
                    "direction": pos_dir,
                    "in_ifvg": in_ifvg,
                    "manipulation_active": (
                        float(feat_row.get("po3_manipulation_active", 0.0)) > 0
                    ),
                    "manipulation_end": (float(feat_row.get("po3_manipulation_end", 0.0)) > 0),
                    "distribution_phase": (float(feat_row.get("po3_distribution", 0.0)) > 0),
                    "sweep_high": float(feat_row.get("sweep_high", 0.0)) > 0,
                    "sweep_low": float(feat_row.get("sweep_low", 0.0)) > 0,
                    "bos_up": float(feat_row.get("bos_up", 0.0)) > 0,
                    "bos_down": float(feat_row.get("bos_down", 0.0)) > 0,
                }

        return float(self.reward_fn(pnl_step, **reward_kwargs))

    def _get_observation(self) -> dict[str, np.ndarray[Any, Any]]:
        """Construct observation dict."""
        # Time-series features (strategy raw price levels excluded — Agent.md §11)
        start_idx = max(0, self.step_idx - self.obs_window)
        seq = np.asarray(
            self._obs_features.iloc[start_idx : self.step_idx].values, dtype=np.float32
        )
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

        if self.normalize_account:
            # Rescale so the account vector matches the ~O(1) magnitude of
            # the z-scored seq features. Without this, equity (1e5) and
            # raw open_pnl dwarf the time-series signal.
            init_bal = self.initial_balance
            norm_equity = float(np.log(self.account.equity / init_bal)) if init_bal > 0 else 0.0
            norm_pnl = open_pnl / init_bal if init_bal > 0 else 0.0
            # The current bar is only known inside step(); in reset() we
            # fall back to the first bar of the obs window. Both are
            # O(1)-magnitude normalizers, so the choice barely matters.
            if 0 <= self.step_idx < len(self.bars):
                current_close = float(self.bars.iloc[self.step_idx]["close"])
            else:
                current_close = 1.0
            norm_dist = dist_to_sl / current_close if current_close > 0 else 0.0
            account_state = np.array(
                [norm_equity, pos_dir, norm_pnl, unrealised_r, norm_dist],
                dtype=np.float32,
            )
        else:
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

        obs: dict[str, np.ndarray[Any, Any]] = {"seq": seq, "account": account_state}

        # Add VAE latent embedding if enabled
        if self.use_vae and self.vae is not None and self.pre_ny_data is not None:
            # Get the current day's pre-NY sequence
            # Assuming pre_ny_data is aligned with bars and features
            # and contains the full pre-NY sequence for each day
            current_idx = min(self.step_idx, len(self.pre_ny_data) - 1)
            pre_ny_seq = np.asarray(self.pre_ny_data.iloc[current_idx].values, dtype=np.float32)
            pre_ny_seq = np.nan_to_num(pre_ny_seq, nan=0.0)

            # Get VAE latent embedding (mu)
            import torch

            pre_ny_tensor = torch.from_numpy(pre_ny_seq).unsqueeze(0).float()
            with torch.no_grad():
                mu, _ = self.vae.encode(pre_ny_tensor)
            vae_z = mu.numpy().astype(np.float32)

            obs["vae_z"] = vae_z

        return obs
