"""Gymnasium environment for RL-based trading with structure-aware SL/TP.

Wraps the backtest engine with a standard Gym interface. Actions support:
- Discrete: {hold=0, enter_long=1-9, enter_short=10-18, exit=19}
- Continuous: Box(-1, 1) for proportional position sizing

Observation: Dict space with time-series features (60-bar window) + account state.
Reward: Differential Sharpe Ratio (DSR) or Sweep Confirmation Reward.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.account import AccountState
from ..backtest.broker import Broker, Position
from ..backtest.costs import COST_US100, CostModel
from ..backtest.guardrails import FTMOGuardrails
from ..backtest.risk import compute_lots, compute_sl_tp_long, compute_sl_tp_short
from ..data.session import ny_session_mask
from ..data.ticks import TickBook
from ..envs.entry_state import (
    BASE_ACCOUNT_DIM,
    ENTRY_STATE_DIM,
    EntryCandidate,
    EntryEvidence,
    EntryStateMachine,
    entry_state_one_hot,
    immutable_setup_levels,
)
from ..envs.feature_row import BarView, FeatureRow
from ..envs.reward import DSRReward, PnLReward, RMultipleReward
from ..envs.strategies import BaselineStrategy, TradingStrategy
from ..envs.sweep_reward import CompositeReward, SweepConfirmationReward
from ..features.build import attach_reachable_r, select_obs_columns

if TYPE_CHECKING:
    from ..models.vae import VAE
from ..train.debug_trace import (
    DebugStepError,
    new_debug_window,
    note_debug_step,
    summarize_window,
    wiring_from_env,
)
from ..train.equity_gate import assess_train_equity
from .observation import (
    DEFAULT_HTF_WINDOWS,
    HTF_BRANCHES,
    closed_window,
    normalized_account_vector,
    pad_observation_window,
    run_starts,
    split_obs_columns,
)

#: Fallback bar spacing when the index has no usable delta (single bar, or a
#: non-monotonic index). The env steps the M1 spine, so one minute is the safe
#: default rather than the old hard-coded five.
_DEFAULT_MINUTES_PER_STEP = 1.0

# Broker session clock used for wall-clock timestamps (trade log, equity curve).
# Frames mode indexes are tz-aware in this zone; the bundle mode stores the same
# UTC epoch as a naive datetime64[ns] array, so both paths are normalised here.
BAR_TIME_TZ = "Etc/GMT-3"


def _minutes_per_step(index: pd.DatetimeIndex) -> float:
    """Median bar spacing in minutes, read from the index itself.

    Assumes nothing about the timeframe: the environment is fed M1 bars today,
    but a resampled feed must not silently make every time-based term wrong.
    """
    if index is None or len(index) < 2:
        return _DEFAULT_MINUTES_PER_STEP
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    deltas = np.diff(nanos)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return _DEFAULT_MINUTES_PER_STEP
    minutes = float(np.median(positive)) / 60_000_000_000.0
    return minutes if minutes > 0.0 else _DEFAULT_MINUTES_PER_STEP


_DECISION_KEYS: tuple[str, ...] = (
    "action_direction",
    "action_intensity",
    "action_stop",
    "action_risk",
    "action_target",
    "action_sl_mode",
    "action_exit",
    "intensity_u",
    "stop_u",
    "risk_u",
    "target_u",
    "sl_mode_u",
    "exit_u",
)


def _empty_decision() -> dict[str, float]:
    return {key: float("nan") for key in _DECISION_KEYS}


def _empty_entry_diag(entry_state_machine: bool = False) -> dict[str, int]:
    """Per-episode entry diagnostics (strategy_actions).

    Defined once so ``__init__`` and ``reset`` cannot drift apart. Every
    ``rejected_*`` bucket corresponds to exactly one ``entry_rejection_reason``
    string; ``entry_rejected_total`` in the step info is derived from the
    ``rejected_`` prefix rather than stored, so it cannot go stale.
    """
    diagnostics = {
        # Requested entry suppressed before the order was built.
        "hold_low_intensity": 0,
        "hold_no_context": 0,
        "hold_risk_floor": 0,
        "rejected_overnight": 0,
        "rejected_blackout": 0,
        "rejected_session_cap": 0,
        "rejected_cooldown": 0,
        "rejected_manipulation": 0,
        "rejected_gate": 0,
        "rejected_soft_brick": 0,
        # Order geometry rejected (place_strategy_order returned no survivor).
        "rejected_sl": 0,
        "rejected_rr": 0,
        "rejected_ema": 0,
        # Geometry existed but the stop was already through the fill quote.
        "rejected_sl_already_hit": 0,
        "opened": 0,
    }
    if entry_state_machine:
        diagnostics["rejected_entry_state"] = 0
    return diagnostics


#: Rejection reason -> ``_entry_diag`` bucket. Every site that refuses an entry
#: must go through :meth:`TradingEnv._reject`, which increments the mapped
#: bucket and sets the reason atomically, so counter and reason cannot drift
#: apart the way the old "ema increments rejected_sl but reports no_valid_tp"
#: mismatch did. A reason missing from this map raises ``KeyError`` at the
#: rejection site instead of silently dropping into the funnel.
_REJECTION_COUNTERS: dict[str, str] = {
    "overnight_boundary": "rejected_overnight",
    "open_blackout": "rejected_blackout",
    "session_cap": "rejected_session_cap",
    "entry_cooldown": "rejected_cooldown",
    "manipulation_unconfirmed": "rejected_manipulation",
    "entry_gate": "rejected_gate",
    "soft_brick": "rejected_soft_brick",
    "entry_state_not_armed": "rejected_entry_state",
    "no_valid_sl": "rejected_sl",
    "no_valid_tp": "rejected_rr",
    "no_ema_exit": "rejected_ema",
    "sl_already_hit": "rejected_sl_already_hit",
}


_PRICE_TOL = 1e-4
# Monotonic SL mode ordering: fixed → breakeven → trailing.
# Used to advance the state machine only forward.
_SL_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}
_TP_MODE_RANK: dict[str, int] = {"fixed": 0, "breakeven": 1, "trailing": 2}


def _at(values: np.ndarray[Any, Any], index: int) -> float:
    if values.size <= index:
        return float("nan")
    return float(values[index])


def _decision_from_action(
    raw: np.ndarray[Any, Any],
    unit: np.ndarray[Any, Any],
    direction_offset: int,
    sl_mode_offset: int = 0,
    tp_mode_offset: int = 0,
    multi_tp_offset: int = 0,
    simplex_offset: int = 0,
) -> dict[str, float]:
    """Raw box endpoints and the unit-interval coordinates of one action."""
    off = int(direction_offset)
    target_width = 3 if multi_tp_offset else 1
    sm_off = int(sl_mode_offset)
    tm_off = int(tp_mode_offset)
    sim_off = int(simplex_offset)
    sl_idx = 3 + off + target_width
    tp_idx = sl_idx + sm_off
    z_idx = tp_idx + tm_off
    exit_idx = z_idx + sim_off

    result = {
        "action_direction": _at(raw, 0) if off else float("nan"),
        "action_intensity": _at(raw, off),
        "action_stop": _at(raw, 1 + off),
        "action_risk": _at(raw, 2 + off),
        "action_target": _at(raw, 3 + off),
        "action_sl_mode": _at(raw, sl_idx) if sm_off else float("nan"),
        "action_exit": _at(raw, exit_idx),
        "intensity_u": _at(unit, off),
        "stop_u": _at(unit, 1 + off),
        "risk_u": _at(unit, 2 + off),
        "target_u": _at(unit, 3 + off),
        "sl_mode_u": _at(unit, sl_idx) if sm_off else float("nan"),
        "exit_u": _at(unit, exit_idx),
    }
    if multi_tp_offset:
        result["action_tp1_sel"] = _at(raw, 3 + off)
        result["action_tp2_sel"] = _at(raw, 4 + off)
        result["action_tp3_sel"] = _at(raw, 5 + off)
        result["tp1_sel_u"] = _at(unit, 3 + off)
        result["tp2_sel_u"] = _at(unit, 4 + off)
        result["tp3_sel_u"] = _at(unit, 5 + off)
    if tm_off:
        result["action_tp_mode"] = _at(raw, tp_idx)
        result["tp_mode_u"] = _at(unit, tp_idx)
    if sim_off:
        result["action_z1"] = _at(raw, z_idx)
        result["action_z2"] = _at(raw, z_idx + 1)
        result["z1_u"] = _at(unit, z_idx)
        result["z2_u"] = _at(unit, z_idx + 1)
    return result


class TradingEnv(gym.Env[dict[str, np.ndarray[Any, Any]], int | np.ndarray[Any, Any]]):
    """RL trading environment with structure-aware SL/TP and hybrid action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
        obs_window: int = 60,
        initial_balance: float = 100_000.0,
        cost_model: CostModel = COST_US100,
        broker_kwargs: dict[str, Any] | None = None,
        guardrail_kwargs: dict[str, Any] | None = None,
        risk_frac_range: tuple[float, float] = (0.005, 0.02),
        rr_ratio_range: tuple[float, float] = (1.0, 3.0),
        swing_buffer_pts: float = 1.0,
        sl_mode: str = "fixed",
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
        pre_ny_by_date: dict[Any, np.ndarray[Any, Any]] | None = None,
        ny_session_start_idx: int | None = None,
        fill_latency_bars: int = 0,
        max_episode_steps: int | None = None,
        normalize_account: bool = True,
        strategy: TradingStrategy | None = None,
        strategy_actions: bool = False,
        trader_actions: bool | None = None,
        sl_buffer_pts: float = 0.0,
        strategy_reward: Any = None,
        strategy_weight: float = 0.0,
        block_overnight: bool = True,
        eod_risk: dict[str, Any] | None = None,
        min_sl_atr_mult: float = 0.5,
        min_sl_points: float = 0.0,
        max_entries_per_session: int = 0,
        open_manipulation_bars: int = 10,
        entry_cooldown_bars: int = 0,
        reward_mode: str = "dsr",
        fixed_risk_usd: float = 0.0,
        allow_ema_exit: bool = True,
        risk_mode: str = "dynamic",
        entry_intensity_threshold: float = 0.0,
        obs_features_mmap: str | None = None,
        peak_trailing_dd_limit: float = 0.0,
        tickbook: TickBook | None = None,
        fill_delay_ms: int = 0,
        agent_direction_control: bool = False,
        direction_override_threshold: float = 0.0,
        risk_floor: float = 0.0005,
        allow_agent_sl_mode: bool = True,
        sl_mode_defer_threshold: float = 0.1,
        allow_agent_tp_mode: bool = False,
        tp_mode_defer_threshold: float = 0.1,
        allow_multi_tp: bool = False,
        tp_breakeven_alpha: float = 0.5,
        tp_mode: str = "fixed",
        tp_reference: str = "structural",
        allow_simplex: bool = True,
        mtf: bool = False,
        mtf_windows: dict[str, int] | None = None,
        entry_state_machine: bool = False,
        candidate_max_age_bars: int = 5,
        arm_requires_retest: bool = False,
        entry_state_observation: bool = True,
        bundle_data: Any | None = None,
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
        pre_ny_by_date : dict | None
            Map of calendar ``date`` → ``(seq_len, n_features)`` float32 array
            (M5 OHLCV for 01:05–16:29). Preferred when ``use_vae=True``.
        pre_ny_data : pd.DataFrame | None
            Deprecated legacy table; ignored when ``pre_ny_by_date`` is set.
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
            If True, use the trader-like four-dimensional Box action
            ``[entry_intensity, sl_anchor, risk, RR]`` in ``[-1, 1]^4``
            (affine-mapped to ``[0, 1]`` at decode so PPO's mean-0 policy
            can enter) and route entries through the strategy's context
            direction, SL candidates, gates, and RR/structural TP. Baseline
            (Idea 3) keeps ``False``.
        trader_actions : bool | None
            Alias for ``strategy_actions``. When set, overrides
            ``strategy_actions``.
        sl_buffer_pts : float
            Buffer in price points for the structural stop in strategy
            modes. ``0.0`` = ``sl_mode: exact``; positive = ``buffered``.
        min_sl_atr_mult, min_sl_points : float
            Minimum stop distance (ATR multiple and absolute points).
            Stops tighter than ``max(min_sl_points, min_sl_atr_mult * atr)``
            are rejected when strategy/trader actions are on.
        max_entries_per_session : int
        fixed_risk_usd : float
            When ``risk_mode="fixed"``, every trade risks exactly this many
            dollars, so 1R is a constant known in advance instead of a fraction
            of drifting equity. 0 (default) keeps fractional compounding on live
            equity.
        allow_ema_exit : bool
            When False the agent cannot pick the EMA-21 exit, so every trade must
            target a structural level at least 1R away. True (default) keeps the
            EMA-21 exit available.
        risk_mode : str
            ``"fixed"`` sizes every trade to ``fixed_risk_usd``; ``"dynamic"``
            sizes to live equity times the agent's chosen risk fraction.
            Cap on opens per ``session_id`` (0 = unlimited).
        open_manipulation_bars : int
            No new entry for this many M1 bars after each NY session open.
            ``0`` disables the blackout.
        entry_cooldown_bars : int
            Bars to wait after a close before the next entry (0 = none).
        reward_mode : str
            ``"dsr"`` (default / baseline) or ``"pnl"`` (Idea 1/2).
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
        self.bars: pd.DataFrame = cast(pd.DataFrame, bars)
        self.features: pd.DataFrame = cast(pd.DataFrame, features)
        self._bundle_arrays: dict[str, np.ndarray[Any, Any]] | None = (
            {name: np.asarray(array) for name, array in bundle_data.arrays.items()}
            if bundle_data is not None
            else None
        )
        bundle_labels = bundle_data.manifest.get("labels", {}) if bundle_data is not None else {}
        self._n_bars = (
            len(self._bundle_arrays["bar_time_ns"])
            if self._bundle_arrays is not None
            else len(bars)
            if bars is not None
            else 0
        )
        if self._bundle_arrays is None:
            assert bars is not None and features is not None
            self._ny_indices = self._resolve_ny_indices(bars)
        else:
            times = pd.to_datetime(self._bundle_arrays["bar_time_ns"], utc=True)
            timezone = bundle_labels.get("bar_timezone")
            if timezone:
                times = times.tz_convert(str(timezone))
            if bundle_labels.get("has_session", False):
                sessions = list(bundle_labels.get("session", []))
                ny_codes = [idx for idx, label in enumerate(sessions) if label == "ny"]
                ny_indices = np.flatnonzero(np.isin(self._bundle_arrays["bar_session"], ny_codes))
                self._ny_indices = ny_indices if ny_indices.size else np.arange(self._n_bars)
            else:
                self._ny_indices = self._resolve_ny_indices(
                    pd.DataFrame(index=pd.DatetimeIndex(times))
                )
        self._bundle_has_session = bool(bundle_labels.get("has_session", False))
        self._session_labels = list(bundle_labels.get("session", []))
        self._bundle_has_session_id = bool(bundle_labels.get("has_session_id", False))
        self.obs_window = obs_window
        self.initial_balance = initial_balance
        self.cost_model = cost_model
        self.broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
        self.guardrails = FTMOGuardrails(**(guardrail_kwargs or {}))

        self.risk_frac_range = risk_frac_range
        self.rr_ratio_range = rr_ratio_range
        self.swing_buffer_pts = float(sl_buffer_pts)
        self.sl_mode = str(sl_mode or "fixed").lower()
        if self.sl_mode not in _SL_MODE_RANK:
            raise ValueError(f"Unknown sl_mode: {self.sl_mode!r}")
        self.tp_mode = str(tp_mode or "fixed").lower()
        if self.tp_mode not in _TP_MODE_RANK:
            raise ValueError(f"Unknown tp_mode: {self.tp_mode!r}")
        self.tp_reference = str(tp_reference or "structural")
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.contract_size = contract_size
        self.max_loss_per_trade_usd = max_loss_per_trade_usd
        self.block_overnight = bool(block_overnight)
        eod = dict(eod_risk or {})
        self._eod_max_loss_usd = float(eod.get("max_loss_usd", 500.0))
        self._eod_max_age_hours = float(eod.get("max_age_hours", 8.0))
        self._eod_stale_bars = int(eod.get("stale_bars", 60))
        self._eod_progress_atr = float(eod.get("progress_threshold_atr", 0.25))
        # Risk/sizing alignment: the intraday adverse-excursion cut never
        # fires closer than the trade's own planned loss at the structural
        # stop. Without this, a $500 cutoff flattens a position sized to
        # risk $1,000 and the structural thesis never gets to play out.
        self._eod_respect_planned_risk = bool(eod.get("respect_planned_risk", True))
        self._eod_planned_risk_mult = float(eod.get("planned_risk_mult", 1.1))
        # Breakeven management: once favorable excursion reaches this many
        # stop distances, the stop moves to the entry plus a small buffer.
        # 0 disables the rewrite.
        self._breakeven_trigger_r = float(eod.get("breakeven_trigger_r", 0.0))
        self._breakeven_buffer_pts = float(eod.get("breakeven_buffer_pts", 2.0))
        self._ny_pos = 0
        self.episodic = episodic
        self.use_sweep_reward = use_sweep_reward
        self.min_sl_atr_mult = float(min_sl_atr_mult)
        self.min_sl_points = float(min_sl_points)
        self.max_entries_per_session = int(max_entries_per_session)
        # Sizing mode. "fixed" risks exactly fixed_risk_usd every trade so 1R is
        # a known constant; "dynamic" risks a fraction of live equity instead.
        if risk_mode not in ("fixed", "dynamic"):
            raise ValueError(f"risk_mode must be 'fixed' or 'dynamic', got {risk_mode!r}")
        self.risk_mode = risk_mode
        self.fixed_risk_usd = float(fixed_risk_usd)
        if self.risk_mode == "fixed" and self.fixed_risk_usd <= 0.0:
            raise ValueError(
                "risk_mode='fixed' requires fixed_risk_usd > 0, otherwise every "
                "trade would size to min_lot and risk ~$0"
            )
        self.open_manipulation_bars = int(open_manipulation_bars)
        self._session_open_idx: dict[int, int] = {}
        self._ny_manip_confirmed: set[int] = set()
        self.entry_cooldown_bars = int(entry_cooldown_bars)
        # EMA-21 exit is opt-out. It was selected on 92% of trades, which
        # collapsed the target column onto one non-structural exit and left the
        # structural target choice effectively untrained. Off forces every exit
        # onto a structural target at least 1R away.
        self.allow_ema_exit = bool(allow_ema_exit)
        self.reward_mode = str(reward_mode or "dsr").lower()
        # 0 = intensity never holds; entry follows context_direction.
        self.entry_intensity_threshold = float(entry_intensity_threshold)
        # Training behavior, not an FTMO kill-switch. 0 disables it.
        # At this fraction of peak equity, flatten the open position at the
        # market quote and refuse new risk. Stops and targets are not rewritten.
        self.peak_trailing_dd_limit = float(peak_trailing_dd_limit)
        self._peak_dd_fired = False
        # Opt-in agent direction control (see agent_implementation_plan.md).
        # False keeps the 5-D overlay layout where the strategy owns the side.
        self.agent_direction_control = bool(agent_direction_control)
        # Conviction band (distance from the neutral 0.5) below which the agent
        # defers to ``strategy.context_direction``. 0.0 means the agent's sign
        # always wins except at exactly neutral.
        self.direction_override_threshold = float(direction_override_threshold)
        self.allow_agent_sl_mode = bool(allow_agent_sl_mode)
        self.sl_mode_defer_threshold = float(sl_mode_defer_threshold)
        self.allow_agent_tp_mode = bool(allow_agent_tp_mode)
        self.tp_mode_defer_threshold = float(tp_mode_defer_threshold)
        self.allow_multi_tp = bool(allow_multi_tp)
        self.tp_breakeven_alpha = float(tp_breakeven_alpha)
        self.allow_simplex = bool(allow_simplex)
        self.entry_state_machine = (
            EntryStateMachine(candidate_max_age_bars, arm_requires_retest)
            if entry_state_machine
            else None
        )
        self.entry_state_observation = bool(entry_state_machine and entry_state_observation)
        self.risk_floor = float(risk_floor)
        self._tickbook = tickbook
        self._fill_delay_ms = int(fill_delay_ms)
        # Resolve trader/strategy action flag early (needed for reward + spaces).
        _strategy_actions = (
            bool(trader_actions) if trader_actions is not None else bool(strategy_actions)
        )

        # Initialize reward function. Training breaches are a small terminal
        # spike; eval keeps the -10 report. The guardrail still ends the episode.
        self.reward_fn: DSRReward | CompositeReward | PnLReward | RMultipleReward
        breach_penalty = 0.0 if episodic else -10.0
        base_pnl = (
            PnLReward(soft_loss_frac=0.01, dsr_weight=0.0, breach_penalty=breach_penalty)
            if self.reward_mode in ("pnl", "rr")
            else None
        )
        if (
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

        # Entry-gate: warn once (not per-step) if required features are missing.
        self._entry_gate_warned: bool = False

        # Strategy semantics (Idea 1/2) vs. legacy behaviour (Idea 3 baseline).
        self.strategy: TradingStrategy = strategy if strategy is not None else BaselineStrategy()
        self.strategy_actions = _strategy_actions
        self.sl_buffer_pts = float(sl_buffer_pts)
        self._selected_sl_anchor = 0.0
        self._selected_tp_fraction = 0.0
        self._selected_exit_mode = "structural"
        self._selected_sl_mode = "fixed"
        self._sl_mode_explicit = False
        self._selected_tp_mode = self.tp_mode
        self._tp_mode_explicit = False
        self._selected_tp_selections = (-1.0, -1.0, -1.0)
        self._selected_tp_z = (0.0, 0.0)
        self._decision_action = _empty_decision()
        self._entry_rejection_reason = ""
        self._session_entry_counts: dict[int, int] = {}
        self._cooldown_until_bar: int = -1
        self._last_realized_close_pnl: float | None = None
        self._last_realized_close_r: float | None = None
        self._opened_this_step = False
        self._trigger_requested_this_step = False
        self._debug_trace = False
        self._debug_window: dict[str, Any] | None = None
        feature_names = (
            list(bundle_data.manifest["column_names"]["features_num"])
            if bundle_data is not None
            else list(features.columns)
            if features is not None
            else []
        )
        bind = getattr(self.strategy, "bind_columns", None)
        if callable(bind):
            bind(feature_names)
        if self.strategy_actions:
            missing = [c for c in self.strategy.required_features if c not in feature_names]
            if missing:
                raise ValueError(f"Missing strategy features: {missing}")
        self._max_tp_dist: np.ndarray[Any, Any] | None = None
        if self._bundle_arrays is not None and self.strategy_actions:
            max_tp = self._bundle_arrays["max_tp_dist"]
            self._max_tp_dist = max_tp if max_tp.size else None
        elif (
            self.strategy_actions
            and bars is not None
            and features is not None
            and "atr_5" in features.columns
        ):
            from quant_rl.backtest.tp_reach import max_tp_distance

            dist = max_tp_distance(bars, features["atr_5"]).reindex(bars.index)
            self._max_tp_dist = dist.to_numpy(dtype=float)

        # Model-facing observation frame: the strategy's raw price-level
        # columns stay in ``features`` for execution/risk but must not enter
        # the normalised ``seq`` tensor (Agent.md §11). MTF blocks reuse the
        # same stems with a ``{TF}_`` prefix (e.g. ``M5_last_swing_high``),
        # matched via MTF_RAW_SUFFIXES so unnormalised HTF price magnitudes
        # never reach the encoder.
        # Drop non-numeric columns (e.g. string ``session`` labels) — they cannot
        # be cast to float32 and are not part of the model's normalized observation.
        if self._bundle_arrays is None:
            assert bars is not None and features is not None
            self._obs_features: pd.DataFrame = attach_reachable_r(
                select_obs_columns(features, self.strategy.raw_columns),
                bars,
                features,
            )
            self._obs_columns = list(self._obs_features.columns)
        else:
            assert bundle_data is not None
            self._obs_features = cast(pd.DataFrame, None)
            self._obs_columns = list(bundle_data.manifest["column_names"]["obs_features"])
        self._obs_features_mmap = obs_features_mmap

        # Cache numpy arrays for hot-path env stepping — avoids per-step pandas
        # iloc/Series-creation overhead that starves the GPU.
        #   - _obs_features / _features_arr: immutable post-construction
        #   - OHLC/spread bar caches: rebuilt via ``_sync_bar_arrays()`` (call
        #     again after in-place ``env.bars`` mutations in tests)
        # A read-only memmap is one shared copy for every SubprocVecEnv worker.
        if self._bundle_arrays is not None:
            self._obs_features_arr = self._bundle_arrays["obs_features"]
        elif obs_features_mmap:
            assert self._obs_features is not None
            shared = np.load(obs_features_mmap, mmap_mode="r")
            assert self._obs_features is not None
            if shared.shape != self._obs_features.shape:
                raise ValueError(
                    f"obs memmap shape {shared.shape} != obs columns {self._obs_features.shape}"
                )
            self._obs_features_arr = shared
        else:
            assert self._obs_features is not None
            self._obs_features_arr = self._obs_features.to_numpy(dtype=np.float32)
        self.mtf = bool(mtf)
        self._mtf_windows = dict(DEFAULT_HTF_WINDOWS)
        if mtf_windows:
            self._mtf_windows.update({str(k): int(v) for k, v in mtf_windows.items()})
        self._m1_idx, self._htf_idx = split_obs_columns(self._obs_columns)
        self._htf_starts: dict[str, np.ndarray[Any, Any]] = {}
        if self.mtf:
            for key, _, _ in HTF_BRANCHES:
                cols = self._htf_idx[key]
                block = (
                    self._obs_features_arr[:, cols]
                    if cols.size
                    else np.zeros((len(self._obs_features_arr), 0), dtype=np.float32)
                )
                self._htf_starts[key] = run_starts(np.asarray(block, dtype=np.float64))
        if self._bundle_arrays is not None:
            assert bundle_data is not None
            self._features_cols = list(bundle_data.manifest["column_names"]["features_num"])
            self._features_arr = self._bundle_arrays["features_num"]
            self._bar_times = self._bundle_arrays["bar_time_ns"].view("datetime64[ns]")
            self._session_ids_arr = (
                self._bundle_arrays["bar_session_id"] if self._bundle_has_session_id else None
            )
        else:
            assert bars is not None and features is not None
            _feat_numeric = features.select_dtypes(include="number")
            self._features_cols = list(_feat_numeric.columns)
            self._features_arr = _feat_numeric.to_numpy()
            self._bar_times = bars.index.to_numpy()
            self._session_ids_arr = (
                bars["session_id"].to_numpy(dtype=np.int64)
                if "session_id" in bars.columns
                else None
            )
        self._features_col_to_idx = {c: i for i, c in enumerate(self._features_cols)}
        self._sync_bar_arrays()

        # Action space: strategy actions, continuous, or discrete
        self.continuous_actions = continuous_actions
        self.max_risk_frac = max_risk_frac

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
            n_dims = (
                5
                + (1 if self.agent_direction_control else 0)
                + (1 if self.allow_agent_sl_mode else 0)
                + (1 if self.allow_agent_tp_mode else 0)
                + (2 if self.allow_multi_tp else 0)
                + (2 if (self.allow_multi_tp and self.allow_simplex) else 0)
            )
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

        # Per-step entry diagnostics (strategy_actions); reset each episode.
        self._entry_diag: dict[str, int] = _empty_entry_diag(self.entry_state_machine is not None)

        # NY session start index for time decay penalty. Default to the first
        # tradable NY bar, which is what "minutes since open" measures from.
        self.ny_session_start_idx = (
            int(ny_session_start_idx)
            if ny_session_start_idx is not None
            else int(self._ny_indices[0])
            if len(self._ny_indices)
            else 0
        )
        # Bar spacing in minutes, read from the index rather than assumed. The
        # env steps the M1 spine, so the old hard-coded *5 made
        # "minutes_since_open" five times too large and let the sweep
        # time-decay term dominate the reward.
        time_index = (
            pd.DatetimeIndex(pd.to_datetime(self._bundle_arrays["bar_time_ns"], utc=True))
            if self._bundle_arrays is not None
            else pd.DatetimeIndex(bars.index)
            if bars is not None
            else pd.DatetimeIndex([])
        )
        self._minutes_per_step = _minutes_per_step(time_index)

        # Observation space: market sequence, account state, and optional entry state.
        # features: (obs_window, n_features)
        # account: [equity, position_direction, open_pnl, unrealised_r, dist_to_sl, trailing_dd]
        # vae_z: latent embedding from VAE (if use_vae=True)
        n_features = len(self._obs_columns) if self._obs_columns else 1
        if self.mtf:
            n_features = max(int(self._m1_idx.size), 1)
        vae_latent_dim = vae.encoder.latent_dim if use_vae and vae is not None else 0

        account_dim = BASE_ACCOUNT_DIM + (ENTRY_STATE_DIM if self.entry_state_observation else 0)
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
                    shape=(account_dim,),
                    dtype=np.float32,
                ),
                "seq_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(obs_window,),
                    dtype=np.float32,
                ),
            }
        )
        if self.mtf:
            for key, _, seq_key in HTF_BRANCHES:
                width = max(int(self._htf_idx[key].size), 1)
                length = int(self._mtf_windows.get(key, DEFAULT_HTF_WINDOWS[key]))
                self.observation_space.spaces[seq_key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(length, width),
                    dtype=np.float32,
                )
                self.observation_space.spaces[f"mask_{key}"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(length,),
                    dtype=np.float32,
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

    @classmethod
    def from_bundle(cls, bundle_path: str | Path, **kwargs: Any) -> TradingEnv:
        """Create an environment whose persistent market arrays are read-only mmaps."""
        from datetime import date

        from .shared_bundle import load_bundle

        bundle = load_bundle(bundle_path)
        arrays = {name: np.asarray(array) for name, array in bundle.arrays.items()}
        manifest = bundle.manifest
        labels = manifest.get("labels", {})
        pre_ny_by_date = {
            date.fromisoformat(day): arrays["pre_ny_seq"][row]
            for day, row in labels.get("pre_ny_day_to_row", {}).items()
        }
        env = cls(
            bars=None,
            features=None,
            bundle_data=bundle,
            pre_ny_by_date=pre_ny_by_date,
            tickbook=TickBook(arrays["tick_ts"], arrays["tick_bid"], arrays["tick_ask"]),
            **kwargs,
        )
        return env

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed, options=options)

        self._ny_pos = int(np.searchsorted(self._ny_indices, self.obs_window, side="left"))
        if self._ny_pos >= len(self._ny_indices):
            self._ny_pos = max(0, len(self._ny_indices) - 1)
        self.step_idx = int(self._ny_indices[self._ny_pos]) if len(self._ny_indices) else 0
        self.account = self._create_account()
        self.position: Position | None = None
        self.equity_curve = [self.initial_balance]
        # Wall-clock stamps for each equity_curve point (NY-stepped; not dense M1).
        t0 = self._bar_time(self.step_idx) if len(self._bar_times) else None
        self.equity_times: list[Any] = [t0]
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
        self._year_failed = False
        self._level_crosses: dict[tuple[int, str], Any] = {}
        self._session_entry_counts = {}
        self._cooldown_until_bar = -1
        self._last_realized_close_pnl = None
        self._last_realized_close_r = None
        # ``float | None``: an exit with no stampable R leaves this unset rather
        # than NaN, so the annotation must allow more than the None it starts as.
        self._pending_realized_r: float | None = None
        self._opened_this_step = False
        self._selected_sl_anchor = 0.0
        self._selected_tp_fraction = 0.0
        self._selected_exit_mode = "structural"
        self._selected_sl_mode = "fixed"
        self._sl_mode_explicit = False
        self._selected_tp_mode = self.tp_mode
        self._tp_mode_explicit = False
        self._selected_tp_selections = (-1.0, -1.0, -1.0)
        self._selected_tp_z = (0.0, 0.0)
        self._decision_action = _empty_decision()
        self._entry_rejection_reason = ""
        self._ep_start_equity = float(self.initial_balance)
        self._ep_reward_sum = 0.0
        self._entry_diag = _empty_entry_diag(self.entry_state_machine is not None)

        if self.entry_state_machine is not None:
            self.entry_state_machine.reset()

        obs = self._get_observation()
        return obs, {}

    def _bar_time(self, idx: int) -> pd.Timestamp:
        """Return bar ``idx`` as a tz-aware Timestamp in the broker timezone.

        Frames mode stores tz-aware Timestamps; bundle mode stores a naive
        ``datetime64[ns]`` array whose values are the UTC epoch of each bar, so
        the naive value is localised to UTC and converted to the broker
        timezone. This keeps the two data paths emitting identical (value +
        timezone) timestamps for the equity curve and trade log.
        """
        value = self._bar_times[idx]
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(BAR_TIME_TZ)

    def _create_account(self) -> AccountState:
        """Factory for fresh account state."""
        return AccountState(initial_balance=self.initial_balance)

    def _sync_bar_arrays(self) -> None:
        """Rebuild OHLC/spread numpy caches from ``self.bars``.

        Call after construction and after any in-place mutation of ``env.bars``
        (tests that poke SL/TP levels via ``env.bars.loc[...] = ...``).
        """
        if self._bundle_arrays is not None:
            bars = self._bundle_arrays
            self._open_arr = bars["bar_ohlc"][:, 0]
            self._high_arr = bars["bar_ohlc"][:, 1]
            self._low_arr = bars["bar_ohlc"][:, 2]
            self._close_arr = bars["bar_ohlc"][:, 3]
            self._spread_arr = bars["bar_spread"]
            return
        assert self.bars is not None
        n = len(self.bars)
        self._open_arr = self.bars["open"].to_numpy(dtype=np.float64, copy=True)
        self._high_arr = self.bars["high"].to_numpy(dtype=np.float64, copy=True)
        self._low_arr = self.bars["low"].to_numpy(dtype=np.float64, copy=True)
        self._close_arr = self.bars["close"].to_numpy(dtype=np.float64, copy=True)
        if "spread" in self.bars.columns:
            self._spread_arr = self.bars["spread"].to_numpy(dtype=np.float64, copy=True)
        else:
            self._spread_arr = np.full(n, np.nan, dtype=np.float64)

    def _bar_at(self, idx: int) -> BarView:
        """OHLC (+ spread) view for bar ``idx`` without pandas."""
        return BarView(
            open=float(self._open_arr[idx]),
            high=float(self._high_arr[idx]),
            low=float(self._low_arr[idx]),
            close=float(self._close_arr[idx]),
            spread=float(self._spread_arr[idx]),
        )

    def _feature_row_at(self, idx: int) -> FeatureRow:
        """Feature-matrix row view for bar ``idx`` without ``pd.Series``."""
        return FeatureRow(self._features_arr[idx], self._features_col_to_idx)

    def _check_entry_gate(
        self,
        price: float,
        discrete_action: int,
        feat_row: FeatureRow | pd.Series,
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

    def _check_guardrails(
        self,
        bar_time: Any,
        session_id: int,
        fill_bid: float,
        fill_ask: float,
    ) -> tuple[str | None, bool, bool, bool]:
        """Check guardrail breaches and apply forced closes.

        Returns ``(reason, session_blocked, done, truncated)`` where:

        - ``reason`` is the breach reason string or ``None`` if no breach.
        - ``session_blocked`` is ``True`` when the session is blocked in eval
          mode (breach already occurred earlier this session).
        - ``done`` / ``truncated`` follow the episode mode: in episodic mode
          a breach ends the episode; in eval mode the rollout continues.

        When a breach occurs, any open position is force-closed and the
        breach is recorded in ``breach_log`` / ``breach_events``.
        """
        if self.episodic:
            reason = self.guardrails.breach_reason(self.account)
            session_blocked = False
        elif self._year_failed:
            session_blocked = True
            reason = None
        else:
            session_blocked = session_id in self.breached_sessions
            reason = None if session_blocked else self.guardrails.breach_reason(self.account)
            if reason:
                self.breached_sessions.add(session_id)
                session_blocked = True
                if reason in ("trailing_dd", "max_drawdown"):
                    self._year_failed = True

        if reason:
            if self.position is not None:
                pnl, fill_price = self.broker.close_position(
                    self.account, self.position, (fill_bid, fill_ask)
                )
                record = {
                    "type": "forced_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": reason,
                    "bar": self.step_idx,
                    "time": bar_time,
                    "equity": self.account.equity,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.position = None
                self.sessions_with_trades.add(session_id)
                self._note_close(pnl)
            self.breach_log.append(reason)
            self.breach_events.append(
                {
                    "time": bar_time,
                    "session_id": session_id,
                    "reason": reason,
                    "equity": self.account.equity,
                }
            )
            # Year-fail: terminated (done), not truncated. Eval latch continues.
            done = self.episodic
            truncated = False
        elif session_blocked:
            done = False
            truncated = False
        else:
            done = False
            truncated = False

        return reason, session_blocked, done, truncated

    def _apply_peak_trailing_dd(self, fill_bid: float, fill_ask: float, bar_time: Any) -> bool:
        """Keep peak trailing drawdown inside the training ceiling.

        This is strategy behavior, not an FTMO kill and not a breakeven stop.
        At the ceiling, the open position is closed at the current quote.
        Its stop and target prices are left unchanged. New entries are refused
        while drawdown from the peak is still at the ceiling.
        """
        limit = self.peak_trailing_dd_limit
        if limit <= 0.0:
            return False
        if float(self.account.trailing_drawdown_pct()) < limit:
            return False
        self._peak_dd_fired = True
        if self.position is not None:
            pnl, fill_price = self.broker.close_position(
                self.account, self.position, (fill_bid, fill_ask)
            )
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "peak_dd_behavior",
                "bar": self.step_idx,
                "time": bar_time,
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
        return True

    def _check_sl_tp(
        self,
        bar: BarView,
        fill_bid: float,
        fill_ask: float,
        bar_idx: int | None = None,
    ) -> None:
        """Check whether the open position hit its SL or TP on this bar.

        With a tick book, the first bid (long) or ask (short) beyond the
        level is the fill. ``fill_delay_ms`` is 0, so a later tick is not
        used: a stop cannot bounce back to a better price. That crossing
        tick is already through the level, so a loss can exceed $100 and a
        target can pay more than $100. With no crossing tick, the fill stays
        at the level. Slippage, when configured, moves that level fill only
        against the position. If both levels sit inside one bar and there is
        no tick, the stop wins.
        """
        if self.position is None:
            return
        idx = self.step_idx if bar_idx is None else bar_idx
        log_time = (
            self._bar_time(idx)
            if 0 <= idx < len(self._bar_times)
            else self._bar_time(self.step_idx)
        )
        # Touch detection uses the bar range. The next quote must not be the fill.
        _ = (fill_bid, fill_ask)

        tick_hit = self._tick_exit_quote(log_time)
        if tick_hit is not None:
            kind, quote = tick_hit
            pnl, fill_price = self.broker.close_position(self.account, self.position, quote)
            record = {
                "type": "stop_close" if kind == "sl" else "tp_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_sl" if kind == "sl" else "structure_tp",
                "bar": idx,
                "time": log_time,
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
            return

        sl_hit = False
        if self.position.sl_price is not None:
            if self.position.direction == 1 and float(bar.low) <= self.position.sl_price:
                sl_hit = True
            elif self.position.direction == -1 and float(bar.high) >= self.position.sl_price:
                sl_hit = True

        if sl_hit:
            assert self.position.sl_price is not None
            pnl, fill_price = self.broker.close_position(
                self.account,
                self.position,
                self._sl_tp_fill_quote(int(self.position.direction), float(self.position.sl_price)),
            )
            record = {
                "type": "stop_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "structure_sl",
                "bar": idx,
                "time": log_time,
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
            return

        has_multi_levels = self.position is not None and (
            self.position.tp1_price is not None
            or self.position.tp2_price is not None
            or self.position.tp3_price is not None
        )
        if has_multi_levels and self.position is not None:
            pos = self.position
            for slot_num in (1, 2, 3):
                if self.position is None or pos.size <= 0:
                    break
                slot_px = getattr(pos, f"tp{slot_num}_price")
                if slot_px is None:
                    continue
                hit_mask_bit = 1 << (slot_num - 1)
                if pos.tp_hit_mask & hit_mask_bit:
                    continue

                slot_hit = False
                if pos.direction == 1 and float(bar.high) >= float(slot_px):
                    slot_hit = True
                elif pos.direction == -1 and float(bar.low) <= float(slot_px):
                    slot_hit = True

                if slot_hit:
                    # Check remaining active slots after this one
                    remaining_slots = [
                        s
                        for s in (1, 2, 3)
                        if s > slot_num and getattr(pos, f"tp{s}_price") is not None
                    ]
                    is_final = len(remaining_slots) == 0
                    if is_final:
                        close_size = pos.size
                    else:
                        init_size = float(pos.tp_initial_size or pos.size)
                        active_slots_at_entry = []
                        for s in (1, 2, 3):
                            if (pos.tp_hit_mask & (1 << (s - 1))) or getattr(
                                pos, f"tp{s}_price"
                            ) is not None:
                                active_slots_at_entry.append(s)
                        if slot_num in active_slots_at_entry:
                            idx_in_active = active_slots_at_entry.index(slot_num)
                            frac = (
                                pos.tp_lot_fractions[idx_in_active]
                                if idx_in_active < len(pos.tp_lot_fractions)
                                else (1.0 / len(active_slots_at_entry))
                            )
                        else:
                            frac = 1.0 / max(1, len(pos.tp_lot_fractions))
                        close_size = round(frac * init_size, 2)
                        close_size = min(close_size, pos.size)
                        if close_size <= 0:
                            close_size = pos.size

                    quote = self._sl_tp_fill_quote(int(pos.direction), float(slot_px))
                    pnl, fill_price = self.broker.close_position(
                        self.account,
                        pos,
                        quote,
                        size=close_size,
                    )
                    pos.tp_hit_mask |= hit_mask_bit
                    setattr(pos, f"tp{slot_num}_price", None)
                    record = {
                        "type": "tp_close" if pos.size <= 0 else "tp_partial_close",
                        "pnl": pnl,
                        "price": fill_price,
                        "reason": f"structure_tp{slot_num}",
                        "bar": idx,
                        "time": log_time,
                        "equity": self.account.equity,
                        "tp_slot": slot_num,
                        "size": close_size,
                    }
                    self._stamp_exit(record, fill_price)
                    self.trade_log.append(record)
                    self.sessions_with_trades.add(self._current_session_id())

                    if pos.size <= 0:
                        self.position = None
                        self._note_close(pnl)
                        return
                    else:
                        pos.tp_price = (
                            pos.tp1_price
                            if pos.tp1_price is not None
                            else (pos.tp2_price if pos.tp2_price is not None else pos.tp3_price)
                        )
                        if self._breakeven_trigger_r > 0.0:
                            buffer = self._breakeven_buffer_pts
                            if pos.direction == 1:
                                new_sl = pos.entry_price + buffer
                                if pos.sl_price is None or new_sl > float(pos.sl_price):
                                    pos.sl_price = new_sl
                                    pos.breakeven_done = True
                            else:
                                new_sl = pos.entry_price - buffer
                                if pos.sl_price is None or new_sl < float(pos.sl_price):
                                    pos.sl_price = new_sl
                                    pos.breakeven_done = True
            if self.position is None:
                return
        elif self.position.tp_price is not None:
            tp_hit = False
            if self.position.direction == 1 and float(bar.high) >= self.position.tp_price:
                tp_hit = True
            elif self.position.direction == -1 and float(bar.low) <= self.position.tp_price:
                tp_hit = True

            if tp_hit:
                assert self.position.tp_price is not None
                pnl, fill_price = self.broker.close_position(
                    self.account,
                    self.position,
                    self._sl_tp_fill_quote(
                        int(self.position.direction), float(self.position.tp_price)
                    ),
                )
                record = {
                    "type": "tp_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": "structure_tp",
                    "bar": idx,
                    "time": log_time,
                    "equity": self.account.equity,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.position = None
                self.sessions_with_trades.add(self._current_session_id())
                self._note_close(pnl)
                return

        self._maybe_ema_exit(bar, idx, log_time)

    def _tick_exit_quote(self, bar_time: pd.Timestamp) -> tuple[str, tuple[float, float]] | None:
        """Delayed tick quote for a stop or target, or ``None`` to use the level."""
        book = self._tickbook
        pos = self.position
        if book is None or pos is None:
            return None
        if pos.sl_price is None and pos.tp_price is None:
            return None
        hit = book.delayed_exit_quote(
            pd.Timestamp(bar_time),
            int(pos.direction),
            None if pos.sl_price is None else float(pos.sl_price),
            None if pos.tp_price is None else float(pos.tp_price),
            pd.Timedelta(milliseconds=self._fill_delay_ms),
        )
        if hit is None:
            return None
        kind, bid, ask = hit
        return kind, (bid, ask)

    def _sl_tp_fill_quote(self, direction: int, hit_price: float) -> tuple[float, float]:
        """Bid/ask that closes at *hit_price*, worsened only by adverse slippage.

        Longs sell the bid and shorts buy the ask, so that side is the stop
        or target. The other side is one model spread away so
        ``Broker.close_position`` still sees a two-sided quote.
        """
        spread = self.cost_model.spread_points * self.cost_model.point_size
        # Slippage is quoted in price points, the same unit as ``min_sl_points``,
        # ``sl_buffer_pts`` and ``swing_buffer_pts``, so it is *not* scaled by
        # ``point_size`` the way the tick-denominated spread fallback is. It was
        # scaled before, which meant ``slippage_points: 10`` bought 0.1 of a
        # point — an order of magnitude below the slippage the sizing floor is
        # written to tolerate. Positive only ever widens the fill against us.
        slip = float(self.cost_model.slippage_points)
        if direction == 1:
            bid = hit_price - slip
            ask = bid + spread
        else:
            ask = hit_price + slip
            bid = ask - spread
        return bid, ask

    def _current_session_id(self) -> int:
        """Return the session_id for the current bar."""
        if self._session_ids_arr is not None and 0 <= self.step_idx < len(self._session_ids_arr):
            return int(self._session_ids_arr[self.step_idx])
        return 0

    def _note_close(self, pnl: float, realized_r: float | None = None) -> None:
        """Record realized close PnL/R and start the post-close entry cooldown."""
        self._last_realized_close_pnl = float(pnl)
        if realized_r is None:
            realized_r = self._pending_realized_r
        self._last_realized_close_r = None if realized_r is None else float(realized_r)
        self._pending_realized_r = None
        if self.strategy_actions and self.entry_cooldown_bars > 0:
            self._cooldown_until_bar = self.step_idx + self.entry_cooldown_bars

    def _stamp_exit(self, record: dict[str, Any], fill_price: float) -> None:
        """Attach planned geometry and realized R while the position still exists."""
        pos = self.position
        if pos is None or not pos.sl_ref:
            return
        risk = float(pos.stop_distance)
        if risk <= 1e-12 and pos.sl_price is not None:
            risk = abs(float(pos.entry_price) - float(pos.sl_price))
        realized = (
            float(pos.direction) * (float(fill_price) - float(pos.entry_price)) / risk
            if risk > 1e-12
            else float("nan")
        )
        planned = pos.planned_rr
        record["sl_ref"] = pos.sl_ref
        record["tp_ref"] = pos.tp_ref
        record["planned_rr"] = float("nan") if planned is None else float(planned)
        record["realized_r"] = realized
        # Stash for the reward: _stamp_exit runs immediately before _note_close at
        # every close site, so the R is already computed and need not be passed.
        self._pending_realized_r = None if not np.isfinite(realized) else float(realized)
        record["exit_mode"] = pos.exit_mode
        record["sl_buffer"] = pos.sl_buffer
        record["manipulation"] = pos.manipulation
        record["n_sl"] = pos.n_sl
        record["n_tp"] = pos.n_tp
        for row in reversed(self.trade_log):
            if row.get("type") == "open" and "realized_r" not in row:
                row["realized_r"] = realized
                break

    def _ema21_at(self, idx: int) -> float:
        if idx < 0 or idx >= self._n_bars:
            return float("nan")
        if self._bundle_arrays is not None:
            value = self._bundle_arrays["ema_21"][idx]
            return float(value) if not np.isnan(value) else float("nan")
        if "ema_21" not in self._features_col_to_idx:
            return float("nan")
        value = self._features_arr[idx, self._features_col_to_idx["ema_21"]]
        if pd.isna(value):
            return float("nan")
        return float(value)

    def _maybe_ema_exit(self, bar: BarView, idx: int, log_time: Any) -> None:
        """Close an EMA-21 trade after 1R of favorable travel, if the stop did not.

        A long exits when the bar close is below ``ema_21``. A short exits when
        the close is above it. The stop check runs before this.
        """
        pos = self.position
        if pos is None or pos.exit_mode != "ema_21":
            return
        risk = float(pos.stop_distance)
        if risk <= 1e-12:
            return
        if pos.direction == 1:
            favorable = float(bar.high) - float(pos.entry_price)
            crossed = float(bar.close) < self._ema21_at(idx)
        else:
            favorable = float(pos.entry_price) - float(bar.low)
            crossed = float(bar.close) > self._ema21_at(idx)
        pos.best_favorable = max(float(pos.best_favorable), favorable)
        if pos.best_favorable + 1e-12 < risk or not crossed or not np.isfinite(self._ema21_at(idx)):
            return
        close_px = float(bar.close)
        pnl, fill_price = self.broker.close_position(self.account, pos, (close_px, close_px))
        record = {
            "type": "close",
            "pnl": pnl,
            "price": fill_price,
            "reason": "ema_21_exit",
            "bar": idx,
            "time": log_time,
            "equity": self.account.equity,
        }
        self._stamp_exit(record, fill_price)
        self.trade_log.append(record)
        self.position = None
        self.sessions_with_trades.add(self._current_session_id())
        self._note_close(pnl)

    def _try_enter_position(
        self,
        discrete_action: int,
        risk_frac: float,
        rr_ratio: float,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_bid: float,
        fill_ask: float,
        entry_candidate: EntryCandidate | None = None,
    ) -> None:
        """Attempt to open or flip a position for the current bar.

        If an opposite-position is open, it is closed first. A new position
        is opened only when structural or swing SL/TP levels are available
        and geometrically valid; otherwise the entry is rejected to avoid
        naked positions.
        """
        if self.position is not None and self.position.direction != discrete_action:
            pnl, fill_price = self.broker.close_position(
                self.account, self.position, (fill_bid, fill_ask)
            )
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "bar": self.step_idx,
                "time": bar_time,
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(session_id)
            self._note_close(pnl)

        if self.position is None and discrete_action in [1, -1]:
            sl_price = None
            tp_price = None

            last_swing_low = (
                float(feat_row["last_swing_low"])
                if "last_swing_low" in feat_row.index
                and np.isfinite(float(feat_row["last_swing_low"]))
                else np.nan
            )
            last_swing_high = (
                float(feat_row["last_swing_high"])
                if "last_swing_high" in feat_row.index
                and np.isfinite(float(feat_row["last_swing_high"]))
                else np.nan
            )

            entry_price = float(fill_ask if discrete_action == 1 else fill_bid)

            has_levels = False
            placed = None
            if self.strategy_actions:
                from quant_rl.backtest.entry_levels import place_strategy_order

                placed = place_strategy_order(
                    self.strategy,
                    direction=discrete_action,
                    entry_price=entry_price,
                    feat_row=feat_row,
                    min_dist=self._min_sl_distance(feat_row),
                    sl_fraction=float(self._selected_sl_anchor),
                    tp_fraction=float(self._selected_tp_fraction),
                    exit_mode=str(self._selected_exit_mode),
                    buffer_pts=self.sl_buffer_pts,
                    max_tp_distance=self._max_tp_here(),
                    rr_bounds=self.rr_ratio_range,
                    htf=tuple(self._mtf_windows.keys()),
                    tp_selections=(self._selected_tp_selections if self.allow_multi_tp else None),
                    tp_z=(
                        self._selected_tp_z
                        if (self.allow_multi_tp and self.allow_simplex)
                        else None
                    ),
                    allow_multi_tp=self.allow_multi_tp,
                    allow_simplex=self.allow_simplex,
                )
                sl_price = placed.sl_price
                tp_price = placed.tp_price
                # Size only after the chosen stop exists. An EMA exit has no target price.
                order_ok = sl_price is not None and (
                    placed.exit_mode == "ema_21" or tp_price is not None
                )
                if order_ok:
                    assert sl_price is not None
                    lots = self._compute_lots_for(risk_frac, entry_price, sl_price)
                    has_levels = True
                else:
                    lots = 0.0
                    if placed.reject_kind == "tp":
                        self._reject("no_valid_tp")
                    elif placed.reject_kind == "ema":
                        # A missing EMA-21 level is an exit problem, not a stop
                        # problem. It used to increment rejected_sl while
                        # reporting no_valid_tp, so the counter and the reason
                        # disagreed and the EMA case was invisible in entry_sl.
                        self._reject("no_ema_exit")
                    else:
                        self._reject("no_valid_sl")
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
                lots = self._compute_lots_for(risk_frac, entry_price, sl_price)
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
                lots = self._compute_lots_for(risk_frac, entry_price, sl_price)
                has_levels = True
            else:
                lots = 0.0

            if has_levels:
                entry_equity = float(self.account.equity)
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
                    # The chosen stop is already through the fill quote, so the
                    # order would open and stop out on the same fill. Previously
                    # this was a silent no-trade: no counter, no reason, and the
                    # funnel lost the opportunity entirely.
                    self.position = None
                    self._reject("sl_already_hit")
                if self.position:
                    self._entry_diag["opened"] += 1
                    self.position.sl_price = sl_price
                    self.position.sl_initial_price = sl_price
                    self.position.breakeven_done = False
                    if sl_price is not None:
                        # Loss in account currency if the structural stop fills.
                        # The intraday adverse guard uses this as its floor so
                        # sizing and the guard cannot disagree.
                        self.position.planned_risk_usd = (
                            abs(entry_price - float(sl_price))
                            * float(self.position.size)
                            * self.contract_size
                        )
                    self.position.tp_price = tp_price
                    self.position.risk_frac = risk_frac
                    if placed is not None:
                        self.position.sl_ref = placed.sl_ref
                        self.position.tp_ref = placed.tp_ref
                        self.position.planned_rr = placed.planned_rr
                        self.position.exit_mode = placed.exit_mode
                        self.position.sl_buffer = placed.sl_buffer
                        self.position.manipulation = placed.manipulation
                        self.position.n_sl = placed.n_sl
                        self.position.n_tp = placed.n_tp
                        assert placed.sl_price is not None
                        self.position.stop_distance = abs(entry_price - float(placed.sl_price))
                        # planned_rr is the structural ratio. rr_ratio stays empty.
                        self.position.rr_ratio = None
                        self.position.tp1_price = placed.tp1_price
                        self.position.tp2_price = placed.tp2_price
                        self.position.tp3_price = placed.tp3_price
                        self.position.tp1_ref = placed.tp1_ref
                        self.position.tp2_ref = placed.tp2_ref
                        self.position.tp3_ref = placed.tp3_ref
                        self.position.tp_lot_fractions = placed.tp_lot_fractions
                        self.position.tp_initial_price = (
                            placed.tp3_price if placed.tp3_price is not None else placed.tp_price
                        )
                    else:
                        self.position.rr_ratio = rr_ratio
                        self.position.tp3_price = tp_price
                        self.position.tp_initial_price = tp_price
                        self.position.tp_lot_fractions = (1.0,)
                        if feat_row is not None and tp_price is not None:
                            try:
                                from quant_rl.backtest.entry_levels import _target_menu

                                sl_d = (
                                    abs(entry_price - float(sl_price))
                                    if sl_price is not None
                                    else 0.0
                                )
                                targets = _target_menu(
                                    direction=discrete_action,
                                    entry_price=entry_price,
                                    row=feat_row,
                                    stop_dist=sl_d,
                                    max_tp_distance=float("inf"),
                                    rr_bounds=None,
                                    htf=tuple(self._mtf_windows.keys()),
                                )
                                if targets:
                                    self.position.entry_tp_ref_price = float(targets[0][1])
                                    self.position.tp_ref = str(targets[0][0])
                                    self.position.tp3_ref = str(targets[0][0])
                            except Exception:
                                pass
                    self.position.tp_initial_size = float(self.position.size)
                    self.position.tp_mode = self._selected_tp_mode
                    self.position.tp_mode_overridden = self._tp_mode_explicit
                    self.position.tp_breakeven_done = False
                    self.position.tp_hit_mask = 0
                    self.position.entry_timestamp = bar_time
                    atr_val = feat_row.get("atr_5", feat_row.get("atr", np.nan))
                    self.position.entry_atr = (
                        float(atr_val) if pd.notna(atr_val) and float(atr_val) > 0 else 1.0
                    )
                    if placed is not None and placed.sl_ref == "sl_reference":
                        ref = None
                        if hasattr(self.strategy, "sl_reference"):
                            ref = self.strategy.sl_reference(
                                direction=discrete_action, row=feat_row
                            )
                        self.position.entry_sl_ref_price = (
                            float(ref) if ref is not None and np.isfinite(float(ref)) else None
                        )
                    try:
                        self.position.entry_last_swing_high = float(feat_row["last_swing_high"])
                    except (KeyError, TypeError, ValueError):
                        self.position.entry_last_swing_high = None
                    try:
                        self.position.entry_last_swing_low = float(feat_row["last_swing_low"])
                    except (KeyError, TypeError, ValueError):
                        self.position.entry_last_swing_low = None
                    self.position.sl_mode = self._selected_sl_mode
                    self.position.sl_mode_overridden = self._sl_mode_explicit
                    self.position.best_favorable = 0.0
                    self.position.last_progress_favorable = 0.0
                    self.position.stale_counter = 0
                    self.position.mfe = 0.0
                    self.position.mae = 0.0
                    if self.strategy_actions:
                        self._session_entry_counts[session_id] = (
                            self._session_entry_counts.get(session_id, 0) + 1
                        )
                    if entry_candidate is not None:
                        self.position.entry_setup_types = entry_candidate.setup_types
                        self.position.entry_origin_bar = entry_candidate.origin_bar
                    entry_level, cross_time = self._matched_entry_level(session_id, discrete_action)
                    sweep_delay = (
                        float("nan")
                        if cross_time is None
                        else float(
                            (
                                pd.Timestamp(cast(Any, bar_time))
                                - pd.Timestamp(cast(Any, cross_time))
                            ).total_seconds()
                        )
                    )
                    self._opened_this_step = True
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
                            "rr_ratio": self.position.rr_ratio,
                            "planned_rr": (
                                float("nan")
                                if self.position.planned_rr is None
                                else float(self.position.planned_rr)
                            ),
                            "sl_ref": self.position.sl_ref,
                            "tp_ref": self.position.tp_ref,
                            "sl_buffer": self.position.sl_buffer,
                            "exit_mode": self.position.exit_mode,
                            "manipulation": self.position.manipulation,
                            "n_sl": self.position.n_sl,
                            "n_tp": self.position.n_tp,
                            "sl_menu": (
                                [] if placed is None else [list(pair) for pair in placed.sl_menu]
                            ),
                            "tp_menu": (
                                [] if placed is None else [list(pair) for pair in placed.tp_menu]
                            ),
                            **(
                                self._decision_fields(
                                    placed,
                                    entry_price=entry_price,
                                    risk_frac=risk_frac,
                                    entry_equity=entry_equity,
                                    feat_row=feat_row,
                                    bar_time=bar_time,
                                )
                                if placed is not None
                                else {}
                            ),
                            "bar": self.step_idx,
                            "time": bar_time,
                            "equity": self.account.equity,
                            **(
                                {
                                    "entry_setup_types": list(entry_candidate.setup_types),
                                    "entry_origin_bar": entry_candidate.origin_bar,
                                    "entry_state": "ARMED",
                                }
                                if entry_candidate is not None
                                else {}
                            ),
                            # Always recorded, on the strategy path and the baseline
                            # path alike, so the logged risk can never disagree with
                            # the filled lots. Baseline rows used to omit these,
                            # leaving no record of what a trade actually risked.
                            "stop_distance": (
                                abs(float(self.position.entry_price) - float(sl_price))
                                if sl_price is not None
                                else float("nan")
                            ),
                            "risk_cash": self._actual_risk_usd(
                                abs(float(self.position.entry_price) - float(sl_price))
                                if sl_price is not None
                                else float("nan")
                            ),
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
                            "entry_rejection_reason": "",
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

    def _prior_open_payload(self) -> dict[str, float | str] | None:
        """Detector fields for the open that happened on this step."""
        if not self._opened_this_step or not self.trade_log:
            return None
        row = self.trade_log[-1]
        if row.get("type") != "open":
            return None

        def _num(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        def _text(value: Any) -> str:
            if value is None:
                return ""
            return str(value)

        return {
            "stop_u": _num(row.get("stop_u")),
            "target_u": _num(row.get("target_u")),
            "sl_index": _num(row.get("sl_index")),
            "n_sl": _num(row.get("n_sl")),
            "tp_index": _num(row.get("tp_index")),
            "n_tp": _num(row.get("n_tp")),
            "planned_rr": _num(row.get("planned_rr")),
            "direction": _num(row.get("direction")),
            "exit_mode": _text(row.get("exit_mode")),
            "sl_ref": _text(row.get("sl_ref")),
            "tp_ref": _text(row.get("tp_ref")),
        }

    def _sizing_equity(self) -> float:
        """Equity basis for lot sizing (dynamic risk mode only).

        In ``risk_mode="dynamic"`` the agent's ``risk_frac`` acts on live account
        equity, so risk per trade drifts with the balance. In ``risk_mode="fixed"``
        sizing never touches this value: the dollar budget comes from
        ``fixed_risk_usd`` via :meth:`_risk_budget_usd` and is passed to
        ``compute_lots`` as an explicit amount, which is what makes 1R exactly
        ``fixed_risk_usd`` at every stop width.
        """
        return float(self.account.equity)

    def _risk_budget_usd(self, risk_frac: float) -> float:
        """Dollar risk actually requested for the next trade, by sizing mode.

        ``fixed``  -> ``fixed_risk_usd``, independent of equity and risk_frac.
        ``dynamic`` -> live equity times the agent's chosen fraction.
        """
        if self.risk_mode == "fixed":
            return float(self.fixed_risk_usd)
        return float(self.account.equity) * float(risk_frac)

    def _compute_lots_for(self, risk_frac: float, entry_price: float, sl_price: float) -> float:
        """Lot size for one trade under the active sizing mode.

        Fixed mode passes an explicit dollar budget and drops ``max_loss_cap``:
        that cap is a fraction-of-equity safety rail, and leaving it on silently
        overrode the requested risk (a $500 request was clamped to the $100
        default, so no stop width could ever risk $500).
        """
        if self.risk_mode == "fixed":
            return compute_lots(
                self._sizing_equity(),
                risk_frac,
                entry_price,
                sl_price,
                contract_size=self.contract_size,
                min_lot=self.min_lot,
                max_lot=self.max_lot,
                max_loss_cap=None,
                risk_usd=self._risk_budget_usd(risk_frac),
            )
        return compute_lots(
            self._sizing_equity(),
            risk_frac,
            entry_price,
            sl_price,
            contract_size=self.contract_size,
            min_lot=self.min_lot,
            max_lot=self.max_lot,
            max_loss_cap=self.max_loss_per_trade_usd,
        )

    def _actual_risk_usd(self, stop_distance: float) -> float:
        """Dollar risk the filled position really carries, in account currency.

        Uses the broker's accepted size rather than the requested budget, so the
        logged number is what a stop would cost. Returns NaN when there is no
        open position or the stop distance is unusable.
        """
        pos = self.position
        if pos is None or not np.isfinite(stop_distance) or stop_distance <= 0.0:
            return float("nan")
        return float(pos.size) * float(stop_distance) * float(self.contract_size)

    def _decision_fields(
        self,
        placed: Any,
        *,
        entry_price: float,
        risk_frac: float,
        entry_equity: float,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
    ) -> dict[str, Any]:
        """Action coordinates and context for one strategy open."""
        from quant_rl.backtest.entry_levels import reference_timeframe
        from quant_rl.data.session import get_session

        target = placed.tp_price
        stop_px = placed.sl_price
        target_distance = (
            abs(float(target) - float(entry_price)) if target is not None else float("nan")
        )
        stop_distance = (
            abs(float(entry_price) - float(stop_px)) if stop_px is not None else float("nan")
        )
        if self._bundle_arrays is not None and self._bundle_has_session:
            code = int(self._bundle_arrays["bar_session"][self.step_idx])
            label = self._session_labels[code] if code >= 0 else ""
            session = str(label) if label else ""
        elif self.bars is not None and "session" in self.bars.columns:
            label = self.bars["session"].iloc[self.step_idx]
            session = str(label) if pd.notna(label) and str(label) else ""
        else:
            try:
                session = get_session(pd.Timestamp(bar_time))
            except (TypeError, ValueError):
                session = ""
        return {
            **self._decision_action,
            "sl_timeframe": reference_timeframe(str(placed.sl_ref)),
            "sl_index": int(placed.sl_index),
            "tp_timeframe": (
                "" if placed.exit_mode == "ema_21" else reference_timeframe(str(placed.tp_ref))
            ),
            "tp_index": float("nan") if int(placed.tp_index) < 0 else int(placed.tp_index),
            "stop_distance": stop_distance,
            "target_distance": target_distance,
            # risk_cash is set unconditionally on the open row from the filled
            # lots, not here: reporting the requested budget let risk_cash read
            # 500.0 on trades that really risked $0.82, because min_lot and
            # max_loss_cap had clamped lots below the request.
            "session": session,
            "volatility_regime": self._volatility_regime(),
            "trend_regime": self._trend_regime(feat_row),
        }

    def _volatility_regime(self) -> str:
        """Causal tertile of ``atr_5`` over bars already seen in this episode."""
        if self._bundle_arrays is not None:
            values = self._bundle_arrays["atr_5"][: self.step_idx + 1]
            finite = values[np.isfinite(values)]
            if finite.size < 3:
                return ""
            current = float(finite[-1])
            low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
            if current <= float(low):
                return "low"
            return "high" if current >= float(high) else "mid"
        if "atr_5" not in self._features_col_to_idx:
            return ""
        end = self.step_idx + 1
        if end <= 0:
            return ""
        values = self._features_arr[:end, self._features_col_to_idx["atr_5"]]
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            return ""
        current = float(finite[-1])
        low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
        if current <= float(low):
            return "low"
        if current >= float(high):
            return "high"
        return "mid"

    def _trend_regime(self, feat_row: FeatureRow | pd.Series) -> str:
        if "htf_day_bias" not in feat_row.index:
            return ""
        try:
            bias = float(feat_row.get("htf_day_bias", np.nan))
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(bias):
            return ""
        if bias > 0.0:
            return "up"
        if bias < 0.0:
            return "down"
        return "flat"

    def _min_sl_distance(self, feat_row: FeatureRow | pd.Series) -> float:
        """Minimum allowed |entry - SL| in price points."""
        atr_val = feat_row.get("atr_5", feat_row.get("atr", np.nan))
        atr_f = float(atr_val) if atr_val is not None else float("nan")
        atr = atr_f if np.isfinite(atr_f) and atr_f > 0 else 0.0
        return max(self.min_sl_points, self.min_sl_atr_mult * atr)

    def _filter_sl_candidates(
        self,
        *,
        direction: int,
        entry_price: float,
        candidates: list[tuple[str, float]],
        min_dist: float,
    ) -> list[tuple[str, float]]:
        from quant_rl.backtest.entry_levels import filter_sl_candidates

        return filter_sl_candidates(
            direction=direction,
            entry_price=entry_price,
            candidates=candidates,
            min_dist=min_dist,
        )

    def _max_tp_here(self) -> float:
        """Reachable take-profit distance at the current bar, or NaN."""
        if self._max_tp_dist is None:
            return float("nan")
        i = self.step_idx
        if i < 0 or i >= len(self._max_tp_dist):
            return float("nan")
        return float(self._max_tp_dist[i])

    def _rr_fits(self, entry_price: float, sl_price: float, rr_ratio: float) -> bool:
        from quant_rl.backtest.tp_reach import rr_reaches

        return rr_reaches(entry_price, sl_price, rr_ratio, self._max_tp_here())

    def _resolve_trader_tp(
        self,
        *,
        direction: int,
        entry_price: float,
        sl_price: float,
        reward_fraction: float,
        feat_row: FeatureRow | pd.Series | None = None,
        max_tp_distance: float | None = None,
    ) -> float | None:
        from quant_rl.backtest.entry_levels import dynamic_tp

        del feat_row
        tp, rejected = dynamic_tp(
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            reward_fraction=reward_fraction,
            max_tp_distance=max_tp_distance,
        )
        if rejected:
            return None
        return tp

    def _apply_direction_control(self, ctx: int, u: np.ndarray[Any, Any]) -> int:
        """Combine the agent's signed conviction with the heuristic side.

        ``u`` is the affine-mapped unit-interval action, where ``u[0] == 0.5``
        is neutral (PPO's deterministic mean) and the sign of
        ``u[0] - 0.5`` is the agent's side. The agent overrides the strategy
        only when its conviction is wider than
        ``direction_override_threshold``; otherwise ``ctx`` stands, so the
        heuristic side still governs at neutral. Returns ``-1``/``0``/``+1``.
        """
        conviction = float(u[0]) - 0.5
        if abs(conviction) < max(self.direction_override_threshold, 1e-6):
            return ctx if ctx in (-1, 1) else 0
        return 1 if conviction > 0.0 else -1

    def _finite(self, series: FeatureRow | pd.Series | BarView | None, name: str) -> float | None:
        if series is None or name not in series.index:
            return None
        value = float(series.get(name, np.nan))
        return value if np.isfinite(value) else None

    def _default_sl_mode(self, pos: Position | None = None) -> str:
        """State-machine default SL mode from market state.

        - ``fixed``: reference level not yet tested and TP is beyond the swing.
        - ``breakeven``: reference level has been tested.
        - ``trailing``: a new local swing formed during the move.
        """
        if pos is None:
            pos = self.position
        if pos is None:
            return "fixed"
        ref = pos.entry_sl_ref_price
        if ref is not None and pos.direction == 1:
            ref_taken = float(pos.mae) >= (pos.entry_price - float(ref))
        elif ref is not None and pos.direction == -1:
            ref_taken = float(pos.mae) >= (float(ref) - pos.entry_price)
        else:
            ref_taken = False
        tp_farther = False
        if pos.tp_price is not None and not np.isnan(pos.tp_price):
            if pos.direction == 1:
                swing = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_high",
                )
                tp_farther = swing is not None and float(pos.tp_price) > float(swing)
            elif pos.direction == -1:
                swing = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_low",
                )
                tp_farther = swing is not None and float(pos.tp_price) < float(swing)
        new_swing = False
        if pos.entry_last_swing_high is not None or pos.entry_last_swing_low is not None:
            cur_high = self._finite(
                self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                "last_swing_high",
            )
            cur_low = self._finite(
                self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                "last_swing_low",
            )
            if (
                pos.direction == 1
                and cur_high is not None
                and pos.entry_last_swing_high is not None
            ):
                new_swing = float(cur_high) > float(pos.entry_last_swing_high)
            elif (
                pos.direction == -1 and cur_low is not None and pos.entry_last_swing_low is not None
            ):
                new_swing = float(cur_low) < float(pos.entry_last_swing_low)
        if not ref_taken and tp_farther:
            return "fixed"
        if ref_taken:
            return "breakeven"
        if new_swing:
            return "trailing"
        return "fixed"

    def _default_tp_mode(self, pos: Position | None = None) -> str:
        """State-machine default TP mode. fixed -> breakeven -> trailing (monotonic)."""
        if pos is None:
            pos = self.position
        if pos is None:
            return "fixed"
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return "fixed"
        risk_pts = abs(pos.entry_price - float(sl0))
        # breakeven: favorable excursion >= 0.5 * risk
        if risk_pts > 0 and float(pos.best_favorable) >= 0.5 * risk_pts:
            # Check for new swing -> trailing
            new_swing = False
            if pos.entry_last_swing_high is not None or pos.entry_last_swing_low is not None:
                cur_high = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_high",
                )
                cur_low = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_low",
                )
                if (
                    pos.direction == 1
                    and cur_high is not None
                    and pos.entry_last_swing_high is not None
                ):
                    new_swing = float(cur_high) > float(pos.entry_last_swing_high)
                elif (
                    pos.direction == -1
                    and cur_low is not None
                    and pos.entry_last_swing_low is not None
                ):
                    new_swing = float(cur_low) < float(pos.entry_last_swing_low)
            return "trailing" if new_swing else "breakeven"
        return "fixed"

    def _decode_action(
        self,
        action: int | float | np.ndarray[Any, Any],
        feat_row: FeatureRow | pd.Series | None = None,
    ) -> tuple[int, float, float, str]:
        """Decode an action into (discrete_action, risk_frac, rr_ratio, tp_mode).

        Handles four action formats:
        - Strategy overlay, 5-D Box [-1,1]^5: [intensity, stop, risk, target,
          exit mode]. The strategy owns the trade side (``context_direction``).
          Exit mode ``u >= 0.5`` is ``ema_21``; below that is structural.
        - Strategy overlay, 6-D Box [-1,1]^6 (``agent_direction_control``): dim 0
          is signed direction conviction; exit mode stays last.
        - Continuous (Box(-1,1)): proportional position sizing
        - Discrete (Discrete(20)): 0=hold, 1-9=long variants, 10-18=short
          variants, 19=exit. Those ids still use the fixed RR menu.
        """
        tp_mode = "rr"
        risk_frac = self.risk_frac_range[0]
        rr_ratio = self.rr_ratio_range[0]
        self._selected_sl_anchor = 0.0
        self._selected_tp_fraction = 0.0
        self._selected_exit_mode = "structural"
        self._decision_action = _empty_decision()
        self._entry_rejection_reason = ""
        if self.strategy_actions:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            raw = np.clip(arr, -1.0, 1.0)
            u = 0.5 * (raw + 1.0)
            direction_offset = 1 if self.agent_direction_control and u.size >= 6 else 0
            intensity = float(u[direction_offset]) if u.size > direction_offset else 0.0
            entry_threshold = self.entry_intensity_threshold
            if feat_row is None:
                discrete_action = 0
            elif entry_threshold > 0.0 and intensity < entry_threshold:
                discrete_action = 0
                self._entry_diag["hold_low_intensity"] += 1
            else:
                ctx = int(self.strategy.context_direction(feat_row))  # type: ignore[arg-type]
                discrete_action = (
                    self._apply_direction_control(ctx, u)
                    if direction_offset
                    else (ctx if ctx in (-1, 1) else 0)
                )
                if discrete_action == 0 and ctx not in (-1, 1):
                    self._entry_diag["hold_no_context"] += 1
            # Calculate target width (1 for single TP, 3 for multi-tp)
            target_width = 3 if self.allow_multi_tp else 1

            # sl_mode: present if flag set AND the vector still fits exit after it
            # (sl_mode comes after the TP dimensions, exit stays last)
            sl_base = 3 + direction_offset + target_width
            sl_mode_offset = 1 if self.allow_agent_sl_mode and u.size >= sl_base + 2 else 0
            # tp_mode: present only if sl_mode also present (tp_mode comes after sl_mode)
            tp_mode_offset = (
                1 if self.allow_agent_tp_mode and u.size >= sl_base + sl_mode_offset + 2 else 0
            )
            # simplex z1/z2: present only with multi_tp enabled
            simplex_offset = (
                2
                if (
                    self.allow_multi_tp
                    and self.allow_simplex
                    and u.size >= sl_base + sl_mode_offset + tp_mode_offset + 3
                )
                else 0
            )

            # Compute indices
            sl_idx = sl_base
            tp_idx = sl_idx + sl_mode_offset
            z_idx = tp_idx + tp_mode_offset
            exit_idx = z_idx + simplex_offset

            # Extract stop/risk (indices unchanged regardless of target_width)
            if u.size >= 4 + direction_offset:
                sl_raw = float(raw[1 + direction_offset])
                self._selected_sl_anchor = sl_raw + 1.0
                r_lo, r_hi = self.risk_frac_range
                risk_frac = r_lo + float(u[2 + direction_offset]) * (r_hi - r_lo)
                if self.allow_multi_tp:
                    # Extract tp1_sel, tp2_sel, tp3_sel (raw unit-interval 0..1)
                    self._selected_tp_selections = (
                        float(u[3 + direction_offset]),
                        float(u[4 + direction_offset]),
                        float(u[5 + direction_offset]),
                    )
                    # rr_ratio from tp3 (primary TP slot) for legacy downstream use
                    self._selected_tp_fraction = float(u[5 + direction_offset])
                    rr_ratio = self._selected_tp_fraction
                else:
                    self._selected_tp_fraction = float(u[3 + direction_offset])
                    rr_ratio = self._selected_tp_fraction
                    self._selected_tp_selections = (-1.0, -1.0, -1.0)
            else:
                risk_frac = self.risk_frac_range[0]
                rr_ratio = 0.0

            # Extract exit dim
            if u.size > exit_idx and self.allow_ema_exit:
                self._selected_exit_mode = "ema_21" if float(u[exit_idx]) >= 0.5 else "structural"

            # Decode sl_mode
            self._selected_sl_mode = "structural"
            self._sl_mode_explicit = False
            if sl_mode_offset and u.size > sl_idx:
                u_sl = float(u[sl_idx])
                if abs(u_sl - 0.5) < self.sl_mode_defer_threshold:
                    self._selected_sl_mode = self._default_sl_mode()
                else:
                    sl_mode_idx = min(2, int(u_sl * 3))
                    self._selected_sl_mode = ("fixed", "breakeven", "trailing")[sl_mode_idx]
                    self._sl_mode_explicit = True
            else:
                self._selected_sl_mode = self.sl_mode

            # Decode tp_mode
            self._tp_mode_explicit = False
            if tp_mode_offset and u.size > tp_idx:
                from quant_rl.envs.tp_decoders import decode_tp_mode

                u_tp = float(u[tp_idx])
                decoded, explicit = decode_tp_mode(
                    u_tp, self.tp_mode, self.tp_mode_defer_threshold, is_unit=True
                )
                self._selected_tp_mode = decoded
                self._tp_mode_explicit = explicit
                tp_mode = decoded
            else:
                # No tp_mode dim: keep the legacy sentinel so flag-off decode stays "rr".
                tp_mode = self.tp_mode if self.allow_agent_tp_mode else "rr"
                self._selected_tp_mode = tp_mode

            # Decode z1/z2 for simplex
            if simplex_offset and u.size > z_idx + 1:
                self._selected_tp_z = (float(u[z_idx]), float(u[z_idx + 1]))
            else:
                self._selected_tp_z = (0.0, 0.0)

            self._decision_action = _decision_from_action(
                raw,
                u,
                direction_offset,
                sl_mode_offset,
                tp_mode_offset=tp_mode_offset,
                multi_tp_offset=(target_width - 1),  # 2 if multi_tp else 0
                simplex_offset=simplex_offset,
            )
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
        # The risk floor rejects an entry whose *fraction* is too small to be worth
        # sizing. Under fixed_risk_usd the fraction is ignored entirely, so applying
        # it there would silently cancel trades the agent asked for (Idea 1's
        # risk_frac_range starts at 0.0, which this gate would reject outright).
        if (
            self.fixed_risk_usd <= 0.0
            and discrete_action in (1, -1)
            and risk_frac <= self.risk_floor
        ):
            self._entry_diag["hold_risk_floor"] += 1
            discrete_action = 0
        return discrete_action, risk_frac, rr_ratio, tp_mode

    def enable_debug_trace(self) -> None:
        """Start per-rollout counters and step-failure snapshots."""
        self._debug_trace = True
        self._debug_window = new_debug_window()

    def debug_flush(self) -> dict[str, Any]:
        """Return the counters since the last flush and start a new window."""
        if not self._debug_trace or self._debug_window is None:
            return {}
        summary = summarize_window(self._debug_window)
        self._debug_window = new_debug_window()
        return summary

    def debug_wiring(self) -> dict[str, Any]:
        """Reward and risk settings this env was built with."""
        return wiring_from_env(self)

    def step(
        self,
        action: int | float | np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        """Execute one step. With tracing on, failures become :class:`DebugStepError`."""
        if not self._debug_trace:
            return self._step_untraced(action)
        try:
            return self._step_untraced(action)
        except DebugStepError:
            raise
        except Exception as exc:
            position = self.position
            raise DebugStepError(
                "exception",
                {
                    "action": action.tolist() if isinstance(action, np.ndarray) else action,
                    "reward": None,
                    "reward_parts": {},
                    "position": position is not None,
                    "position_direction": 0 if position is None else int(position.direction),
                },
                exc,
            ) from exc

    def _step_untraced(
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
        if self.step_idx >= self._n_bars or self._ny_pos >= len(self._ny_indices):
            # Survived to data / last NY bar: truncated year end (not a fail).
            return self._get_observation(), 0.0, False, True, {}

        bar = self._bar_at(self.step_idx)
        feat_row = self._feature_row_at(self.step_idx)
        bar_time = self._bar_time(self.step_idx)
        session_id = (
            int(self._session_ids_arr[self.step_idx]) if self._session_ids_arr is not None else 0
        )
        self._last_realized_close_pnl = None
        self._last_realized_close_r = None
        self._pending_realized_r = None
        self._opened_this_step = False
        self._trigger_requested_this_step = False

        # Fill quote for next action (latency-shifted: decision at bar t
        # fills at the quote of bar t + fill_latency_bars, not t + 1)
        fill_idx = self.step_idx + 1 + self.fill_latency_bars
        fill_bid, fill_ask, next_bar = self._progress_market(
            bar, feat_row, bar_time, session_id, fill_idx
        )

        # Decode action (discrete or continuous)
        self._note_session_open(session_id, feat_row)
        self._arm_po3_confirm(session_id)
        entry_machine = self.entry_state_machine
        entry_state_for_action = entry_machine.state if entry_machine is not None else None
        discrete_action, risk_frac, rr_ratio, tp_mode = self._decode_action(action, feat_row)
        self._selected_tp_mode = tp_mode
        entry_candidate: EntryCandidate | None = None
        if entry_machine is not None:
            action_requested = self._entry_action_requested(discrete_action)
            if entry_state_for_action in ("FLAT", "CANDIDATE"):
                if action_requested:
                    self._reject("entry_state_not_armed")
                discrete_action = 0
                risk_frac = 0.0
            elif entry_state_for_action == "IN_POSITION":
                discrete_action = 0
                risk_frac = 0.0
            elif action_requested:
                entry_candidate = entry_machine.request_trigger()
                if entry_candidate is not None:
                    self._trigger_requested_this_step = True
                    discrete_action = entry_candidate.direction
                    self._decision_action["action_direction"] = float(entry_candidate.direction)
                else:
                    discrete_action = 0
                    risk_frac = 0.0
            else:
                discrete_action = 0
                risk_frac = 0.0

        # Check entry gate for new positions
        # Suppress new entries on the last bar of a session (overnight block
        # just closed a position or would have if there was one).
        entering_new_session = self.block_overnight and self._is_ny_session_boundary()
        if entering_new_session and discrete_action != 0:
            # Counting only when an entry was actually requested keeps this a
            # true funnel: a plain hold is not a rejection.
            discrete_action = 0
            if self.strategy_actions:
                self._reject("overnight_boundary")

        # Session entry cap + post-close cooldown (trader / strategy actions).
        if self.strategy_actions and discrete_action != 0:
            if self._in_open_blackout(session_id):
                discrete_action = 0
                risk_frac = 0.0
                self._reject("open_blackout")
            elif (
                self.max_entries_per_session > 0
                and self._session_entry_counts.get(session_id, 0) >= self.max_entries_per_session
            ):
                discrete_action = 0
                risk_frac = 0.0
                self._reject("session_cap")
            elif self.entry_cooldown_bars > 0 and self.step_idx < self._cooldown_until_bar:
                discrete_action = 0
                risk_frac = 0.0
                self._reject("entry_cooldown")

        if self.strategy_actions:
            discrete_action, risk_frac = self._lock_unconfirmed_side(
                discrete_action, feat_row, risk_frac
            )

        # Check entry gate for new positions
        if discrete_action != 0:  # Only check for long/short entries
            if entry_machine is not None and entry_state_for_action == "ARMED":
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
            elif self.strategy_actions:
                gate_ok = self.strategy.validate_entry(
                    direction=discrete_action,
                    row=feat_row,  # type: ignore[arg-type]
                )
            else:
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
            if not gate_ok:
                # Gate not satisfied, force to hold
                discrete_action = 0
                risk_frac = 0.0
                if self.strategy_actions:
                    self._reject("entry_gate")

        # Check guardrails
        reason, session_blocked, done, truncated = self._check_guardrails(
            bar_time, session_id, fill_bid, fill_ask
        )

        if not reason and not session_blocked:
            self._check_sl_tp(bar, fill_bid, fill_ask)
            self._apply_position_guards(
                bar, bar_time, fill_bid, fill_ask, eod=False, feat_row=feat_row
            )

        self._peak_dd_fired = False
        peak_dd_block = self._apply_peak_trailing_dd(fill_bid, fill_ask, bar_time)

        # Action handling
        if not done:
            if not self.strategy_actions and action == 19:  # exit action
                if self.position is not None:
                    pnl, fill_price = self.broker.close_position(
                        self.account, self.position, (fill_bid, fill_ask)
                    )
                    record = {
                        "type": "close",
                        "pnl": pnl,
                        "price": fill_price,
                        "bar": self.step_idx,
                        "time": bar_time,
                        "equity": self.account.equity,
                    }
                    self._stamp_exit(record, fill_price)
                    self.trade_log.append(record)
                    self.position = None
                    self.sessions_with_trades.add(session_id)
                    self._note_close(pnl)
            elif discrete_action != 0 and not session_blocked and not peak_dd_block:
                # Soft brick: stop new entries on daily-loss or trailing-DD
                # soft limits; hard breaches still use session_blocked.
                if self.guardrails.check_soft_daily(
                    self.account
                ) or self.guardrails.check_soft_trailing_dd(self.account):
                    # Soft brick on daily loss or trailing DD. Renamed from
                    # ``soft_brick`` so it counts in the rejected_* funnel; the
                    # reason key below was never set before, so the rejection
                    # was invisible in both the buckets and the trade log.
                    self._reject("soft_brick")
                else:
                    self._try_enter_position(
                        discrete_action,
                        risk_frac,
                        rr_ratio,
                        bar,
                        feat_row,
                        bar_time,
                        session_id,
                        fill_bid,
                        fill_ask,
                        entry_candidate=entry_candidate,
                    )
                    if entry_machine is not None and self._opened_this_step:
                        entry_machine.mark_entered()

        self.equity_curve.append(self.account.equity)
        self.equity_times.append(bar_time)
        pnl_step = self.account.equity - self.equity_curve[-2]
        self.pnl_history.append(pnl_step)

        reward = self._calculate_reward(pnl_step, bar, feat_row, done, truncated)
        reward_parts = {
            key: float(value) for key, value in getattr(self.reward_fn, "last_parts", {}).items()
        }
        if self.peak_trailing_dd_limit > 0.0 and self._peak_dd_fired:
            # Penalty each bar the training peak-DD ceiling is still hit.
            reward -= 1.0
            reward_parts["peak_dd"] = float(reward_parts.get("peak_dd", 0.0)) - 1.0

        if entry_machine is not None:
            entry_machine.observe(self.entry_evidence_context(feat_row))
            if entry_machine.state == "IN_POSITION" and self.position is None:
                entry_machine.mark_closed()

        obs = self._get_observation()
        close_reason = ""
        if self.trade_log:
            last_trade = self.trade_log[-1]
            if last_trade.get("bar") == self.step_idx and last_trade.get("type") not in (
                "open",
                "reject",
            ):
                close_reason = str(last_trade.get("reason") or "")
        if self._entry_rejection_reason and not self._opened_this_step:
            self.trade_log.append(
                {
                    "type": "reject",
                    "entry_rejection_reason": self._entry_rejection_reason,
                    "bar": self.step_idx,
                    "time": bar_time,
                    **({"entry_state": entry_machine.state} if entry_machine is not None else {}),
                    **self._decision_action,
                }
            )
        info: dict[str, Any] = {
            "equity": self.account.equity,
            "position": self.position is not None,
            "position_direction": 0 if self.position is None else int(self.position.direction),
            "close_reason": close_reason,
            "reward_parts": reward_parts,
            "entry_rejection_reason": self._entry_rejection_reason,
            **{f"entry_{k}": v for k, v in self._entry_diag.items()},
            # Derived, never stored, so it cannot drift from the buckets.
            "entry_rejected_total": sum(
                v for k, v in self._entry_diag.items() if k.startswith("rejected_")
            ),
        }
        if entry_machine is not None:
            info["entry_state"] = entry_machine.state
            info["entry_state_action"] = entry_state_for_action
        prior_open = self._prior_open_payload()
        if prior_open is not None:
            info["prior_open"] = prior_open
        risk_u = self._decision_action.get("risk_u")
        if isinstance(risk_u, (int, float)) and np.isfinite(risk_u):
            info["risk_u"] = float(risk_u)

        self._advance_ny_step()
        if entry_machine is not None and entry_machine.state == "IN_POSITION":
            if self.position is None:
                entry_machine.mark_closed()
                info["entry_state"] = entry_machine.state
                if self.entry_state_observation:
                    obs["account"][BASE_ACCOUNT_DIM:] = np.asarray(
                        entry_state_one_hot(entry_machine.state), dtype=np.float32
                    )
        self.episode_step_count += 1

        # Past last NY bar → year survived (truncated, not terminated).
        if self._ny_pos >= len(self._ny_indices) and not done:
            truncated = True

        # Apply max_episode_steps truncation last so it overrides
        # data-end signals consistently. Do not convert a year-fail
        # (terminated) into a pure truncation when the cap also fires.
        if self.max_episode_steps is not None and self.episode_step_count >= self.max_episode_steps:
            truncated = True

        if self._debug_trace and self._debug_window is not None:
            note_debug_step(
                self._debug_window,
                reward=float(reward),
                reward_parts=reward_parts,
                opened=bool(self._opened_this_step),
                entered=discrete_action in (1, -1),
                risk_frac=float(risk_frac),
                close_reason=close_reason,
                action=action,
                position=self.position is not None,
                position_direction=0 if self.position is None else int(self.position.direction),
            )

        self._ep_reward_sum += float(reward)
        if done or truncated:
            n_trades = sum(1 for t in self.trade_log if t.get("type") == "open")
            gate = assess_train_equity(
                pd.Series(self.equity_curve, dtype=float),
                initial=float(self._ep_start_equity),
                breached=bool(self.breach_log),
            )
            info["episode_equity"] = {
                "end_equity": float(self.account.equity),
                "start_equity": float(self._ep_start_equity),
                "reward_sum": float(self._ep_reward_sum),
                "n_trades": int(n_trades),
                "breach_reason": str(self.breach_log[-1]) if self.breach_log else "",
                "slope": float(gate["slope"]),
                "max_peak_trailing_dd": float(gate["max_peak_trailing_dd"]),
            }

        return obs, float(reward), done, truncated, info

    def _note_session_open(self, session_id: int, feat_row: FeatureRow | pd.Series) -> None:
        """Remember the first bar of this session and a post-open manip end."""
        self._session_open_idx.setdefault(session_id, self.step_idx)
        if float(feat_row.get("po3_manipulation_end", 0.0) or 0.0) > 0.0:
            self._ny_manip_confirmed.add(session_id)

    def _arm_po3_confirm(self, session_id: int) -> None:
        """Let Idea 1 use a side only after this session's manipulation ends."""
        if getattr(self.strategy, "name", "") != "po3_ifvg":
            return
        setattr(
            self.strategy,
            "_session_manip_confirmed",
            session_id in self._ny_manip_confirmed,
        )

    def _entry_action_requested(self, discrete_action: int) -> bool:
        """Interpret whether the decoded policy action requests entry execution."""
        if not self.strategy_actions:
            return discrete_action in (-1, 1)
        intensity = self._decision_action.get("intensity_u", float("nan"))
        return bool(np.isfinite(intensity) and intensity >= self.entry_intensity_threshold)

    def entry_evidence_context(self, feat_row: FeatureRow | pd.Series) -> EntryEvidence:
        """Translate existing strategy predicates into one causal evidence record."""
        context_direction = int(self.strategy.context_direction(feat_row))  # type: ignore[arg-type]
        setup_predicate = getattr(self.strategy, "entry_setup", None)
        setup_matches = (
            bool(setup_predicate(feat_row, context_direction))
            if context_direction in (-1, 1) and callable(setup_predicate)
            else context_direction in (-1, 1) and not callable(setup_predicate)
        )

        names = tuple(str(name) for name in feat_row.index)

        def flagged(stem: str) -> bool:
            columns = (name for name in names if name == stem or name.endswith("_" + stem))
            for column in columns:
                try:
                    value = float(feat_row.get(column, 0.0))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return True
            return False

        bull_retest = flagged("ifvg_retest_bull") or flagged("fvg_retest_bull")
        bear_retest = flagged("ifvg_retest_bear") or flagged("fvg_retest_bear")
        direction = context_direction
        if direction not in (-1, 1):
            direction = (
                1
                if bull_retest and not bear_retest
                else -1
                if bear_retest and not bull_retest
                else 0
            )
        suffix = "bull" if direction == 1 else "bear"
        retest_confirmed = (direction == 1 and bull_retest) or (direction == -1 and bear_retest)
        distribution_ready = getattr(self.strategy, "distribution_ready", None)
        distribution_arm = (
            bool(distribution_ready(feat_row, direction))
            if direction in (-1, 1) and callable(distribution_ready)
            else False
        )

        setup_types: set[str] = set()
        if direction in (-1, 1):
            sweep_stem = "sweep_low" if direction == 1 else "sweep_high"
            swing_stem = "swing_low_event" if direction == 1 else "swing_high_event"
            if flagged(sweep_stem):
                setup_types.add("sweep")
            if flagged(swing_stem):
                setup_types.add("swing")
            for setup_type, stems in (
                ("ifvg", (f"ifvg_{suffix}_active", f"ifvg_retest_{suffix}")),
                ("fvg", (f"fvg_{suffix}_active", f"fvg_retest_{suffix}")),
            ):
                if any(flagged(stem) for stem in stems):
                    setup_types.add(setup_type)
        if distribution_arm:
            setup_types.add("distribution")
        if setup_matches and not setup_types:
            setup_types.add(str(getattr(self.strategy, "name", "strategy")))

        snapshot: dict[str, float] = {}
        if direction in (-1, 1):
            geometry_stems = (
                f"ifvg_{suffix}_low",
                f"ifvg_{suffix}_high",
                f"fvg_{suffix}_low",
                f"fvg_{suffix}_high",
            )
            for geometry_stem in geometry_stems:
                for column in names:
                    if column != geometry_stem and not column.endswith("_" + geometry_stem):
                        continue
                    try:
                        value = float(feat_row.get(column, np.nan))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        snapshot[column] = value

        conflicting_sweeps = flagged("sweep_low") and flagged("sweep_high")
        conflicting_swings = flagged("swing_low_event") and flagged("swing_high_event")
        return EntryEvidence(
            bar=self.step_idx,
            evidence_detected=setup_matches,
            invalidated=conflicting_sweeps or conflicting_swings or (bull_retest and bear_retest),
            arm_condition=setup_matches or retest_confirmed or distribution_arm,
            retest_confirmed=retest_confirmed,
            direction=direction,
            setup_types=tuple(sorted(setup_types)),
            setup_levels_snapshot=immutable_setup_levels(snapshot),
        )

    def _lock_unconfirmed_side(
        self,
        discrete_action: int,
        feat_row: FeatureRow | pd.Series,
        risk_frac: float,
    ) -> tuple[int, float]:
        """During an open manipulation, keep only the side the strategy already offers.

        A direction override, or a side invented when the strategy has none,
        becomes no trade. The strategy's own sweep or swing entry stays.
        """
        if not self.strategy_actions or discrete_action == 0:
            return discrete_action, risk_frac
        from quant_rl.backtest.entry_levels import manipulation_state

        if manipulation_state(feat_row) != "against":
            return discrete_action, risk_frac
        ctx = int(self.strategy.context_direction(feat_row))  # type: ignore[arg-type]
        if discrete_action == ctx:
            return discrete_action, risk_frac
        self._reject("manipulation_unconfirmed")
        return 0, 0.0

    def _reject(self, reason: str) -> None:
        """Refuse an entry: bump its counter and stamp the reason, atomically.

        The mapping lives in :data:`_REJECTION_COUNTERS`. An unregistered
        reason raises ``KeyError`` here rather than silently producing a ratio
        or trade-log entry with no backing bucket.
        """
        self._entry_diag[_REJECTION_COUNTERS[reason]] += 1
        # A refusal that arrives after PPO requested an ARMED trigger counts as
        # a trigger refusal, so the arm->trigger conversion rate is measurable.
        if self._trigger_requested_this_step and self.entry_state_machine is not None:
            self.entry_state_machine.trigger_refused += 1
        self._entry_rejection_reason = reason

    def _in_open_blackout(self, session_id: int) -> bool:
        if self.open_manipulation_bars <= 0:
            return False
        start = self._session_open_idx.get(session_id, self.step_idx)
        return (self.step_idx - start) < self.open_manipulation_bars

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

    def _bar_quote(self, bar: BarView) -> tuple[float, float]:
        """Get (bid, ask) from a bar using cost model.

        The bar's raw ``spread`` column is in MT5 broker points, not price
        units, so it must be scaled by ``point_size`` before being used as a
        price-unit spread (see ``quant_rl.backtest.engine._bar_spread_price_units``).
        """
        if np.isfinite(bar.spread):
            bar_spread = float(bar.spread) * self.cost_model.point_size
        else:
            bar_spread = None
        return self.cost_model.bar_quote(float(bar.close), bar_spread=bar_spread)

    def _progress_market(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
    ) -> tuple[float, float, None]:
        """Advance market state for the current bar.

        Tracks liquidity-level crossings, updates eval-mode session
        bookkeeping, marks open positions to market, computes the latency-shifted
        fill quote, and force-closes positions at session boundaries when
        ``block_overnight`` is enabled.

        Returns ``(fill_bid, fill_ask, None)`` — ``None`` is retained for
        API compatibility; the caller in :meth:`step` does not consume the
        latency-shifted bar.
        """
        price = float(bar.close)
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

        self.all_sessions.add(session_id)
        if self.prev_session is not None and session_id != self.prev_session:
            self.account.reset_daily()
        self.prev_session = session_id

        bid, ask = self._bar_quote(bar)
        if self.position is not None:
            self.broker.mark_to_market(self.account, self.position, (bid, ask))

        if fill_idx < self._n_bars:
            fill_bid, fill_ask = self._bar_quote(self._bar_at(fill_idx))
        else:
            fill_bid, fill_ask = bid, ask

        if self.block_overnight and self.position is not None and self._is_ny_session_boundary():
            pnl, fill_price = self.broker.close_position(
                self.account, self.position, (fill_bid, fill_ask)
            )
            record = {
                "type": "eod_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "session_end",
                "bar": self.step_idx,
                "time": bar_time,
                "equity": self.account.equity,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(session_id)
            self._note_close(pnl)

        return fill_bid, fill_ask, None

    def _calculate_reward(
        self,
        pnl_step: float,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        done: bool,
        truncated: bool,
    ) -> float:
        """Compute the step reward from PnL and market state.

        Delegates to the active reward implementation (DSR or composite)
        with the appropriate kwargs.
        """
        daily_loss = float(self.account.daily_loss)
        # Elapsed session time in minutes. Derived from the bar spacing rather
        # than assumed, because the sweep time-decay term is a linear penalty
        # in this value and a wrong factor dominates the whole reward.
        minutes_since_open = max(0, self.step_idx - self.ny_session_start_idx) * (
            self._minutes_per_step
        )

        reward_kwargs: dict[str, Any] = {
            "daily_loss": daily_loss,
            "daily_loss_limit": self.guardrails.daily_loss_limit,
            "soft_daily_loss_limit": self.guardrails.soft_daily_loss_limit,
            "loss_from_initial": float(self.account.loss_from_initial()),
            "soft_max_loss_limit": self.guardrails.soft_max_loss_limit,
            "max_loss_limit": self.guardrails.max_loss_limit,
            "trailing_dd": float(self.account.trailing_drawdown_pct()),
            "soft_trailing_dd_limit": self.guardrails.soft_trailing_dd_limit,
            "trailing_dd_limit": self.guardrails.trailing_dd_limit,
            "initial_balance": self.initial_balance,
            # Year-fail is terminated (done) with truncated=False.
            "breach": bool(done and not truncated),
            "realized_close_pnl": self._last_realized_close_pnl,
            "realized_r": self._last_realized_close_r,
            "equity": float(self.account.equity),
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
                    "price": float(bar.close),
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
                    "position_changed": bool(self._opened_this_step),
                    "direction": pos_dir,
                    "in_ifvg": in_ifvg,
                    "manipulation_active": (
                        float(feat_row.get("po3_manipulation_active", 0.0)) > 0
                    ),
                    "manipulation_end": (float(feat_row.get("po3_manipulation_end", 0.0)) > 0),
                    "distribution_phase": (float(feat_row.get("po3_distribution", 0.0)) > 0),
                    "liquidity": (
                        self.strategy.entry_liquidity(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_liquidity")
                        else True
                    ),
                    "in_gap": (
                        self.strategy.entry_gap(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_gap")
                        else in_ifvg
                    ),
                    "entry_setup": (
                        self.strategy.entry_setup(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_setup")
                        else False
                    ),
                    "distribution_after_ifvg": (
                        self.strategy.distribution_ready(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "distribution_ready")
                        else False
                    ),
                    "sweep_high": float(feat_row.get("sweep_high", 0.0)) > 0,
                    "sweep_low": float(feat_row.get("sweep_low", 0.0)) > 0,
                    "sweep_high_reclaimed": float(feat_row.get("sweep_high_reclaimed", 0.0)) > 0,
                    "sweep_low_reclaimed": float(feat_row.get("sweep_low_reclaimed", 0.0)) > 0,
                    "bos_up": float(feat_row.get("bos_up", 0.0)) > 0,
                    "bos_down": float(feat_row.get("bos_down", 0.0)) > 0,
                }

        return float(self.reward_fn(pnl_step, **reward_kwargs))

    @staticmethod
    def _resolve_ny_indices(bars: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Absolute indices of NY tradable bars, or all bars if none match."""
        if "session" in bars.columns:
            mask = bars["session"].eq("ny").to_numpy()
        else:
            mask = ny_session_mask(pd.DatetimeIndex(bars.index)).to_numpy()
        ny = np.flatnonzero(mask)
        if ny.size == 0:
            ny = np.arange(len(bars), dtype=int)
        return ny

    def _is_ny_session_boundary(self) -> bool:
        """True on the last NY bar of a session (next NY index jumps)."""
        if self._ny_pos + 1 >= len(self._ny_indices):
            return False
        return int(self._ny_indices[self._ny_pos + 1]) > int(self.step_idx) + 1

    def _advance_ny_step(self) -> None:
        """Move to the next NY bar, replaying skipped bars on an overnight hold."""
        cur = int(self.step_idx)
        if self._ny_pos + 1 >= len(self._ny_indices):
            self.step_idx = cur + 1
            self._ny_pos += 1
            return
        nxt = int(self._ny_indices[self._ny_pos + 1])
        if nxt > cur + 1 and self.position is not None and not self.block_overnight:
            bar = self._bar_at(cur)
            bid, ask = self._bar_quote(bar)
            self._apply_position_guards(
                bar,
                self._bar_time(cur),
                bid,
                ask,
                eod=True,
                feat_row=self._feature_row_at(cur),
            )
            self._replay_skipped_bars(cur + 1, nxt)
        self._ny_pos += 1
        self.step_idx = nxt

    def _replay_skipped_bars(self, start: int, end: int) -> None:
        """Mark-to-market and check SL/TP/age on bars the agent does not step."""
        for i in range(start, end):
            if self.position is None:
                break
            bar = self._bar_at(i)
            bid, ask = self._bar_quote(bar)
            self._check_sl_tp(bar, bid, ask, bar_idx=i)
            if self.position is None:
                break
            self.broker.mark_to_market(self.account, self.position, (bid, ask))
            self._apply_position_guards(
                bar,
                self._bar_time(i),
                bid,
                ask,
                eod=False,
                feat_row=self._feature_row_at(i),
            )

    def _eod_adverse_cut_usd(self, pos: Position) -> float:
        """Dollar adverse excursion that fires the intraday max-loss guard.

        The guard must never sit *inside* the trade's own structural stop, or
        a position sized to risk ``ftmo.risk_per_trade_limit`` gets flattened
        at half that risk and never reaches its target. When
        ``respect_planned_risk`` is on, the cutoff is the larger of the
        configured cap and ``planned_risk_mult`` times the loss the position
        would take at its structural stop.
        """
        configured = float(self._eod_max_loss_usd)
        if not self._eod_respect_planned_risk:
            return configured
        planned = float(pos.planned_risk_usd or 0.0)
        if planned <= 0.0:
            return configured
        return max(configured, self._eod_planned_risk_mult * planned)

    def _apply_breakeven(self, pos: Position, favorable: float) -> None:
        """Move the stop to breakeven once the trade has earned its risk.

        Only ever tightens toward entry: never widens a stop and never rewrites
        a stop that already sits at or beyond breakeven. Disabled when
        ``breakeven_trigger_r <= 0``.
        """
        if self._breakeven_trigger_r <= 0.0 or pos.breakeven_done:
            return
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return
        risk_pts = abs(pos.entry_price - float(sl0))
        if risk_pts <= 0.0:
            return
        if favorable < self._breakeven_trigger_r * risk_pts:
            return
        buffer = self._breakeven_buffer_pts
        if pos.direction == 1:
            new_sl = pos.entry_price + buffer
            if pos.sl_price is None or new_sl > float(pos.sl_price):
                pos.sl_price = new_sl
                pos.breakeven_done = True
        else:
            new_sl = pos.entry_price - buffer
            if pos.sl_price is None or new_sl < float(pos.sl_price):
                pos.sl_price = new_sl
                pos.breakeven_done = True

    def _apply_trailing_stop(
        self,
        pos: Position,
        feat_row: FeatureRow | pd.Series | BarView | None,
    ) -> None:
        """Structure trailing: follow the local swing formed during the move."""
        if pos.sl_mode != "trailing":
            return
        buf = self.sl_buffer_pts
        if pos.direction == 1:
            swing = self._finite(feat_row, "last_swing_high")
            if swing is not None:
                new_sl = float(swing) - buf
                if pos.sl_price is None or new_sl > float(pos.sl_price):
                    pos.sl_price = new_sl
        elif pos.direction == -1:
            swing = self._finite(feat_row, "last_swing_low")
            if swing is not None:
                new_sl = float(swing) + buf
                if pos.sl_price is None or new_sl < float(pos.sl_price):
                    pos.sl_price = new_sl

    def _reorder_tp_levels_safe(self, pos: Position) -> None:
        """Ensure tp1 < tp2 < tp3 for long (or tp3 < tp2 < tp1 for short) after adjustment."""
        if pos.direction == 1:
            # Long: TP1 < TP2 < TP3
            prices = [pos.tp1_price, pos.tp2_price, pos.tp3_price]
            active = [i for i, p in enumerate(prices) if p is not None]
            if len(active) >= 2:
                for i in range(len(active) - 1):
                    ai, bi = active[i], active[i + 1]
                    p_ai = prices[ai]
                    p_bi = prices[bi]
                    if p_ai is not None and p_bi is not None and p_ai >= p_bi:
                        # Clamp the nearer TP to be strictly less than the farther one
                        prices[ai] = p_bi - _PRICE_TOL
                        setattr(pos, f"tp{ai + 1}_price", p_ai)
        elif pos.direction == -1:
            # Short: TP3 < TP2 < TP1
            prices = [pos.tp1_price, pos.tp2_price, pos.tp3_price]
            active = [i for i, p in enumerate(prices) if p is not None]
            if len(active) >= 2:
                for i in range(len(active) - 1):
                    ai, bi = active[i], active[i + 1]
                    p_ai = prices[ai]
                    p_bi = prices[bi]
                    if p_ai is not None and p_bi is not None and p_ai <= p_bi:
                        # Clamp the nearer TP to be strictly greater than the farther one
                        prices[ai] = p_bi + _PRICE_TOL
                        setattr(pos, f"tp{ai + 1}_price", p_ai)

    def _apply_tp_breakeven(self, pos: Position, favorable: float) -> None:
        """Tighten nearest active TP to entry + alpha*R. Only tightens; never widens."""
        if not self.allow_agent_tp_mode or pos.tp_breakeven_done:
            return
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return
        risk_pts = abs(pos.entry_price - float(sl0))
        if risk_pts <= 0:
            return
        alpha = self.tp_breakeven_alpha  # config field, default 0.5
        if pos.direction == 1:
            new_tp = pos.entry_price + alpha * risk_pts
        else:
            new_tp = pos.entry_price - alpha * risk_pts
        # Apply only to nearest active TP (tp1 if active, else tp2, else tp3)
        for attr in ("tp1_price", "tp2_price", "tp3_price"):
            current = getattr(pos, attr)
            if current is not None:
                if pos.direction == 1 and new_tp < float(current):
                    setattr(pos, attr, new_tp)
                    pos.tp_breakeven_done = True
                elif pos.direction == -1 and new_tp > float(current):
                    setattr(pos, attr, new_tp)
                    pos.tp_breakeven_done = True
                break  # only nearest active TP

    def _apply_tp_trailing(self, pos: Position, feat_row: Any) -> None:
        """Monotonically move TPs toward profit following the local swing."""
        if not self.allow_agent_tp_mode or pos.tp_mode != "trailing":
            return
        if pos.direction == 1:
            swing = self._finite(feat_row, "last_swing_high")
            if swing is not None:
                # For each active TP: move toward profit monotonically (only increase for long)
                for attr in ("tp1_price", "tp2_price", "tp3_price"):
                    current = getattr(pos, attr)
                    if current is not None and float(swing) > float(current):
                        setattr(pos, attr, float(swing))
        elif pos.direction == -1:
            swing = self._finite(feat_row, "last_swing_low")
            if swing is not None:
                for attr in ("tp1_price", "tp2_price", "tp3_price"):
                    current = getattr(pos, attr)
                    if current is not None and float(swing) < float(current):
                        setattr(pos, attr, float(swing))
        # Revalidate ordering after trailing adjustment
        self._reorder_tp_levels_safe(pos)

    def _apply_position_guards(
        self,
        bar: BarView,
        bar_time: Any,
        fill_bid: float,
        fill_ask: float,
        *,
        eod: bool,
        feat_row: FeatureRow | pd.Series | None = None,
    ) -> None:
        """Age, stale-progress, breakeven, and max-loss guards for a position."""
        pos = self.position
        if pos is None:
            return
        if pos.entry_timestamp is not None:
            age_h = (
                pd.Timestamp(cast(Any, bar_time)) - pd.Timestamp(cast(Any, pos.entry_timestamp))
            ).total_seconds() / 3600.0
            if age_h > self._eod_max_age_hours:
                self._force_close_guard(fill_bid, fill_ask, "max_age", bar_time)
                return
        if pos.direction == 1:
            favorable = float(bar.high) - pos.entry_price
            adverse = pos.entry_price - float(bar.low)
        else:
            favorable = pos.entry_price - float(bar.low)
            adverse = float(bar.high) - pos.entry_price
        pos.mfe = max(pos.mfe, favorable)
        pos.mae = max(pos.mae, adverse)
        pos.best_favorable = max(pos.best_favorable, favorable)
        # Advance the SL mode state machine (monotonic: fixed → breakeven →
        # trailing).  An agent override (outside the defer band) locks the mode;
        # only the default state machine advances here, and only forward.
        if not pos.sl_mode_overridden:
            default_mode = self._default_sl_mode(pos)
            if _SL_MODE_RANK.get(default_mode, 0) > _SL_MODE_RANK.get(pos.sl_mode, 0):
                pos.sl_mode = default_mode
        if pos.sl_mode != "fixed":
            self._apply_breakeven(pos, pos.best_favorable)
        self._apply_trailing_stop(pos, feat_row if feat_row is not None else bar)
        # Advance the TP mode state machine (monotonic: fixed → breakeven → trailing).
        if self.allow_agent_tp_mode and not pos.tp_mode_overridden:
            default_tp = self._default_tp_mode(pos)
            if _TP_MODE_RANK.get(default_tp, 0) > _TP_MODE_RANK.get(pos.tp_mode, 0):
                pos.tp_mode = default_tp
        if self.allow_agent_tp_mode and pos.tp_mode != "fixed":
            self._apply_tp_breakeven(pos, pos.best_favorable)
        if self.allow_agent_tp_mode:
            self._apply_tp_trailing(pos, feat_row if feat_row is not None else bar)
        thresh = self._eod_progress_atr * float(pos.entry_atr or 0.0)
        if thresh > 0.0 and (favorable - pos.last_progress_favorable) >= thresh:
            pos.last_progress_favorable = favorable
            pos.stale_counter = 0
        else:
            pos.stale_counter += 1
        notional_adverse = adverse * pos.size * self.contract_size
        if notional_adverse >= self._eod_adverse_cut_usd(pos):
            self._force_close_guard(fill_bid, fill_ask, "eod_max_loss", bar_time)
            return
        if eod and pos.stale_counter >= self._eod_stale_bars:
            self._force_close_guard(fill_bid, fill_ask, "stale", bar_time)

    def _force_close_guard(
        self, fill_bid: float, fill_ask: float, reason: str, bar_time: Any
    ) -> None:
        if self.position is None:
            return
        pnl, fill_price = self.broker.close_position(
            self.account, self.position, (fill_bid, fill_ask)
        )
        record = {
            "type": "eod_close",
            "pnl": pnl,
            "price": fill_price,
            "reason": reason,
            "bar": self.step_idx,
            "time": bar_time,
            "equity": self.account.equity,
        }
        self._stamp_exit(record, fill_price)
        self.trade_log.append(record)
        self.position = None
        self.sessions_with_trades.add(self._current_session_id())
        self._note_close(pnl)

    def _session_seq_start(self) -> int:
        """First absolute bar index included in the session-scoped ``seq`` window.

        Only bars belonging to the current ``session_id`` are kept (same-day
        NY context). Left-pads when fewer than ``obs_window`` bars exist.
        """
        window_start = max(0, self.step_idx - self.obs_window)
        if self._session_ids_arr is None or not (0 <= self.step_idx < len(self._session_ids_arr)):
            return window_start
        sid = int(self._session_ids_arr[self.step_idx])
        i = self.step_idx
        while i > window_start and int(self._session_ids_arr[i - 1]) == sid:
            i -= 1
        return i

    def _get_observation(self) -> dict[str, np.ndarray[Any, Any]]:
        """Construct observation dict.

        ``seq`` is clipped to the **current** ``session_id`` (same-day NY
        bars only) and left-padded to ``obs_window``. Same-day pre-session
        features live in the current feature row inside that window.
        """
        start_idx = self._session_seq_start()
        seq = self._obs_features_arr[start_idx : self.step_idx]
        if self.mtf and self._m1_idx.size:
            seq = seq[:, self._m1_idx]
        elif self.mtf:
            seq = np.zeros((seq.shape[0], 1), dtype=np.float32)
        seq = cast(np.ndarray[Any, Any], np.nan_to_num(seq, nan=0.0))
        seq, seq_mask = pad_observation_window(seq, self.obs_window)

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

        if 0 <= self.step_idx < self._n_bars:
            current_close = float(self._close_arr[self.step_idx])
        else:
            current_close = 1.0
        trailing_dd = float(self.account.trailing_drawdown_pct())
        exposed_entry_state = (
            self.entry_state_machine.state
            if self.entry_state_observation and self.entry_state_machine is not None
            else None
        )
        if self.normalize_account:
            # Rescale so the account vector matches the ~O(1) magnitude of
            # the z-scored seq features. Without this, equity (1e5) and
            # raw open_pnl dwarf the time-series signal.
            account_state = normalized_account_vector(
                equity=float(self.account.equity),
                initial_balance=float(self.initial_balance),
                pos_dir=pos_dir,
                open_pnl=open_pnl,
                dist_to_sl=dist_to_sl,
                trailing_dd=trailing_dd,
                close=current_close,
                entry_state=exposed_entry_state,
            )
        else:
            raw_account = [
                self.account.equity,
                pos_dir,
                open_pnl,
                unrealised_r,
                dist_to_sl,
                float(self.account.trailing_drawdown_pct()),
            ]
            if exposed_entry_state is not None:
                raw_account.extend(entry_state_one_hot(exposed_entry_state))
            account_state = np.asarray(raw_account, dtype=np.float32)

        obs: dict[str, np.ndarray[Any, Any]] = {
            "seq": seq,
            "seq_mask": seq_mask,
            "account": account_state,
        }
        if self.mtf:
            decision = int(self.step_idx)
            for key, _, seq_key in HTF_BRANCHES:
                cols = self._htf_idx[key]
                block = (
                    self._obs_features_arr[:, cols]
                    if cols.size
                    else np.zeros((self._obs_features_arr.shape[0], 0), dtype=np.float32)
                )
                window, mask = closed_window(
                    np.asarray(block),
                    self._htf_starts[key],
                    decision,
                    int(self._mtf_windows.get(key, DEFAULT_HTF_WINDOWS[key])),
                )
                obs[seq_key] = window
                obs[f"mask_{key}"] = mask

        # Add VAE latent embedding if enabled
        if self.use_vae and self.vae is not None:
            import torch

            latent_dim = int(self.vae.encoder.latent_dim)
            seq_len = int(self.vae.encoder.seq_len)
            n_feat = int(self.vae.encoder.n_features)
            pre_ny_seq: np.ndarray[Any, Any] | None = None

            if self.pre_ny_by_date:
                if self._bundle_arrays is not None:
                    from datetime import date

                    day_key = date.fromordinal(
                        int(self._bundle_arrays["bar_day_ord"][self.step_idx])
                    )
                else:
                    assert self.bars is not None
                    day_key = pd.Timestamp(self.bars.index[self.step_idx]).date()
                pre_ny_seq = self.pre_ny_by_date.get(day_key)
            elif self.pre_ny_data is not None and len(self.pre_ny_data) > 0:
                # Legacy: each row is a flattened (seq_len * n_features) vector
                current_idx = min(self.step_idx, len(self.pre_ny_data) - 1)
                flat = np.asarray(self.pre_ny_data.iloc[current_idx].values, dtype=np.float32)
                flat = np.nan_to_num(flat, nan=0.0)
                if flat.size == seq_len * n_feat:
                    pre_ny_seq = flat.reshape(seq_len, n_feat)
                elif flat.ndim == 2:
                    pre_ny_seq = flat.astype(np.float32, copy=False)

            if pre_ny_seq is None:
                vae_z = np.zeros(latent_dim, dtype=np.float32)
            else:
                pre_ny_seq = np.nan_to_num(np.asarray(pre_ny_seq, dtype=np.float32), nan=0.0)
                if pre_ny_seq.shape != (seq_len, n_feat):
                    # Pad / truncate to the encoder contract
                    fixed = np.zeros((seq_len, n_feat), dtype=np.float32)
                    t = min(seq_len, pre_ny_seq.shape[0])
                    f = min(n_feat, pre_ny_seq.shape[1] if pre_ny_seq.ndim == 2 else n_feat)
                    if pre_ny_seq.ndim == 1:
                        pre_ny_seq = pre_ny_seq.reshape(-1, n_feat)[:t, :f]
                    fixed[-t:, :f] = pre_ny_seq[-t:, :f]
                    pre_ny_seq = fixed
                pre_ny_tensor = torch.from_numpy(pre_ny_seq).unsqueeze(0).float()
                vae_device = next(self.vae.parameters()).device
                pre_ny_tensor = pre_ny_tensor.to(vae_device)
                with torch.no_grad():
                    mu, _ = self.vae.encode(pre_ny_tensor)
                vae_z = mu.detach().cpu().numpy().astype(np.float32).reshape(-1)

            obs["vae_z"] = vae_z

        return obs

    def entry_state_diagnostics(self) -> dict[str, Any]:
        """Aggregate entry-state lifecycle counters for reporting.

        Reads the FSM's own counters (never infers them from feature data), so
        the reporting surface is owned by the object that owns the lifecycle.
        Returns an empty dict when the FSM is off (A3), so callers can merge
        unconditionally.
        """
        machine = self.entry_state_machine
        if machine is None:
            return {}
        total_steps = max(1, int(self.episode_step_count))
        visits = dict(machine.visits)
        armed_bars = max(1, machine.armed_bars)
        return {
            "entry_state_machine": True,
            "entry_state_visits": visits,
            "entry_state_share": {
                state: round(count / total_steps, 6) for state, count in visits.items()
            },
            "candidate_created": int(machine.candidate_created),
            "candidate_expired": int(machine.candidate_expired),
            "candidate_invalidated": int(machine.candidate_invalidated),
            "trigger_requested": int(machine.trigger_requested),
            "trigger_refused": int(machine.trigger_refused),
            "entered": int(machine.entered),
            "closed": int(machine.closed),
            "armed_bars": int(machine.armed_bars),
            "arm_to_trigger_rate": round(machine.entered / armed_bars, 6),
            "trigger_refusal_rate": round(
                machine.trigger_refused / max(1, machine.trigger_requested), 6
            ),
        }
