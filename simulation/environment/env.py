from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from decisioning.observations import OBSERVATION_SCHEMA_V1

from market_data.loading.ticks import TickBook
from simulation.contracts import SimulationTransition
from simulation.environment.diagnostics import DebugStepError
from trading.execution.broker import Position
from trading.execution.costs import COST_US100, CostModel
from trading.strategies import TradingStrategy

from .action_diagnostics import _empty_decision
from .action_runtime import ActionDecodingMixin
from .diagnostics_runtime import DiagnosticsRuntimeMixin
from .entry_diagnostics import _empty_entry_diag
from .entry_execution_runtime import EntryExecutionRuntimeMixin
from .entry_runtime import EntryRuntimeMixin
from .episode_runtime import EpisodeRuntimeMixin
from .execution_runtime import ExecutionRuntimeMixin
from .initialization_runtime import InitializationRuntimeMixin
from .market_access import MarketAccessMixin
from .observation_runtime import ObservationRuntimeMixin
from .position_runtime import PositionRuntimeMixin
from .risk_runtime import RiskRuntimeMixin
from .step_runtime import StepRuntimeMixin

__all__ = ["TradingEnv"]


class TradingEnv(
    InitializationRuntimeMixin,
    MarketAccessMixin,
    RiskRuntimeMixin,
    ExecutionRuntimeMixin,
    ActionDecodingMixin,
    EntryExecutionRuntimeMixin,
    EntryRuntimeMixin,
    DiagnosticsRuntimeMixin,
    StepRuntimeMixin,
    EpisodeRuntimeMixin,
    PositionRuntimeMixin,
    ObservationRuntimeMixin,
    gym.Env[Any, np.ndarray[Any, Any]],
):
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
        vae: Any | None = None,
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
        tick_execution_required: bool = False,
        fill_delay_ms: int = 0,
        randomized_fill_latency_ms: float = 0.0,
        max_tick_quote_wait_ms: int = 60_000,
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
        observation_schema_version: str = OBSERVATION_SCHEMA_V1,
        observation_age_scale_bars: float = 60.0,
        bundle_data: Any | None = None,
        strict_research_eligibility: bool = False,
        legacy_sequence_columns: bool | None = None,
        trading_policy: dict[str, Any] | None = None,
    ):
        """Initialize the trading environment through semantic runtime stages.

        The public constructor remains stable for saved configs and checkpoints while
        initialization mechanics live outside the Gymnasium facade.
        """
        self._initialize_market_source(bars, features, bundle_data, strict_research_eligibility)
        self._initialize_account_policy(
            obs_window,
            initial_balance,
            cost_model,
            broker_kwargs,
            trading_policy,
            guardrail_kwargs,
            risk_frac_range,
            rr_ratio_range,
            sl_buffer_pts,
            sl_mode,
            tp_mode,
            tp_reference,
            min_lot,
            max_lot,
            contract_size,
            max_loss_per_trade_usd,
            block_overnight,
            eod_risk,
        )
        self._initialize_risk_and_entry_controls(
            episodic,
            legacy_sequence_columns,
            use_sweep_reward,
            min_sl_atr_mult,
            min_sl_points,
            max_entries_per_session,
            risk_mode,
            fixed_risk_usd,
            open_manipulation_bars,
            entry_cooldown_bars,
            allow_ema_exit,
            reward_mode,
            entry_intensity_threshold,
            peak_trailing_dd_limit,
            agent_direction_control,
            direction_override_threshold,
            allow_agent_sl_mode,
            sl_mode_defer_threshold,
            allow_agent_tp_mode,
            tp_mode_defer_threshold,
            allow_multi_tp,
            tp_breakeven_alpha,
            allow_simplex,
            entry_state_machine,
            candidate_max_age_bars,
            arm_requires_retest,
            entry_state_observation,
            observation_schema_version,
            observation_age_scale_bars,
            risk_floor,
        )
        self._initialize_execution_controls(
            tickbook,
            tick_execution_required,
            fill_delay_ms,
            randomized_fill_latency_ms,
            max_tick_quote_wait_ms,
        )
        resolved_strategy_actions = self._initialize_reward_runtime(
            trader_actions,
            strategy_actions,
            use_sweep_reward,
            strategy_reward,
            dsr_eta,
            sweep_alpha,
            sweep_beta,
            sweep_hold_bars,
            dsr_weight,
            sweep_weight,
            strategy_weight,
        )
        self._initialize_strategy_runtime(
            strategy, resolved_strategy_actions, sl_buffer_pts, bundle_data, bars, features
        )
        self._initialize_observation_arrays(
            bundle_data,
            bars,
            features,
            legacy_sequence_columns,
            episodic,
            obs_features_mmap,
            mtf,
            mtf_windows,
        )
        self._initialize_action_and_episode_contracts(
            continuous_actions,
            max_risk_frac,
            use_vae,
            vae,
            pre_ny_by_date,
            pre_ny_data,
            fill_latency_bars,
            max_episode_steps,
            normalize_account,
        )
        self._initialize_episode_spaces(
            ny_session_start_idx, bars, bundle_data, use_vae, vae, obs_window
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
        self._pending_entry_order: dict[str, Any] | None = None
        self.equity_curve = [self.initial_balance]
        # Wall-clock stamps for each equity_curve point (NY-stepped; not dense M1).
        t0 = self._bar_time(self.step_idx) if len(self._bar_times) else None
        self.equity_times: list[Any] = [t0]
        self.pnl_history = [0.0]
        self.trade_log: list[dict[str, Any]] = []
        self._active_step_execution: dict[str, Any] | None = None
        self._position_sequence = 0
        self._episode_identity = getattr(self, "_episode_identity", 0) + 1
        # Step counter within the current episode (0 at reset). Distinct
        # from ``step_idx`` (the absolute bar index); used to enforce
        # ``max_episode_steps`` truncation.
        self.episode_step_count = 0
        # Warn-once flag for the entry gate; re-arm per episode so a new
        # feature matrix (possibly missing gate columns) warns again.
        self._entry_gate_warned = False

        # Reset reward function
        self.reward_fn.reset()

        # Non-episodic full-split bookkeeping, used by evaluation and
        # full-split training; unused in risk-terminated training mode.
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
        self._previous_constraint_ratios = {"daily": 0.0, "max_loss": 0.0, "trailing": 0.0}
        self.cumulative_payouts = 0.0
        self.cycle_count = 1
        self.cycle_successes = 0
        self.cycle_failures = 0
        self.cycle_events: list[dict[str, Any]] = []
        self.payout_events: list[dict[str, Any]] = []
        self._last_requested_risk_usd = 0.0
        self._last_approved_risk_usd = 0.0
        self._last_execution_reserve_usd = 0.0
        self._payout_withdrawn_this_step = 0.0

        if self.entry_state_machine is not None:
            self.entry_state_machine.reset()

        obs = self._get_observation()
        return obs, {}

    def step(
        self,
        action: int | float | np.ndarray[Any, Any],
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        """Gymnasium adapter for :meth:`transition`."""
        return self.transition(action).as_gymnasium()

    def transition(
        self,
        action: int | float | np.ndarray[Any, Any],
    ) -> SimulationTransition[dict[str, np.ndarray[Any, Any]]]:
        """Advance the simulator through its framework-neutral transition API."""
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
