"""Focused simulation initialization service."""

from __future__ import annotations

from decisioning.observations import build_observation_schema

from .entry_state import EntryStateMachine
from .mixin_base import RuntimeMixinBase


class RiskEntryInitializationMixin(RuntimeMixinBase):
    def _initialize_risk_and_entry_controls(
        self,
        episodic: bool,
        legacy_sequence_columns: bool | None,
        use_sweep_reward: bool,
        min_sl_atr_mult: float,
        min_sl_points: float,
        max_entries_per_session: int,
        risk_mode: str,
        fixed_risk_usd: float,
        open_manipulation_bars: int,
        entry_cooldown_bars: int,
        allow_ema_exit: bool,
        reward_mode: str,
        entry_intensity_threshold: float,
        peak_trailing_dd_limit: float,
        agent_direction_control: bool,
        direction_override_threshold: float,
        allow_agent_sl_mode: bool,
        sl_mode_defer_threshold: float,
        allow_agent_tp_mode: bool,
        tp_mode_defer_threshold: float,
        allow_multi_tp: bool,
        tp_breakeven_alpha: float,
        allow_simplex: bool,
        entry_state_machine: bool,
        candidate_max_age_bars: int,
        arm_requires_retest: bool,
        entry_state_observation: bool,
        observation_schema_version: str,
        observation_age_scale_bars: float,
        risk_floor: float,
    ) -> None:
        self._ny_pos = 0
        self.episodic = episodic
        if legacy_sequence_columns is True and episodic:
            raise ValueError("legacy sequence columns are restricted to non-episodic replay")
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
        # Explicit training behavior, independent of the account policy. 0
        # disables it. At the configured fraction of peak equity, flatten the
        # open position at the market quote and refuse new risk. Stops and
        # targets are not rewritten.
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
        self.observation_schema = build_observation_schema(
            observation_schema_version,
            entry_state_visibility=self.entry_state_observation,
        )
        self.observation_age_scale_bars = float(observation_age_scale_bars)
        if self.observation_age_scale_bars <= 0.0:
            raise ValueError("observation_age_scale_bars must be positive")
        self.risk_floor = 0.0 if self.repeated_payout_policy else float(risk_floor)


__all__ = ["RiskEntryInitializationMixin"]
