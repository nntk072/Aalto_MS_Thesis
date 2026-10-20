# ruff: noqa: F403, F405
from __future__ import annotations

from .runtime_dependencies import *


class InitializationRuntimeMixin(RuntimeMixinBase):
    action_space: Any
    observation_space: Any

    def _initialize_market_source(
        self,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
        bundle_data: Any | None,
        strict_research_eligibility: bool,
    ) -> None:
        self.bars: pd.DataFrame = cast(pd.DataFrame, bars)
        self.features: pd.DataFrame = cast(pd.DataFrame, features)
        self._bundle_arrays: dict[str, np.ndarray[Any, Any]] | None = (
            {name: np.asarray(array) for name, array in bundle_data.arrays.items()}
            if bundle_data is not None
            else None
        )
        if self._bundle_arrays is not None:
            self._feature_time_ms = (self._bundle_arrays["bar_time_ns"] // 1_000_000).astype(
                np.int64, copy=False
            )
        else:
            self._feature_time_ms = (
                pd.DatetimeIndex(self.features.index)
                .to_numpy(dtype="datetime64[ns]")
                .view(np.int64)
                // 1_000_000
            ).astype(np.int64, copy=False)
        bundle_labels = bundle_data.manifest.get("labels", {}) if bundle_data is not None else {}
        self._n_bars = (
            len(self._bundle_arrays["bar_time_ns"])
            if self._bundle_arrays is not None
            else len(bars)
            if bars is not None
            else 0
        )
        # Legacy permissive fallback remains available for old checkpoints.
        # Confirmatory runs can explicitly prohibit missing eligibility data.
        self.strict_research_eligibility = bool(strict_research_eligibility)
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
                if not ny_indices.size and self.strict_research_eligibility:
                    raise ValueError("Strict research eligibility: no eligible NY session bars")
                self._ny_indices = ny_indices if ny_indices.size else np.arange(self._n_bars)
            else:
                self._ny_indices = self._resolve_ny_indices(
                    pd.DataFrame(index=pd.DatetimeIndex(times))
                )
        self._bundle_has_session = bool(bundle_labels.get("has_session", False))
        self._session_labels = list(bundle_labels.get("session", []))
        self._bundle_has_session_id = bool(bundle_labels.get("has_session_id", False))

    def _initialize_account_policy(
        self,
        obs_window: int,
        initial_balance: float,
        cost_model: CostModel,
        broker_kwargs: dict[str, Any] | None,
        trading_policy: dict[str, Any] | None,
        guardrail_kwargs: dict[str, Any] | None,
        risk_frac_range: tuple[float, float],
        rr_ratio_range: tuple[float, float],
        sl_buffer_pts: float,
        sl_mode: str,
        tp_mode: str,
        tp_reference: str,
        min_lot: float,
        max_lot: float,
        contract_size: float,
        max_loss_per_trade_usd: float,
        block_overnight: bool,
        eod_risk: dict[str, Any] | None,
    ) -> None:
        self.obs_window = obs_window
        self.initial_balance = initial_balance
        self.cost_model = cost_model
        self.broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
        policy_values = dict(trading_policy or {})
        self.trading_policy = TradingPolicyConfig(**policy_values).validate()
        self.repeated_payout_policy = self.trading_policy.repeated_payout
        if (
            self.repeated_payout_policy
            and abs(float(initial_balance) - self.trading_policy.operating_balance) > 1e-9
        ):
            raise ValueError("repeated-payout initial balance must equal operating_balance")
        resolved_guardrails = dict(guardrail_kwargs or {})
        if self.repeated_payout_policy:
            resolved_guardrails.update(
                daily_loss_limit=self.trading_policy.daily_loss_limit,
                max_loss_limit=self.trading_policy.operating_balance
                - self.trading_policy.account_floor,
                risk_per_trade_limit=self.trading_policy.operating_balance
                * self.trading_policy.max_agent_risk_frac,
                trailing_dd_limit=0.0,
                soft_trailing_dd_limit=0.0,
            )
        self.guardrails = FTMOGuardrails(**resolved_guardrails)

        self.risk_frac_range = (
            (0.0, self.trading_policy.max_agent_risk_frac)
            if self.repeated_payout_policy
            else risk_frac_range
        )
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
        # Training behavior, not an FTMO kill-switch. 0 disables it.
        # At this fraction of peak equity, flatten the open position at the
        # market quote and refuse new risk. Stops and targets are not rewritten.
        self.peak_trailing_dd_limit = (
            0.0 if self.repeated_payout_policy else float(peak_trailing_dd_limit)
        )
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

    def _initialize_execution_controls(
        self,
        tickbook: TickBook | None,
        tick_execution_required: bool,
        fill_delay_ms: int,
        randomized_fill_latency_ms: float,
        max_tick_quote_wait_ms: int,
    ) -> None:
        self._tickbook = tickbook
        self.tick_execution_required = bool(tick_execution_required)
        self._fill_delay_ms = int(fill_delay_ms)
        self.randomized_fill_latency_ms = float(randomized_fill_latency_ms)
        self.max_tick_quote_wait_ms = int(max_tick_quote_wait_ms)
        if self.randomized_fill_latency_ms < 0:
            raise ValueError("randomized_fill_latency_ms cannot be negative")
        if self.randomized_fill_latency_ms > 30.0:
            raise ValueError("randomized_fill_latency_ms must be at most 30ms exclusive")
        if self.max_tick_quote_wait_ms <= 0:
            raise ValueError("max_tick_quote_wait_ms must be positive")
        if (
            self.tick_execution_required or self.randomized_fill_latency_ms > 0
        ) and tickbook is None:
            raise ValueError("tick-only execution requires a TickBook")

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

    def _initialize_strategy_runtime(
        self,
        strategy: TradingStrategy | None,
        strategy_actions: bool,
        sl_buffer_pts: float,
        bundle_data: Any | None,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
    ) -> None:
        self._entry_gate_warned: bool = False
        # Strategy semantics (Idea 1/2) vs. legacy behaviour (Idea 3 baseline).
        self.strategy: TradingStrategy = strategy if strategy is not None else BaselineStrategy()
        self.strategy_actions = strategy_actions
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
            from trading.positions.tp_reach import max_tp_distance

            dist = max_tp_distance(bars, features["atr_5"]).reindex(bars.index)
            self._max_tp_dist = dist.to_numpy(dtype=float)

    def _initialize_observation_arrays(
        self,
        bundle_data: Any | None,
        bars: pd.DataFrame | None,
        features: pd.DataFrame | None,
        legacy_sequence_columns: bool | None,
        episodic: bool,
        obs_features_mmap: str | None,
        mtf: bool,
        mtf_windows: dict[str, int] | None,
    ) -> None:
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
            self._legacy_sequence_columns = bool(legacy_sequence_columns)
            select_columns = (
                select_legacy_obs_columns if self._legacy_sequence_columns else select_obs_columns
            )
            self._obs_features: pd.DataFrame = attach_reachable_r(
                select_columns(features, self.strategy.raw_columns),
                bars,
                features,
            )
            self._obs_columns = list(self._obs_features.columns)
        else:
            assert bundle_data is not None
            self._obs_features = cast(pd.DataFrame, None)
            self._obs_columns = list(bundle_data.manifest["column_names"]["obs_features"])
            saved_legacy_columns = bool(
                bundle_data.manifest.get("config", {}).get("legacy_sequence_columns", False)
            )
            if legacy_sequence_columns is not None and bool(legacy_sequence_columns) != (
                saved_legacy_columns
            ):
                raise ValueError(
                    "requested legacy sequence-column mode differs from the saved env bundle"
                )
            self._legacy_sequence_columns = saved_legacy_columns
        if self._legacy_sequence_columns and episodic:
            raise ValueError("legacy sequence columns are restricted to non-episodic replay")
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

    def _initialize_episode_spaces(
        self,
        ny_session_start_idx: int | None,
        bars: pd.DataFrame | None,
        bundle_data: Any | None,
        use_vae: bool,
        vae: Any,
        obs_window: int,
    ) -> None:
        # Per-step entry diagnostics (strategy_actions); reset each episode.
        self._entry_diag: dict[str, int] = _empty_entry_diag(self.entry_state_machine is not None)
        self._previous_constraint_ratios: dict[str, float] = {
            "daily": 0.0,
            "max_loss": 0.0,
            "trailing": 0.0,
        }

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

        account_dim = self.observation_schema.width
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
