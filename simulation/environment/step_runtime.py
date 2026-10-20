# ruff: noqa: F403, F405
from __future__ import annotations

from .runtime_dependencies import *


class StepRuntimeMixin(RuntimeMixinBase):
    position: Position | None
    _pending_entry_order: dict[str, Any] | None
    _decision_action: dict[str, float]
    _previous_constraint_ratios: dict[str, float]
    _opened_this_step: bool
    episode_step_count: int
    _ep_reward_sum: float

    def _prepare_action_intent(
        self,
        action: int | float | np.ndarray[Any, Any],
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
        pending_order: dict[str, Any] | None,
    ) -> tuple[Any, str | None, int, float, float, str, EntryCandidate | None]:
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

        if pending_order is not None:
            if discrete_action != 0:
                self._reject("entry_pending")
            discrete_action = 0
            risk_frac = 0.0
            entry_candidate = None

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

        if self.block_overnight and discrete_action != 0 and fill_idx < self._n_bars:
            fill_ny_pos = int(
                np.searchsorted(np.asarray(self._ny_indices, dtype=np.int64), int(fill_idx))
            )
            has_ny_fill = (
                fill_ny_pos < len(self._ny_indices)
                and int(self._ny_indices[fill_ny_pos]) == fill_idx
            )
            has_same_session_followup = False
            if has_ny_fill and fill_ny_pos + 1 < len(self._ny_indices):
                next_ny_idx = int(self._ny_indices[fill_ny_pos + 1])
                fill_session = (
                    int(self._session_ids_arr[fill_idx])
                    if self._session_ids_arr is not None
                    else session_id
                )
                next_session = (
                    int(self._session_ids_arr[next_ny_idx])
                    if self._session_ids_arr is not None
                    else fill_session
                )
                has_same_session_followup = fill_session == next_session
            if not has_ny_fill or not has_same_session_followup:
                discrete_action = 0
                risk_frac = 0.0
                if self.strategy_actions:
                    self._reject("overnight_boundary")

        # Check entry gate for new positions
        if discrete_action != 0:  # Only check for long/short entries
            if entry_machine is not None and entry_state_for_action == "ARMED":
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
                if bool(getattr(self.strategy, "previous_session_pd_gate", False)):
                    pd_eligible = getattr(self.strategy, "previous_session_pd_eligible")
                    gate_ok = gate_ok and bool(pd_eligible(feat_row, discrete_action))
            elif self.strategy_actions:
                gate_ok = self.strategy.validate_entry(
                    direction=discrete_action,
                    row=feat_row,
                )
            else:
                gate_ok = self._check_entry_gate(float(bar.close), discrete_action, feat_row)
            if not gate_ok:
                # Gate not satisfied, force to hold
                rejected_direction = discrete_action
                discrete_action = 0
                risk_frac = 0.0
                if self.strategy_actions:
                    reason_getter = getattr(self.strategy, "entry_rejection_reason", None)
                    reject_reason = (
                        str(reason_getter(direction=rejected_direction, row=feat_row))
                        if callable(reason_getter)
                        else "entry_gate"
                    )
                    self._reject(reject_reason)

        return (
            entry_machine,
            entry_state_for_action,
            discrete_action,
            risk_frac,
            rr_ratio,
            tp_mode,
            entry_candidate,
        )

    def _enforce_step_constraints(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any,
        pending_due: bool,
        entry_fill_bid: float,
        entry_fill_ask: float,
        entry_fill_time: Any,
        market_fill_idx: int,
        entry_machine: Any,
    ) -> tuple[bool, bool, bool, bool]:
        # Check guardrails
        reason, session_blocked, done, truncated = self._check_guardrails(
            bar_time, session_id, fill_bid, fill_ask, fill_time=fill_time
        )

        position_has_completed_bar = True
        if self.position is not None and self.position.entry_timestamp is not None:
            bar_end = bar_time + pd.Timedelta(minutes=self._minutes_per_step)
            position_has_completed_bar = (
                pd.Timestamp(cast(Any, self.position.entry_timestamp)) < bar_end
            )
        if not reason and not session_blocked and position_has_completed_bar:
            # A bar-close entry is effective at this bar's end. Its earlier
            # high/low cannot trigger a stop/target or position guard.
            self._check_sl_tp(bar, fill_bid, fill_ask)
            self._apply_position_guards(
                bar,
                bar_time,
                fill_bid,
                fill_ask,
                eod=False,
                feat_row=feat_row,
                bar_idx=self.step_idx,
                fill_time=fill_time,
            )

        self._peak_dd_fired = False
        peak_dd_block = self._apply_peak_trailing_dd(
            fill_bid, fill_ask, bar_time, fill_time=fill_time
        )

        # A delayed entry becomes active only when its scheduled quote is the
        # market bar being processed. It cannot be marked or stopped earlier.
        if pending_due:
            pending = self._pending_entry_order
            self._pending_entry_order = None
            pending_soft_block = self.guardrails.check_soft_daily(
                self.account
            ) or self.guardrails.check_soft_trailing_dd(self.account)
            if (
                pending is not None
                and not done
                and not session_blocked
                and not peak_dd_block
                and not pending_soft_block
            ):
                current_decision = self._decision_action
                selected_names = (
                    "_selected_sl_anchor",
                    "_selected_tp_fraction",
                    "_selected_exit_mode",
                    "_selected_sl_mode",
                    "_sl_mode_explicit",
                    "_selected_tp_mode",
                    "_tp_mode_explicit",
                    "_selected_tp_selections",
                    "_selected_tp_z",
                )
                current_selected = {name: getattr(self, name) for name in selected_names}
                self._decision_action = dict(pending["decision_action"])
                for name, value in pending["selected"].items():
                    setattr(self, name, value)
                self._try_enter_position(
                    int(pending["direction"]),
                    float(pending["risk_frac"]),
                    float(pending["rr_ratio"]),
                    bar,
                    pending["feat_row"],
                    pending["decision_time"],
                    session_id,
                    entry_fill_bid,
                    entry_fill_ask,
                    entry_fill_time,
                    entry_candidate=pending["entry_candidate"],
                    fill_bar_idx=market_fill_idx,
                    order_submission_time=pending["order_submission_time"],
                    decision_bar_idx=int(pending["decision_bar_idx"]),
                    signal_bar_idx=int(pending["signal_bar_idx"]),
                )
                self._decision_action = current_decision
                for name, value in current_selected.items():
                    setattr(self, name, value)
                if entry_machine is not None and self._opened_this_step:
                    entry_machine.mark_entered()
            elif pending is not None:
                self.trade_log.append(
                    {
                        "type": "order_cancel",
                        "reason": "execution_blocked_before_scheduled_fill",
                        "decision_time": pending["decision_time"],
                        "order_submission_time": pending["order_submission_time"],
                        "time": fill_time,
                        "fill_time": None,
                        "bar": self.step_idx,
                    }
                )

        return done, truncated, session_blocked, peak_dd_block

    def _execute_step_action(
        self,
        action: int | float | np.ndarray[Any, Any],
        discrete_action: int,
        risk_frac: float,
        rr_ratio: float,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
        action_fill_time: Any,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any,
        entry_fill_bid: float,
        entry_fill_ask: float,
        entry_fill_time: Any,
        market_fill_idx: int,
        entry_candidate: EntryCandidate | None,
        entry_machine: Any,
        done: bool,
        pending_due: bool,
        session_blocked: bool,
        peak_dd_block: bool,
    ) -> None:
        # Action handling
        if not done and not pending_due:
            if not self.strategy_actions and action == 19:  # exit action
                if self.position is not None:
                    execution_meta = {}
                    action_fill_bid, action_fill_ask = fill_bid, fill_ask
                    action_fill_time = fill_time
                    if getattr(self, "tick_execution_required", False):
                        quote, execution_meta = self._resolve_randomized_tick_quote(
                            pd.Timestamp(fill_time)
                        )
                        action_fill_bid, action_fill_ask = quote.bid, quote.ask
                        action_fill_time = quote.tick_time
                    pnl, fill_price = self.broker.close_position(
                        self.account, self.position, (action_fill_bid, action_fill_ask)
                    )
                    record = {
                        "type": "close",
                        "pnl": pnl,
                        "price": fill_price,
                        "fill_quote_source": "tick" if execution_meta else "bar_open",
                        "bar": self.step_idx,
                        "time": action_fill_time,
                        "decision_time": bar_time,
                        "order_submission_time": bar_time,
                        "fill_time": action_fill_time,
                        "equity": self.account.equity,
                        **execution_meta,
                    }
                    self._stamp_exit(record, fill_price)
                    self.trade_log.append(record)
                    self.position = None
                    self.sessions_with_trades.add(session_id)
                    self._note_close(pnl)
            elif discrete_action != 0 and fill_idx >= self._n_bars:
                self._reject("terminal_no_fill_bar")
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
                    if self.fill_latency_bars > 0:
                        stored_feat_row: FeatureRow | pd.Series = feat_row
                        if isinstance(feat_row, pd.Series):
                            stored_feat_row = feat_row.copy()
                        selected_names = (
                            "_selected_sl_anchor",
                            "_selected_tp_fraction",
                            "_selected_exit_mode",
                            "_selected_sl_mode",
                            "_sl_mode_explicit",
                            "_selected_tp_mode",
                            "_tp_mode_explicit",
                            "_selected_tp_selections",
                            "_selected_tp_z",
                        )
                        self._pending_entry_order = {
                            "fill_idx": fill_idx,
                            "direction": discrete_action,
                            "risk_frac": risk_frac,
                            "rr_ratio": rr_ratio,
                            "feat_row": stored_feat_row,
                            "decision_time": bar_time,
                            "decision_bar_idx": self.step_idx,
                            "signal_bar_idx": self.step_idx,
                            "signal_time": bar_time + pd.Timedelta(minutes=self._minutes_per_step),
                            "order_submission_time": bar_time
                            + pd.Timedelta(minutes=self._minutes_per_step),
                            "entry_candidate": entry_candidate,
                            "decision_action": dict(self._decision_action),
                            "selected": {name: getattr(self, name) for name in selected_names},
                        }
                        self.trade_log.append(
                            {
                                "type": "order_submit",
                                "direction": discrete_action,
                                "time": bar_time,
                                "decision_time": bar_time,
                                "signal_bar": self.step_idx,
                                "signal_time": bar_time
                                + pd.Timedelta(minutes=self._minutes_per_step),
                                "order_submission_bar": self.step_idx,
                                "order_submission_time": bar_time
                                + pd.Timedelta(minutes=self._minutes_per_step),
                                "expected_fill_time": action_fill_time,
                                "fill_time": None,
                                "bar": self.step_idx,
                            }
                        )
                    else:
                        self._try_enter_position(
                            discrete_action,
                            risk_frac,
                            rr_ratio,
                            bar,
                            feat_row,
                            bar_time,
                            session_id,
                            entry_fill_bid,
                            entry_fill_ask,
                            entry_fill_time,
                            entry_candidate=entry_candidate,
                            fill_bar_idx=market_fill_idx,
                            order_submission_time=bar_time
                            + pd.Timedelta(minutes=self._minutes_per_step),
                            decision_bar_idx=self.step_idx,
                            signal_bar_idx=self.step_idx,
                        )
                        if entry_machine is not None and self._opened_this_step:
                            entry_machine.mark_entered()

    def _record_step_accounting(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        step_trade_log_start: int,
        done: bool,
        truncated: bool,
    ) -> tuple[bool, Any, float, float, dict[str, float]]:
        payout_terminal = self._handle_repeated_payout(bar_time)
        if payout_terminal:
            done = True

        valuation_time = bar_time + pd.Timedelta(minutes=self._minutes_per_step)
        for event in self.trade_log[step_trade_log_start:]:
            event_time = event.get("fill_time")
            if event_time is None and event.get("time_precision") == "bar_interval":
                event_time = event.get("event_time_end")
            if event_time is not None:
                event_timestamp = pd.Timestamp(event_time)
                if event_timestamp.tzinfo is None:
                    event_timestamp = event_timestamp.tz_localize("UTC")
                event_timestamp = event_timestamp.tz_convert(BAR_TIME_TZ)
                if event_timestamp > valuation_time:
                    valuation_time = event_timestamp
        self.equity_curve.append(self.account.equity)
        self.equity_times.append(valuation_time)
        pnl_step = (
            self.account.equity + float(self._payout_withdrawn_this_step) - self.equity_curve[-2]
        )
        self.pnl_history.append(pnl_step)

        reward = self._calculate_reward(pnl_step, bar, feat_row, done, truncated)
        reward_parts = {
            key: float(value) for key, value in getattr(self.reward_fn, "last_parts", {}).items()
        }
        if self.peak_trailing_dd_limit > 0.0 and self._peak_dd_fired:
            # Penalty each bar the training peak-DD ceiling is still hit.
            reward -= 1.0
            reward_parts["peak_dd"] = float(reward_parts.get("peak_dd", 0.0)) - 1.0
        return done, valuation_time, pnl_step, float(reward), reward_parts

    def _update_entry_funnel(
        self,
        feat_row: FeatureRow | pd.Series,
        entry_machine: Any,
        entry_diag_before: dict[str, int],
    ) -> dict[str, int]:
        entry_funnel: dict[str, int] = {"bars_seen": 1}
        if entry_machine is not None:
            prior_state = entry_machine.state
            prior_candidates = entry_machine.candidate_created
            evidence = self.entry_evidence_context(feat_row)
            entry_machine.observe(evidence)
            if entry_machine.state == "IN_POSITION" and self.position is None:
                entry_machine.mark_closed()
            entry_funnel.update(
                {
                    "strategy_context_available": int(evidence.direction in (-1, 1)),
                    "entry_evidence": int(evidence.evidence_detected),
                    "candidate_created": int(entry_machine.candidate_created > prior_candidates),
                    "candidate_armed": int(
                        entry_machine.state == "ARMED" and prior_state != "ARMED"
                    ),
                    "trigger_requested": int(self._trigger_requested_this_step),
                    "hold_low_intensity": int(
                        self._entry_diag["hold_low_intensity"]
                        > entry_diag_before.get("hold_low_intensity", 0)
                    ),
                    "hold_no_context": int(
                        self._entry_diag["hold_no_context"]
                        > entry_diag_before.get("hold_no_context", 0)
                    ),
                    "hold_risk_floor": int(
                        self._entry_diag["hold_risk_floor"]
                        > entry_diag_before.get("hold_risk_floor", 0)
                    ),
                    "entry_rejected": int(bool(self._entry_rejection_reason)),
                    "position_opened": int(self._opened_this_step),
                }
            )
        return entry_funnel

    def _build_step_info(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        valuation_time: Any,
        reward_parts: dict[str, float],
        entry_funnel: dict[str, int],
        entry_machine: Any,
        entry_state_for_action: str | None,
        done: bool,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], str, dict[str, Any]]:
        obs = self._get_observation()
        close_reason = ""
        if self.trade_log:
            last_trade = self.trade_log[-1]
            if last_trade.get("bar") == self.step_idx and last_trade.get("type") not in (
                "open",
                "reject",
                "sl_update",
            ):
                close_reason = str(last_trade.get("reason") or "")
        if self._entry_rejection_reason and not self._opened_this_step:
            self.trade_log.append(
                {
                    "type": "reject",
                    "entry_rejection_reason": self._entry_rejection_reason,
                    "bar": self.step_idx,
                    "time": bar_time,
                    **(
                        _previous_session_audit(self.strategy, feat_row, float(bar.close))
                        if feat_row is not None
                        else {}
                    ),
                    **({"entry_state": entry_machine.state} if entry_machine is not None else {}),
                    **self._decision_action,
                }
            )
        info: dict[str, Any] = {
            "equity": self.account.equity,
            "valuation_time": valuation_time,
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
            "entry_funnel": entry_funnel,
        }
        if self.repeated_payout_policy:
            info.update(
                {
                    "trading_policy_mode": self.trading_policy.mode,
                    "daily_loss": float(self.account.canonical_daily_loss()),
                    "daily_floor": float(
                        self.account.daily_floor(self.guardrails.daily_loss_limit)
                    ),
                    "remaining_daily_capacity": float(
                        self.account.remaining_daily_capacity(self.guardrails.daily_loss_limit)
                    ),
                    "remaining_account_capacity": float(
                        self.account.remaining_account_capacity(self.trading_policy.account_floor)
                    ),
                    "cumulative_payouts": float(self.cumulative_payouts),
                    "total_economic_value": float(self.account.equity + self.cumulative_payouts),
                    "cycle_count": int(self.cycle_count),
                    "cycle_successes": int(self.cycle_successes),
                    "cycle_failures": int(self.cycle_failures),
                    "requested_risk_usd": float(self._last_requested_risk_usd),
                    "approved_risk_usd": float(self._last_approved_risk_usd),
                    "execution_reserve_usd": float(self._last_execution_reserve_usd),
                }
            )
        constraint_ratios = {
            "daily": (
                self.account.canonical_daily_loss() / self.guardrails.daily_loss_limit
                if self.guardrails.daily_loss_limit > 0.0
                else 0.0
            ),
            "max_loss": (
                self.account.loss_from_initial() / self.guardrails.max_loss_limit
                if self.guardrails.max_loss_limit > 0.0
                else 0.0
            ),
            "trailing": (
                self.account.trailing_drawdown_pct() / self.guardrails.trailing_dd_limit
                if self.guardrails.trailing_dd_limit > 0.0
                else 0.0
            ),
        }
        breach_reason = str(self.breach_log[-1]) if done and self.breach_log else ""
        daily_breach = breach_reason == "daily_loss"
        max_loss_breach = breach_reason == "max_drawdown"
        trailing_breach = breach_reason == "trailing_dd"
        info["constraints"] = {
            "daily_ratio": float(constraint_ratios["daily"]),
            "max_loss_ratio": float(constraint_ratios["max_loss"]),
            "trailing_ratio": float(constraint_ratios["trailing"]),
            "daily_cost": max(
                0.0, constraint_ratios["daily"] - self._previous_constraint_ratios["daily"]
            )
            + float(daily_breach),
            "max_loss_cost": max(
                0.0,
                constraint_ratios["max_loss"] - self._previous_constraint_ratios["max_loss"],
            )
            + float(max_loss_breach),
            "trailing_cost": max(
                0.0,
                constraint_ratios["trailing"] - self._previous_constraint_ratios["trailing"],
            )
            + float(trailing_breach),
            "daily_breach": daily_breach,
            "max_loss_breach": max_loss_breach,
            "trailing_breach": trailing_breach,
            "breach_reason": breach_reason,
            "previous_daily_ratio": self._previous_constraint_ratios["daily"],
            "previous_max_loss_ratio": self._previous_constraint_ratios["max_loss"],
            "previous_trailing_ratio": self._previous_constraint_ratios["trailing"],
        }
        self._previous_constraint_ratios = constraint_ratios
        if entry_machine is not None:
            info["entry_state"] = entry_machine.state
            info["entry_state_action"] = entry_state_for_action
        prior_open = self._prior_open_payload()
        if prior_open is not None:
            info["prior_open"] = prior_open
        risk_u = self._decision_action.get("risk_u")
        if isinstance(risk_u, (int, float)) and np.isfinite(risk_u):
            info["risk_u"] = float(risk_u)
        return obs, close_reason, info

    def _resolve_step_end(
        self,
        action: int | float | np.ndarray[Any, Any],
        discrete_action: int,
        risk_frac: float,
        entry_machine: Any,
        valuation_time: Any,
        close_reason: str,
        reward: float,
        reward_parts: dict[str, float],
        obs: dict[str, np.ndarray[Any, Any]],
        info: dict[str, Any],
        done: bool,
        truncated: bool,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        self._advance_ny_step()
        if entry_machine is not None and entry_machine.state == "IN_POSITION":
            if self.position is None:
                entry_machine.mark_closed()
                info["entry_state"] = entry_machine.state
                if self.entry_state_observation:
                    entry_start = self.observation_schema.width - ENTRY_STATE_DIM
                    obs["account"][entry_start:] = np.asarray(
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

        if (done or truncated) and self._pending_entry_order is not None:
            pending = self._pending_entry_order
            self._pending_entry_order = None
            self.trade_log.append(
                {
                    "type": "order_cancel",
                    "reason": "episode_ended_before_scheduled_fill",
                    "decision_time": pending["decision_time"],
                    "order_submission_time": pending["order_submission_time"],
                    "time": valuation_time,
                    "fill_time": None,
                    "bar": self.step_idx,
                }
            )

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
            n_trades = len([t for t in self.trade_log if t.get("type") == "open"])
            gate = assess_train_equity(
                pd.Series(self.equity_curve, dtype=float),
                initial=float(self._ep_start_equity),
                breached=bool(self.breach_log),
            )
            completed_data_split = self._ny_pos >= len(self._ny_indices)
            if done:
                if self.repeated_payout_policy and self.cycle_successes > 0 and not self.breach_log:
                    episode_end_reason = "payout_success"
                else:
                    episode_end_reason = "risk_breach" if self.breach_log else "terminated"
            elif completed_data_split:
                episode_end_reason = "data_end"
            elif self.max_episode_steps is not None and (
                self.episode_step_count >= self.max_episode_steps
            ):
                episode_end_reason = "max_episode_steps"
            else:
                episode_end_reason = "truncated"
            info["episode_equity"] = {
                "end_equity": float(self.account.equity),
                "start_equity": float(self._ep_start_equity),
                "reward_sum": float(self._ep_reward_sum),
                "n_trades": int(n_trades),
                "episode_steps": int(self.episode_step_count),
                "completed_data_split": bool(completed_data_split),
                "episode_end_reason": episode_end_reason,
                "breach_reason": str(self.breach_log[-1]) if self.breach_log else "",
                "slope": float(gate["slope"]),
                "max_peak_trailing_dd": float(gate["max_peak_trailing_dd"]),
            }

        return obs, float(reward), done, truncated, info
