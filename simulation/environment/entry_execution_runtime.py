# ruff: noqa: F403, F405
from __future__ import annotations

from .runtime_dependencies import *


class EntryExecutionRuntimeMixin(RuntimeMixinBase):
    position: Position | None
    _opened_this_step: bool
    _position_sequence: int

    def _resolve_entry_execution_quote(
        self,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any,
        order_submission_time: Any | None,
    ) -> tuple[float, float, Any, dict[str, Any]]:
        execution_meta: dict[str, Any] = {}
        if (
            getattr(self, "tick_execution_required", False)
            or getattr(self, "randomized_fill_latency_ms", 0.0) > 0
        ):
            active_execution = getattr(self, "_active_step_execution", None)
            if active_execution is not None:
                execution_quote = active_execution["quote"]
                execution_meta = dict(active_execution["metadata"])
                execution_order_time = execution_quote.order_time
            else:
                execution_order_time = (
                    pd.Timestamp(fill_time)
                    if self.fill_latency_bars > 0
                    else pd.Timestamp(
                        fill_time if order_submission_time is None else order_submission_time
                    )
                )
                execution_quote, execution_meta = self._resolve_randomized_tick_quote(
                    execution_order_time
                )
            fill_bid, fill_ask = execution_quote.bid, execution_quote.ask
            fill_time = execution_quote.tick_time
            execution_meta["execution_order_time"] = execution_order_time
        return fill_bid, fill_ask, fill_time, execution_meta

    def _close_opposing_position(
        self,
        discrete_action: int,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any,
        bar_time: Any,
        order_submission_time: Any | None,
        session_id: int,
        execution_meta: dict[str, Any],
    ) -> None:
        if self.position is not None and self.position.direction != discrete_action:
            pnl, fill_price = self.broker.close_position(
                self.account, self.position, (fill_bid, fill_ask)
            )
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "fill_quote_source": "tick" if execution_meta else "bar_open",
                "bar": self.step_idx,
                "time": fill_time,
                "decision_time": bar_time,
                "order_submission_time": (
                    bar_time if order_submission_time is None else order_submission_time
                ),
                "fill_time": fill_time,
                "equity": self.account.equity,
                **execution_meta,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(session_id)
            self._note_close(pnl)

    def _build_entry_levels(
        self,
        discrete_action: int,
        risk_frac: float,
        rr_ratio: float,
        feat_row: FeatureRow | pd.Series,
        fill_bid: float,
        fill_ask: float,
    ) -> tuple[float, float | None, float | None, float, bool, Any]:
        sl_price = None
        tp_price = None

        last_swing_low = (
            float(feat_row["last_swing_low"])
            if "last_swing_low" in feat_row.index and np.isfinite(float(feat_row["last_swing_low"]))
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
            from trading.orders.entries import place_strategy_order

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
                    self._selected_tp_z if (self.allow_multi_tp and self.allow_simplex) else None
                ),
                allow_multi_tp=self.allow_multi_tp,
                allow_simplex=self.allow_simplex,
                price_increment=float(self.cost_model.point_size),
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
        elif discrete_action == 1 and not np.isnan(last_swing_low) and last_swing_low < entry_price:
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
        return entry_price, sl_price, tp_price, lots, has_levels, placed

    def _entry_risk_capacity_allows(
        self,
        entry_price: float,
        sl_price: float | None,
        lots: float,
        has_levels: bool,
    ) -> bool:
        if has_levels and self.repeated_payout_policy:
            assert sl_price is not None
            approved_risk = float(self._last_approved_risk_usd)
            modeled_risk = abs(entry_price - float(sl_price)) * float(lots) * self.contract_size
            modeled_execution = self._modeled_execution_reserve_usd(lots)
            self._last_execution_reserve_usd = (
                float(self.trading_policy.execution_reserve_usd) + modeled_execution
            )
            if lots <= 0.0 or approved_risk <= 0.0:
                self._reject("risk_capacity")
                return False
            if modeled_risk + modeled_execution > approved_risk + 1e-9:
                self._reject("risk_rounding_cap")
                return False
        return True

    def _open_entry_position(
        self,
        discrete_action: int,
        fill_bid: float,
        fill_ask: float,
        lots: float,
        sl_price: float | None,
    ) -> tuple[float, bool]:
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
            self.position = None
            self._reject("sl_already_hit")
        return entry_equity, self.position is not None

    def _initialize_open_position(
        self,
        discrete_action: int,
        risk_frac: float,
        rr_ratio: float,
        entry_price: float,
        sl_price: float | None,
        tp_price: float | None,
        placed: Any,
        feat_row: FeatureRow | pd.Series,
        fill_time: Any,
        session_id: int,
        entry_candidate: EntryCandidate | None,
        bar_time: Any,
        signal_bar_idx: int | None,
    ) -> tuple[str | None, float, int, pd.Timestamp]:
        assert self.position is not None
        self._entry_diag["opened"] += 1
        self._position_sequence += 1
        self.position.position_id = f"e{self._episode_identity}:p{self._position_sequence}"
        self.position.initial_size = float(self.position.size)
        self.position.sl_price = sl_price
        self.position.sl_initial_price = sl_price
        self.position.breakeven_done = False
        if sl_price is not None:
            # Loss in account currency if the structural stop fills.
            # The intraday adverse guard uses this as its floor so
            # sizing and the guard cannot disagree.
            self.position.planned_risk_usd = (
                abs(entry_price - float(sl_price)) * float(self.position.size) * self.contract_size
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
                    from trading.orders.entries import _target_menu

                    sl_d = abs(entry_price - float(sl_price)) if sl_price is not None else 0.0
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
        self.position.entry_timestamp = fill_time
        atr_val = feat_row.get("atr_5", feat_row.get("atr", np.nan))
        self.position.entry_atr = (
            float(atr_val) if pd.notna(atr_val) and float(atr_val) > 0 else 1.0
        )
        if placed is not None and placed.sl_ref == "sl_reference":
            ref = None
            if hasattr(self.strategy, "sl_reference"):
                ref = self.strategy.sl_reference(direction=discrete_action, row=feat_row)
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
                    pd.Timestamp(cast(Any, bar_time)) - pd.Timestamp(cast(Any, cross_time))
                ).total_seconds()
            )
        )
        self._opened_this_step = True
        signal_bar = self.step_idx if signal_bar_idx is None else int(signal_bar_idx)
        signal_time = pd.Timestamp(cast(Any, bar_time)) + pd.Timedelta(
            minutes=self._minutes_per_step
        )
        return entry_level, sweep_delay, signal_bar, signal_time

    def _append_open_trade(
        self,
        *,
        discrete_action: int,
        risk_frac: float,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_time: Any,
        sl_price: float | None,
        tp_price: float | None,
        placed: Any,
        entry_price: float,
        entry_equity: float,
        execution_meta: dict[str, Any],
        entry_candidate: EntryCandidate | None,
        fill_bar_idx: int | None,
        order_submission_time: Any | None,
        decision_bar_idx: int | None,
        signal_bar: int,
        signal_time: pd.Timestamp,
        entry_level: str | None,
        sweep_delay: float,
    ) -> None:
        assert self.position is not None
        self.trade_log.append(
            {
                "type": "open",
                "position_id": self.position.position_id,
                "requested_risk_usd": float(self._last_requested_risk_usd),
                "approved_risk_usd": float(self._last_approved_risk_usd),
                "execution_reserve_usd": float(self._last_execution_reserve_usd),
                "modeled_stop_risk_usd": float(self.position.planned_risk_usd or 0.0),
                "strategy": self.strategy.name,
                "variant_id": str(getattr(self.strategy, "variant_id", self.strategy.name)),
                "direction": discrete_action,
                "price": self.position.entry_price,
                "bar": self.step_idx if decision_bar_idx is None else decision_bar_idx,
                "signal_bar": signal_bar,
                "signal_time": signal_time,
                "order_submission_bar": signal_bar,
                "fill_bar": self.step_idx if fill_bar_idx is None else fill_bar_idx,
                "fill_phase": "tick" if execution_meta else "bar_open",
                "fill_quote_source": "tick" if execution_meta else "bar_open",
                "entry_mode": str(getattr(self.strategy, "entry_mode", "legacy")),
                "time": fill_time,
                "decision_time": bar_time,
                "order_submission_time": (
                    signal_time if order_submission_time is None else order_submission_time
                ),
                "fill_time": fill_time,
                "lots": self.position.size,
                "sl_price": sl_price,
                "tp_price": tp_price,
                **_previous_session_audit(self.strategy, feat_row, float(bar.close)),
                "executed_target_prices": [
                    float(price)
                    for price in (
                        self.position.tp1_price,
                        self.position.tp2_price,
                        self.position.tp3_price,
                    )
                    if price is not None
                ]
                or ([float(tp_price)] if tp_price is not None else []),
                "tp_anchor_ref_price": self.position.entry_tp_ref_price,
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
                "sl_menu": ([] if placed is None else [list(pair) for pair in placed.sl_menu]),
                "sl_candidate_audit": (
                    []
                    if placed is None
                    else [dict(candidate) for candidate in placed.sl_candidate_audit]
                ),
                "tp_menu": ([] if placed is None else [list(pair) for pair in placed.tp_menu]),
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
                "equity": self.account.equity,
                **execution_meta,
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
                "manipulation_high": float(feat_row.get("po3_manipulation_high", float("nan"))),
                "manipulation_low": float(feat_row.get("po3_manipulation_low", float("nan"))),
                "manipulation_end": float(feat_row.get("po3_manipulation_end", float("nan"))),
                "distribution_phase": float(feat_row.get("po3_distribution", float("nan"))),
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
        fill_time: Any,
        entry_candidate: EntryCandidate | None = None,
        fill_bar_idx: int | None = None,
        order_submission_time: Any | None = None,
        decision_bar_idx: int | None = None,
        signal_bar_idx: int | None = None,
    ) -> None:
        """Attempt a flip or new entry through explicit execution stages."""
        fill_bid, fill_ask, fill_time, execution_meta = self._resolve_entry_execution_quote(
            fill_bid, fill_ask, fill_time, order_submission_time
        )
        self._close_opposing_position(
            discrete_action,
            fill_bid,
            fill_ask,
            fill_time,
            bar_time,
            order_submission_time,
            session_id,
            execution_meta,
        )
        if self.position is not None or discrete_action not in (1, -1):
            return
        entry_price, sl_price, tp_price, lots, has_levels, placed = self._build_entry_levels(
            discrete_action, risk_frac, rr_ratio, feat_row, fill_bid, fill_ask
        )
        if not has_levels:
            return
        if not self._entry_risk_capacity_allows(entry_price, sl_price, lots, has_levels):
            return
        entry_equity, opened = self._open_entry_position(
            discrete_action, fill_bid, fill_ask, lots, sl_price
        )
        if not opened:
            return
        entry_level, sweep_delay, signal_bar, signal_time = self._initialize_open_position(
            discrete_action,
            risk_frac,
            rr_ratio,
            entry_price,
            sl_price,
            tp_price,
            placed,
            feat_row,
            fill_time,
            session_id,
            entry_candidate,
            bar_time,
            signal_bar_idx,
        )
        self._append_open_trade(
            discrete_action=discrete_action,
            risk_frac=risk_frac,
            bar=bar,
            feat_row=feat_row,
            bar_time=bar_time,
            session_id=session_id,
            fill_time=fill_time,
            sl_price=sl_price,
            tp_price=tp_price,
            placed=placed,
            entry_price=entry_price,
            entry_equity=entry_equity,
            execution_meta=execution_meta,
            entry_candidate=entry_candidate,
            fill_bar_idx=fill_bar_idx,
            order_submission_time=order_submission_time,
            decision_bar_idx=decision_bar_idx,
            signal_bar=signal_bar,
            signal_time=signal_time,
            entry_level=entry_level,
            sweep_delay=sweep_delay,
        )
